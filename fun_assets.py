#!/usr/bin/env python3
import argparse
import json
import logging
import re
import shutil
import struct
import sys
from pathlib import Path

MAGIC = b"BiN\x00"

STANDARD_EXTS = {".tga", ".lua", ".log"}
TEXT_EXTS = {".model", ".script"}
PROPRIETARY_MESH_EXTS = {".obj"}
PROPRIETARY_ANIM_EXTS = {".anim"}
UNSUPPORTED_EXTS = {".bsp"}

ANIM_KEY_FLOATS = {"Pos": 4, "Rot": 5, "PosRot": 8}
ANIM_KEY_BYTES = {"Pos": 16, "Rot": 20, "PosRot": 32, "PosRotVis": 34}
# PosRotVis differs from the others: time is stored as u32 (not f32), then
# f32[3] pos, f32[4] quat, then a trailing u16 (observed always 0 in real
# files - likely a visibility flag). Decoded generically below.


# --------------------------------------------------------------------------- #
# Shared binary helpers
# --------------------------------------------------------------------------- #

def read_pstr(d: bytes, off: int):
    n = d[off]
    s = d[off + 1:off + 1 + n].decode("latin-1")
    return s, off + 1 + n


NODE_EXTRA_FLOATS = {3: 3, 4: 3}  # node types 3 and 4 carry an extra f32[3]
                                    # (observed as a scale vector, sometimes
                                    # non-uniform/negative - e.g. mirrored
                                    # finger helpers, muzzle-flash effects)


def read_node(d: bytes, off: int):
    parent, off = read_pstr(d, off)
    name, off = read_pstr(d, off)
    ntype = struct.unpack_from("<H", d, off)[0]; off += 2
    pos = struct.unpack_from("<3f", d, off); off += 12
    quat = struct.unpack_from("<4f", d, off); off += 16
    scale = None
    nf = NODE_EXTRA_FLOATS.get(ntype, 0)
    if nf:
        scale = struct.unpack_from(f"<{nf}f", d, off); off += nf * 4
    node = {"parent": parent, "name": name, "type": ntype,
            "pos": pos, "quat": quat}
    if scale is not None:
        node["scale"] = scale
    return node, off


# --------------------------------------------------------------------------- #
# .OBJ (mesh) conversion
# --------------------------------------------------------------------------- #

def try_parse_simple_mesh(d: bytes, off: int, owner_name: str, logger, ctx: str):
    """
    Attempt to parse a geometry section for a single static mesh starting at
    `off`. Returns (materials, new_off) on success, or None if the shape
    doesn't match (caller should fall back to hierarchy-only export).
    """
    try:
        name, o2 = read_pstr(d, off)
        if name != owner_name:
            logger.debug(f"  [{ctx}] geometry owner mismatch: expected "
                         f"'{owner_name}', got '{name}' - not a simple mesh")
            return None
        mat_count = struct.unpack_from("<H", d, o2)[0]; o2 += 2
        if not (0 < mat_count <= 64):
            logger.debug(f"  [{ctx}] implausible material_count={mat_count} - "
                         f"not a simple mesh")
            return None
        materials = []
        for mi in range(mat_count):
            matname, o2 = read_pstr(d, o2)
            vcount = struct.unpack_from("<H", d, o2)[0]; o2 += 2
            if o2 + vcount * 32 > len(d):
                logger.debug(f"  [{ctx}] vertex block overruns file - not a simple mesh")
                return None
            verts = []
            for vi in range(vcount):
                vals = struct.unpack_from("<8f", d, o2); o2 += 32
                verts.append(vals)
            tcount = struct.unpack_from("<H", d, o2)[0]; o2 += 2
            if o2 + tcount * 6 > len(d):
                logger.debug(f"  [{ctx}] triangle block overruns file - not a simple mesh")
                return None
            tris = struct.unpack_from(f"<{tcount * 3}H", d, o2); o2 += tcount * 3 * 2
            materials.append({"material": matname, "vertices": verts,
                             "triangles": [tris[i:i + 3] for i in range(0, len(tris), 3)]})
            logger.debug(f"  [{ctx}] material '{matname}': {vcount} verts, {tcount} tris")
        return materials, o2
    except (IndexError, struct.error) as e:
        logger.debug(f"  [{ctx}] exception while probing simple mesh shape: {e}")
        return None


def write_wavefront_obj(out_path: Path, mesh_name: str, materials):
    lines = [f"# converted from proprietary .OBJ mesh container ({mesh_name})", ""]
    v_offset = 0
    for mat in materials:
        lines.append(f"# material: {mat['material']}")
        lines.append(f"usemtl {mat['material']}")
        for (x, y, z, nx, ny, nz, u, v) in mat["vertices"]:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        for (x, y, z, nx, ny, nz, u, v) in mat["vertices"]:
            n = (nx, ny, nz)
            mag = (n[0]**2 + n[1]**2 + n[2]**2) ** 0.5
            if mag > 1e-8:
                n = (n[0]/mag, n[1]/mag, n[2]/mag)
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        for (x, y, z, nx, ny, nz, u, v) in mat["vertices"]:
            lines.append(f"vt {u:.6f} {1.0 - v:.6f}")
        for tri in mat["triangles"]:
            idxs = [i + 1 + v_offset for i in tri]
            face = " ".join(f"{i}/{i}/{i}" for i in idxs)
            lines.append(f"f {face}")
        v_offset += len(mat["vertices"])
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def convert_obj_file(path: Path, out_path: Path, logger: logging.Logger):
    d = path.read_bytes()
    if d[:4] != MAGIC:
        logger.warning(f"{path}: missing 'BiN' magic - skipping (not this format)")
        return "skipped-not-bin"

    off = 4
    node_count = struct.unpack_from("<H", d, off)[0]; off += 2
    nodes = []
    for i in range(node_count):
        node, off = read_node(d, off)
        nodes.append(node)
    logger.debug(f"{path.name}: parsed {node_count} hierarchy nodes")

    mesh_nodes = [n for n in nodes if n["type"] == 1]

    geometry = None
    if len(mesh_nodes) == 1 and off < len(d):
        result = try_parse_simple_mesh(d, off, mesh_nodes[0]["name"], logger, path.name)
        if result is not None:
            materials, end_off = result
            if end_off == len(d):
                geometry = materials
                logger.info(f"{path.name}: simple static mesh - full geometry "
                           f"converted ({sum(len(m['vertices']) for m in materials)} "
                           f"verts, {sum(len(m['triangles']) for m in materials)} tris)")
            else:
                logger.debug(f"{path.name}: mesh parse consumed {end_off}/{len(d)} "
                             f"bytes (leftover {len(d)-end_off}) - discarding, "
                             f"treating as non-simple")

    hierarchy = {"nodes": nodes, "source_file": str(path)}

    if geometry is not None:
        obj_out = out_path.with_suffix(".obj")
        write_wavefront_obj(obj_out, mesh_nodes[0]["name"], geometry)
        logger.info(f"  -> wrote {obj_out}")
        return "converted"
    else:
        json_out = out_path.with_suffix(".hierarchy.json")
        json_out.write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")
        reason = ("no mesh-type node found" if not mesh_nodes else
                 f"{len(mesh_nodes)} mesh nodes (skinned/multi-part - "
                 f"geometry format not yet decoded)")
        logger.info(f"{path.name}: {reason} - exported hierarchy only -> {json_out}")
        return "hierarchy-only"


# --------------------------------------------------------------------------- #
# .ANIM conversion
# --------------------------------------------------------------------------- #

def decode_keyframe(d: bytes, off: int, ttype: str):
    """Decode one keyframe of the given track type. Returns (key_dict, new_off)."""
    if ttype == "PosRotVis":
        time_int = struct.unpack_from("<I", d, off)[0]
        pos = struct.unpack_from("<3f", d, off + 4)
        quat = struct.unpack_from("<4f", d, off + 16)
        vis = struct.unpack_from("<H", d, off + 32)[0]
        return {"time": time_int, "pos": pos, "quat": quat, "vis": vis}, off + 34
    elif ttype == "Pos":
        vals = struct.unpack_from("<4f", d, off)
        return {"time": vals[0], "pos": vals[1:4]}, off + 16
    elif ttype == "Rot":
        vals = struct.unpack_from("<5f", d, off)
        return {"time": vals[0], "quat": vals[1:5]}, off + 20
    elif ttype == "PosRot":
        vals = struct.unpack_from("<8f", d, off)
        return {"time": vals[0], "pos": vals[1:4], "quat": vals[4:8]}, off + 32
    else:
        raise ValueError(f"unhandled keyframe track type '{ttype}'")


def convert_anim_file(path: Path, out_path: Path, logger: logging.Logger):
    d = path.read_bytes()
    if d[:4] != MAGIC:
        logger.warning(f"{path}: missing 'BiN' magic - skipping (not this format)")
        return "skipped-not-bin"

    off = 4
    track_count = struct.unpack_from("<H", d, off)[0]; off += 2
    tracks = []
    for i in range(track_count):
        cls, off = read_pstr(d, off)

        if cls == "AnimCtrl_KeyFrame":
            ttype, off = read_pstr(d, off)
            bone, off = read_pstr(d, off)
            nkeys = struct.unpack_from("<H", d, off)[0]; off += 2
            if ttype not in ANIM_KEY_BYTES:
                logger.warning(f"{path.name}: unrecognized keyframe track type "
                              f"'{ttype}' on track {i} (bone '{bone}') - "
                              f"stopping, file needs further RE")
                return "unrecognized-track-type"
            if off + nkeys * ANIM_KEY_BYTES[ttype] > len(d):
                logger.warning(f"{path.name}: track {i} ('{ttype}'/{bone}) key "
                              f"data overruns file - aborting conversion")
                return "corrupt-or-mismatched"
            keys = []
            for k in range(nkeys):
                key, off = decode_keyframe(d, off, ttype)
                keys.append(key)
            tracks.append({"class": cls, "type": ttype, "bone": bone, "keys": keys})
            logger.debug(f"  track {i}: class={cls} type={ttype} bone={bone} "
                         f"keys={nkeys}")

        elif cls == "AnimCtrl_Texture":
            # Sprite/UV texture-swap animation (e.g. eye-blink effects via
            # texture-atlas frame swapping). name/frame_count/key_count/
            # tex_ids/times are fully understood; the per-frame UV/blend
            # payload that follows is NOT fully decoded yet, so it is kept
            # as an opaque blob (only safe when this is the LAST track in
            # the file, which is true for every real instance found so far -
            # all such files have exactly 1 track).
            name, off = read_pstr(d, off)
            frame_count = struct.unpack_from("<H", d, off)[0]; off += 2
            key_count = struct.unpack_from("<H", d, off)[0]; off += 2
            tex_ids = struct.unpack_from(f"<{key_count}H", d, off); off += key_count * 2
            times = struct.unpack_from(f"<{key_count}f", d, off); off += key_count * 4
            if i != track_count - 1:
                logger.warning(f"{path.name}: AnimCtrl_Texture track {i} is not "
                              f"the last track - can't safely locate the next "
                              f"track without decoding its payload, aborting")
                return "texture-track-not-last-unsupported"
            payload = d[off:]
            off = len(d)
            tracks.append({"class": cls, "type": "Texture", "name": name,
                          "frame_count": frame_count, "tex_ids": list(tex_ids),
                          "times": list(times),
                          "undecoded_payload_bytes": len(payload)})
            logger.info(f"{path.name}: AnimCtrl_Texture track '{name}' - "
                       f"frame_count/tex_ids/times decoded, "
                       f"{len(payload)} bytes of per-frame UV/blend data left "
                       f"undecoded (not yet reverse-engineered)")

        else:
            logger.warning(f"{path.name}: unrecognized animation controller "
                          f"class '{cls}' on track {i} - stopping, file needs "
                          f"further RE (known classes: AnimCtrl_KeyFrame, "
                          f"AnimCtrl_Texture)")
            return "unrecognized-controller-class"

    if off != len(d):
        logger.warning(f"{path.name}: {len(d)-off} leftover bytes after parsing "
                      f"{track_count} tracks - output may be incomplete")

    json_out = out_path.with_suffix(".anim.json")
    json_out.write_text(json.dumps({"tracks": tracks, "source_file": str(path)},
                                   indent=2), encoding="utf-8")
    any_texture = any(t["class"] == "AnimCtrl_Texture" for t in tracks)
    logger.info(f"{path.name}: converted {track_count} tracks -> {json_out}")
    return "converted-partial" if any_texture else "converted"


def parse_skybox_face(d: bytes, off: int):
    """Parse one 176-byte quad-face record: 4 vertices x 10 floats
    (pos.xyz, normal.xyz, uv.xy, uv2.xy - second UV channel's exact purpose
    is unconfirmed, kept as-is) + 4 trailing bytes of small integers whose
    exact meaning (tessellation/neighbor indices?) is not resolved."""
    verts = []
    for v in range(4):
        vals = struct.unpack_from("<10f", d, off); off += 40
        verts.append({"pos": vals[0:3], "normal": vals[3:6], "uv": vals[6:8],
                     "uv2_unconfirmed": vals[8:10]})
    extra = struct.unpack_from("<4f", d, off); off += 16
    return {"vertices": verts, "extra_unconfirmed": list(extra)}, off


NAME_FIELD_RE = re.compile(rb'^([A-Za-z0-9_.\- ]{1,32}?)\x00+$')


def find_bsp_entity_start(d: bytes):
    search_from = 0
    while True:
        idx = d.find(b'entity("', search_from)
        if idx == -1:
            return len(d)
        p = idx
        while p > 0 and d[p - 1] == 0xFF:
            p -= 1
        if p >= 12:
            count, len_a, len_b = struct.unpack_from("<3I", d, p - 12)
            if len_b == len(d) - idx and count < 1000:
                return idx
        search_from = idx + 8


def find_bsp_named_records(d: bytes, end: int):
    """Find every [32-byte zero-padded name][u32 flag][0xFFFFFFFF] header up
    to `end`. Used both for the skybox's inline-geometry records and the
    separate material name table (same header shape, no geometry follows a
    table entry since the next header starts only 40 bytes later)."""
    positions = []
    search_from = 0
    while True:
        idx = d.find(b'\xff\xff\xff\xff', search_from)
        if idx == -1 or idx > end:
            break
        m = NAME_FIELD_RE.match(d[idx - 36:idx - 4])
        if m:
            positions.append((idx - 36, m.group(1).decode("ascii")))
        search_from = idx + 4
    return positions


def convert_bsp_file(path: Path, out_path: Path, logger: logging.Logger):
    """
    Partial .bsp support - three things are extracted with confidence:

    1. Entity/property script (see module docstring) -> <name>.entities.json
    2. Inline skybox quad geometry: a handful of "named record" headers
       (same 32-byte-name+flag+0xFFFFFFFF shape used by the material table)
       are followed directly by N x 176-byte quad-face blocks instead of
       nothing. Verified byte-exact (zero leftover bytes) on every skybox
       face across all 26 real archive .bsp files. -> <name>.skybox.obj
    3. Material name table (just names, no geometry - referenced by index
       elsewhere) -> included in <name>.entities.json as "materials".

    NOT extracted (see the module docstring's "known gaps" for detail): the
    actual architectural level geometry (walls/floors/props) and per-material
    data. Two large opaque regions were mapped structurally but not solved
    semantically:
      - A BVH/spatial-partition node array (repeating fixed-size records
        with sentinel +-9999998.0 "unset" bounding boxes and small integer
        fields that look like child/parent indices) sitting between the
        skybox and the material table.
      - A material-indexed chunk chain right after the material table
        ([u32 material_index][u32 length][length bytes]) whose per-material
        payload is small-integer/0xFF-heavy byte arrays - most likely a
        lighting or visibility lookup rather than raw vertex data, but this
        is not confirmed.
    Both were confirmed present (same shape, proportionally larger) in every
    one of the 26 real .bsp files, including the largest (32MB) gameplay
    levels, so they are not cutscene-specific artifacts - genuinely the bulk
    of each level's data, and a materially bigger reverse-engineering project
    than the containers solved so far.
    """
    d = path.read_bytes()
    entity_start = find_bsp_entity_start(d)
    records = find_bsp_named_records(d, entity_start)

    # Split records into "has inline geometry" (skybox) vs "name-only" (table)
    skybox_faces = {}
    materials = []
    for i, (pos, name) in enumerate(records):
        data_start = pos + 40
        data_end = records[i + 1][0] if i + 1 < len(records) else entity_start
        span = data_end - data_start
        if span > 0 and span % 176 == 0 and span <= 50000:
            n_faces = span // 176
            faces = []
            off = data_start
            for _ in range(n_faces):
                face, off = parse_skybox_face(d, off)
                faces.append(face)
            # Sanity check: every real skybox face has unit-length normals.
            # A material chunk's length can coincidentally be a multiple of
            # 176 bytes without being this geometry shape - check ALL faces
            # (not just the first, which can pass by chance) and reject the
            # whole record as a false positive if any fail.
            all_unit_normals = all(
                0.9 <= sum(c * c for c in vert["normal"]) ** 0.5 <= 1.1
                for face in faces for vert in face["vertices"])
            if all_unit_normals:
                skybox_faces[name] = faces
                logger.debug(f"  '{name}' @0x{pos:08X}: {n_faces} quad faces "
                            f"(inline geometry)")
            else:
                materials.append(name)
                logger.debug(f"  '{name}' @0x{pos:08X}: span is a multiple of "
                            f"176 bytes but not all normals are unit-length - "
                            f"false positive, treating as a material table "
                            f"entry instead")
        else:
            materials.append(name)

    # Entities (reuses the same logic as before)
    text = d[entity_start:].rstrip(b"\x00").decode("ascii", errors="replace")
    entities = []
    current = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'entity\("([^"]*)"\)', line)
        if m:
            current = {"class": m.group(1), "properties": {}}
            entities.append(current)
            continue
        m = re.match(r'property\("([^"]*)",\s*"(.*)"\)', line)
        if m and current is not None:
            current["properties"][m.group(1)] = m.group(2)

    json_out = out_path.with_suffix(".entities.json")
    json_out.write_text(json.dumps({
        "source_file": str(path),
        "entity_count": len(entities),
        "entities": entities,
        "materials_referenced_but_not_geometry_decoded": materials,
        "skybox_face_groups": list(skybox_faces.keys()),
        "note": ("Architectural level geometry (walls/floors/props) and "
                "per-material lighting/visibility data are NOT included - "
                "not yet reverse-engineered. See fun_asset_convert.py's "
                "convert_bsp_file docstring for what was mapped."),
    }, indent=2), encoding="utf-8")

    # Skybox geometry -> Wavefront OBJ, if any was found
    result = "entities-only"
    if skybox_faces:
        obj_lines = [f"# partial geometry from {path.name}: skybox faces only", ""]
        v_off = 0
        for name, faces in skybox_faces.items():
            obj_lines.append(f"g {name}")
            obj_lines.append(f"usemtl {name}")
            for face in faces:
                for vert in face["vertices"]:
                    x, y, z = vert["pos"]
                    obj_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
                for vert in face["vertices"]:
                    nx, ny, nz = vert["normal"]
                    obj_lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")
                for vert in face["vertices"]:
                    u, v = vert["uv"]
                    obj_lines.append(f"vt {u:.6f} {1.0-v:.6f}")
                i0, i1, i2, i3 = (v_off+1, v_off+2, v_off+3, v_off+4)
                obj_lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")
                obj_lines.append(f"f {i0}/{i0}/{i0} {i2}/{i2}/{i2} {i3}/{i3}/{i3}")
                v_off += 4
            obj_lines.append("")
        obj_out = out_path.with_suffix(".skybox.obj")
        obj_out.write_text("\n".join(obj_lines), encoding="utf-8")
        result = "skybox-and-entities"

    logger.info(f"{path.name}: {len(entities)} entities, {len(materials)} "
               f"materials (geometry not decoded), {len(skybox_faces)} skybox "
               f"face groups converted -> {json_out}"
               + (f" + {obj_out.name}" if skybox_faces else ""))

    raw_out = out_path.with_suffix(path.suffix + ".raw")
    shutil.copy2(path, raw_out)
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("fun_asset_convert")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def main():
    ap = argparse.ArgumentParser(
        description="Convert FUN Labs proprietary .OBJ/.ANIM assets to standard "
                    "formats (Wavefront .obj / .json). Leaves .tga/.lua/.model/"
                    ".script/.log untouched (already standard) and .bsp untouched "
                    "(format not yet reverse-engineered).")
    ap.add_argument("input_dir", type=Path, help="Extracted archive directory "
                    "(output of fun_extract.py)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output directory (default: <input_dir>_converted)")
    ap.add_argument("--log", type=Path, default=None,
                    help="Debug log path (default: <input_dir>_convert_debug.log)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show full debug detail on the console too")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        print(f"error: '{args.input_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out if args.out is not None else Path(f"{args.input_dir.name}_converted")
    log_path = args.log or Path(f"{args.input_dir.name}_convert_debug.log")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_path, args.verbose)
    logger.debug(f"Command line: {' '.join(sys.argv)}")

    stats = {}

    def bump(key):
        stats[key] = stats.get(key, 0) + 1

    for path in sorted(args.input_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(args.input_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()

        if ext in STANDARD_EXTS:
            shutil.copy2(path, out_path)
            logger.debug(f"{rel}: standard format ({ext}) - copied as-is")
            bump(f"standard{ext}")

        elif ext in TEXT_EXTS:
            shutil.copy2(path, out_path)
            logger.debug(f"{rel}: already plain text ({ext}) - copied as-is")
            bump(f"text{ext}")

        elif ext in PROPRIETARY_MESH_EXTS:
            logger.debug("-" * 78)
            try:
                result = convert_obj_file(path, out_path, logger)
            except (struct.error, IndexError, UnicodeDecodeError) as e:
                logger.error(f"{rel}: parse error ({e}) - file doesn't match "
                            f"the known .OBJ shape, skipping")
                result = "parse-error"
            bump(f"obj:{result}")

        elif ext in PROPRIETARY_ANIM_EXTS:
            logger.debug("-" * 78)
            try:
                result = convert_anim_file(path, out_path, logger)
            except (struct.error, IndexError, UnicodeDecodeError) as e:
                logger.error(f"{rel}: parse error ({e}) - file doesn't match "
                            f"the known .ANIM shape, skipping")
                result = "parse-error"
            if result not in ("converted", "converted-partial"):
                shutil.copy2(path, out_path)
                logger.debug(f"  -> copied raw .anim untouched as fallback")
            bump(f"anim:{result}")

        elif ext in UNSUPPORTED_EXTS:
            logger.debug("-" * 78)
            try:
                result = convert_bsp_file(path, out_path, logger)
            except Exception as e:
                logger.error(f"{rel}: unexpected error ({e}) - copying raw untouched")
                shutil.copy2(path, out_path)
                result = "error"
            bump(f"bsp:{result}")

        else:
            shutil.copy2(path, out_path)
            logger.debug(f"{rel}: unclassified extension '{ext}' - copied as-is")
            bump(f"unclassified{ext}")

    logger.info("=" * 78)
    logger.info("Summary:")
    for k in sorted(stats):
        logger.info(f"  {k:35s} {stats[k]}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Debug log: {log_path}")


if __name__ == "__main__":
    main()
