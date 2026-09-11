#!/usr/bin/env python
import argparse
import logging
import struct
import sys
import zlib
from pathlib import Path

ZIP_LOCAL_SIG = 0x04034B50
SIG_KEY, SIG_INC = 0x89, 0x01   # fixed keystream used only for the signature


# --------------------------------------------------------------------------- #
# Core crypto primitive
# --------------------------------------------------------------------------- #

def rolling_xor(buf: bytes, key: int, inc: int):
    """
    Decrypt (or encrypt - it's symmetric) `buf` with an additive rolling XOR
    keystream: out[i] = buf[i] ^ key ; key = (key + inc) & 0xFF, repeated.

    Returns (decrypted_bytes, key_after). Note: for this format each field
    (header/name/extra) is decrypted starting FRESH at the same per-entry
    seed, not chained - key_after is unused by the caller but kept for
    completeness/reuse.
    """
    out = bytearray(len(buf))
    k = key & 0xFF
    inc &= 0xFF
    for i, b in enumerate(buf):
        out[i] = b ^ k
        k = (k + inc) & 0xFF
    return bytes(out), k


# --------------------------------------------------------------------------- #
# Logging setup: full detail always goes to the debug log file; the console
# gets a concise per-entry summary by default, or everything with -v/--verbose.
# --------------------------------------------------------------------------- #

def setup_logging(debug_log_path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("fun_extract")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(debug_log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def sanitize_name(name: str) -> str:
    """Defang path separators / traversal so extraction can't write outside out_dir."""
    name = name.replace("\\", "/")
    parts = [p for p in name.split("/") if p not in ("", ".", "..")]
    return "/".join(parts) if parts else "_unnamed_"


# --------------------------------------------------------------------------- #
# Main parsing / extraction loop
# --------------------------------------------------------------------------- #

def extract(archive_path: Path, out_dir: Path, logger: logging.Logger,
            inflate: bool = True, dry_run: bool = False):
    data = archive_path.read_bytes()
    full_size = len(data)
    limit = full_size - 2  # mirrors the reference script's "FULLSIZE -= 2"

    logger.info(f"Archive : {archive_path}  ({full_size} bytes)")
    logger.debug(f"full_size = {full_size}, scan_limit (full_size-2) = {limit}")
    if not dry_run:
        logger.info(f"Output  : {out_dir}")

    off = 0
    entry_index = 0
    manifest = []

    while off < limit:
        entry_index += 1
        entry_start = off
        logger.debug("-" * 78)
        logger.debug(f"Entry #{entry_index}  @ file offset 0x{off:08X} ({off})")

        # ---- 1. signature (fixed key) ----
        if off + 4 > full_size:
            logger.warning(f"Truncated: not enough bytes for signature at 0x{off:08X}.")
            break
        raw_sig = data[off:off + 4]
        dec_sig, _ = rolling_xor(raw_sig, SIG_KEY, SIG_INC)
        sig_val = struct.unpack("<I", dec_sig)[0]
        logger.debug(f"  raw signature      : {raw_sig.hex(' ')}")
        logger.debug(f"  decrypted signature: {dec_sig.hex(' ')}  -> 0x{sig_val:08X}")

        if sig_val != ZIP_LOCAL_SIG:
            logger.info(f"Signature mismatch at 0x{off:08X} "
                        f"(got 0x{sig_val:08X}, expected 0x{ZIP_LOCAL_SIG:08X}) "
                        f"- stopping scan. {entry_index - 1} entr"
                        f"{'y' if entry_index - 1 == 1 else 'ies'} found.")
            break
        pos = off + 4

        # ---- 2. raw 2-byte key seed for this entry ----
        if pos + 2 > full_size:
            logger.warning("Truncated: missing 2-byte key seed.")
            break
        key_byte, inc_byte = data[pos], data[pos + 1]
        logger.debug(f"  key seed (raw, unencrypted) @0x{pos:08X}: "
                     f"key=0x{key_byte:02X} inc=0x{inc_byte:02X}")
        pos += 2

        # ---- 3. 24-byte mini local header ----
        if pos + 0x18 > full_size:
            logger.warning("Truncated: missing 24-byte header block.")
            break
        raw_hdr = data[pos:pos + 0x18]
        dec_hdr, _ = rolling_xor(raw_hdr, key_byte, inc_byte)
        (flag, method, modtime, moddate,
         crc, comp_size, uncomp_size,
         name_len, extra_len) = struct.unpack("<4H3I2H", dec_hdr)

        logger.debug(f"  raw header (24B)   : {raw_hdr.hex(' ')}")
        logger.debug(f"  decrypted header   : {dec_hdr.hex(' ')}")
        logger.debug(f"  flag=0x{flag:04X} method={method} "
                     f"modtime=0x{modtime:04X} moddate=0x{moddate:04X}")
        logger.debug(f"  crc32=0x{crc:08X} comp_size={comp_size} "
                     f"uncomp_size={uncomp_size} name_len={name_len} extra_len={extra_len}")
        pos += 0x18

        # ---- 4. filename (keystream continues) ----
        if pos + name_len > full_size:
            logger.warning("Truncated: missing filename bytes.")
            break
        raw_name = data[pos:pos + name_len]
        dec_name, _ = rolling_xor(raw_name, key_byte, inc_byte)
        name = dec_name.decode("latin-1", errors="replace")
        safe_name = sanitize_name(name)
        logger.debug(f"  raw name bytes     : {raw_name.hex(' ')}")
        logger.debug(f"  decrypted name     : {name!r}  -> sanitized: {safe_name!r}")
        pos += name_len

        # ---- 5. extra field, optional (fresh key again) ----
        if extra_len > 0:
            if pos + extra_len > full_size:
                logger.warning("Truncated: missing extra-field bytes.")
                break
            raw_extra = data[pos:pos + extra_len]
            dec_extra, _ = rolling_xor(raw_extra, key_byte, inc_byte)
            logger.debug(f"  raw extra bytes    : {raw_extra.hex(' ')}")
            logger.debug(f"  decrypted extra    : {dec_extra.hex(' ')}")
            pos += extra_len

        # ---- 6/7. payload (unencrypted) + the +2 trailing quirk ----
        data_start = pos
        stream_len = comp_size            # actual DEFLATE/stored byte count
        entry_span = stream_len           # next entry starts right after this

        if data_start + entry_span > full_size:
            avail = max(0, full_size - data_start)
            logger.warning(f"Entry '{safe_name}' claims {entry_span} data bytes "
                           f"but only {avail} remain - truncating.")
            entry_span = avail
            stream_len = avail

        payload = data[data_start:data_start + stream_len]
        method_name = {0: "stored", 8: "deflate"}.get(method, f"unknown(0x{method:04X})")

        logger.debug(f"  payload @0x{data_start:08X}, stream_len={stream_len}, "
                     f"span={entry_span}")
        logger.info(f"[{entry_index:5d}] {safe_name}  "
                   f"({method_name}, comp={stream_len}, uncomp={uncomp_size}, "
                   f"crc=0x{crc:08X})")

        manifest.append({
            "index": entry_index, "name": safe_name, "header_offset": entry_start,
            "data_offset": data_start, "method": method_name, "flag": flag,
            "comp_size": stream_len, "uncomp_size": uncomp_size, "crc32": crc,
        })

        if not dry_run:
            out_path = out_dir / safe_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            final_bytes = payload

            if method == 8 and inflate and payload:
                try:
                    final_bytes = zlib.decompressobj(wbits=-15).decompress(payload)
                    logger.debug(f"  inflate OK: {len(payload)} -> {len(final_bytes)} bytes")
                    if uncomp_size and len(final_bytes) != uncomp_size:
                        logger.warning(f"  inflated size {len(final_bytes)} != "
                                      f"header uncomp_size {uncomp_size} for '{safe_name}'")
                except zlib.error as e:
                    logger.error(f"  inflate FAILED for '{safe_name}': {e} "
                                f"- writing raw compressed bytes instead")
                    final_bytes = payload
            elif method == 0:
                logger.debug("  stored (no compression) - writing as-is")
            elif method != 8:
                logger.warning(f"  unrecognized method {method} for '{safe_name}' "
                              f"- writing raw bytes")

            try:
                out_path.write_bytes(final_bytes)
                logger.debug(f"  wrote {len(final_bytes)} bytes -> {out_path}")
            except OSError as e:
                logger.error(f"  FAILED to write '{out_path}': {e}")

        off = data_start + entry_span

    logger.info("-" * 78)
    logger.info(f"Finished: {len(manifest)} entr"
               f"{'y' if len(manifest) == 1 else 'ies'} parsed"
               f"{'' if dry_run else ' and extracted'}.")
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Extract FUN Labs .fun archives (XOR-obfuscated ZIP local headers "
                    "+ standard DEFLATE payloads).")
    ap.add_argument("archive", type=Path, help="Path to the .fun archive file")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output directory (default: <archive_stem>_extracted)")
    ap.add_argument("--log", type=Path, default=None,
                    help="Debug log file path (default: <archive_stem>_debug.log)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print full per-byte debug detail to the console too "
                        "(the log file always gets full detail regardless)")
    ap.add_argument("--no-inflate", action="store_true",
                    help="Write the raw DEFLATE/stored bytes without decompressing")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and log the archive only; do not write any files")
    args = ap.parse_args()

    if not args.archive.is_file():
        print(f"error: '{args.archive}' not found", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or Path(f"{args.archive.stem}_extracted")
    log_path = args.log or Path(f"{args.archive.stem}_debug.log")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_path, args.verbose)
    logger.debug(f"Command line: {' '.join(sys.argv)}")
    logger.debug(f"Options: out={out_dir} log={log_path} verbose={args.verbose} "
                f"no_inflate={args.no_inflate} dry_run={args.dry_run}")

    try:
        manifest = extract(args.archive, out_dir, logger,
                          inflate=not args.no_inflate, dry_run=args.dry_run)
    except Exception:
        logger.exception("Unhandled error during extraction")
        raise

    logger.info(f"Debug log written to: {log_path}")
    if not args.dry_run:
        logger.info(f"Files extracted to  : {out_dir}")
    logger.info(f"Total entries        : {len(manifest)}")


if __name__ == "__main__":
    main()
