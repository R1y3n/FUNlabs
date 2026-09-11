# FUNlabs
A script to decrypt .fun proprietary files by FUNlabs Studio

# debug: 
- Console (default): one line per entry — name, method, compressed/uncompressed size, CRC.
- `-v` / `--verbose`: same full per-byte detail as the log, also on console.
- `--log`: full debug detail (timestamped) always written to a log file regardless of `-v` — defaults to `<archive>_debug.log`.

# Use :
```
python fun_extract.py archive.fun -v
```
