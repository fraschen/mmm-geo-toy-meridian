"""Downloads the raw Meridian geo dataset and verifies its SHA-256.

Idempotent: if the file exists with the expected checksum, nothing happens.
The raw file is never modified after download.
"""

import sys
import tempfile
import urllib.request

from common import DATA_SHA256, DATA_URL, RAW_CSV, sha256


def main():
  if RAW_CSV.exists():
    if sha256(RAW_CSV) == DATA_SHA256:
      print(f"{RAW_CSV} present, checksum OK")
      return
    sys.exit(f"{RAW_CSV} exists but its checksum does not match; refusing to overwrite.")
  RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=RAW_CSV.parent, delete=False) as tmp:
    with urllib.request.urlopen(DATA_URL, timeout=60) as r:
      tmp.write(r.read())
  from pathlib import Path
  got = sha256(Path(tmp.name))
  if got != DATA_SHA256:
    Path(tmp.name).unlink()
    sys.exit(f"Checksum mismatch: expected {DATA_SHA256}, got {got}")
  Path(tmp.name).rename(RAW_CSV)
  RAW_CSV.chmod(0o444)
  print(f"Downloaded {RAW_CSV} (sha256 {got})")


if __name__ == "__main__":
  main()
