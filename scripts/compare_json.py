"""Compares two JSON files numerically (used to verify EDA reproduces in Docker)."""

import json
import math
import sys


def walk(a, b, path=""):
  diffs = []
  if isinstance(a, dict):
    for k in set(a) | set(b):
      if k not in a or k not in b:
        diffs.append(f"{path}/{k}: missing")
      else:
        diffs += walk(a[k], b[k], f"{path}/{k}")
  elif isinstance(a, list):
    if len(a) != len(b):
      return [f"{path}: length {len(a)} vs {len(b)}"]
    for i, (x, y) in enumerate(zip(a, b)):
      diffs += walk(x, y, f"{path}[{i}]")
  elif isinstance(a, float) or isinstance(b, float):
    if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9):
      diffs.append(f"{path}: {a} vs {b}")
  elif a != b:
    diffs.append(f"{path}: {a} vs {b}")
  return diffs


def main():
  a, b = (json.load(open(p)) for p in sys.argv[1:3])
  diffs = walk(a, b)
  for d in diffs[:20]:
    print(d)
  print(f"{len(diffs)} differences")
  sys.exit(1 if diffs else 0)


if __name__ == "__main__":
  main()
