"""Copy the author tree into a writable run area and fix absolute paths.

The shipped .prm files reference ``/calculate/davis-viscoelasticity`` (the
authors' cluster path).  We never edit the downloaded tree; this script makes
a private copy under ``runs/author_tree`` and rewrites that prefix to the copy
location.  Nothing from the copied tree is imported by this script.

Run:  python -I prepare_author_run.py --source <extract>/davis-viscoelasticity-1.0.0 \
                                      --dest runs/author_tree [--force]
"""

from __future__ import annotations

import argparse
import os
import shutil

AUTHOR_PREFIX = "/calculate/davis-viscoelasticity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.source)
    dst = os.path.abspath(args.dest)
    if os.path.exists(dst):
        if not args.force:
            print(f"{dst} exists; pass --force to recreate")
            return
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True,
                    ignore=shutil.ignore_patterns("build", "output"))
    print(f"copied {src} -> {dst}")

    # MATLAB and CMake never need the old prefix in the copied tree
    changed = []
    for root, _dirs, files in os.walk(dst):
        for name in files:
            if not name.endswith((".prm", ".prm.in")):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if AUTHOR_PREFIX not in text:
                continue
            text = text.replace(AUTHOR_PREFIX, dst)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            changed.append(os.path.relpath(path, dst))
    for c in changed:
        print("patched", c)
    print(f"{len(changed)} files patched")


if __name__ == "__main__":
    main()
