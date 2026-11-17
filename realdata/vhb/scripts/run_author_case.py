"""Drive the author's C++ solver for one identification case (Stage B).

The MATLAB driver only does bookkeeping around the solver: it sets
``Identification stage type`` and ``Experiment name`` in the .prm, calls the
executable, and reads the files the solver writes into
``calibration/input``.  This script performs the same steps from Python so
that the native forward model and its sensitivities can be used without
MATLAB.

Only the private copy under ``runs/author_tree`` is edited; the downloaded
tree stays untouched.

Run:  python -I run_author_case.py --solver <exe> --prm <prm> \
        --stage PREPARE --experiment <rel/path/> [--log file]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time


def set_key(path, key, value):
    # [ \t] rather than \s: the latter would let the value swallow the rest of
    # the line's newline and eat the following line.
    pat = re.compile(r"(?m)^([ \t]*set[ \t]+" + re.escape(key)
                     + r"[ \t]*=[ \t]*).*$")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    new, n = pat.subn(lambda m: m.group(1) + value, text)
    if n != 1:
        raise RuntimeError(f"{key}: {n} matches in {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--prm", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["PREPARE", "EXECUTE", "NONE"])
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra KEY=VALUE prm settings")
    args = ap.parse_args()

    set_key(args.prm, "Identification stage type", args.stage)
    if args.experiment is not None:
        set_key(args.prm, "Experiment name", args.experiment)
    for item in args.extra:
        key, value = item.split("=", 1)
        set_key(args.prm, key, value)
    with open(args.prm, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"(?m)^[ \t]*set[ \t]+Experiment name[ \t]*=[ \t]*(.*)$", text)
    exp = m.group(1).strip() if m else ""

    log = open(args.log, "w") if args.log else sys.stdout
    t0 = time.time()
    print(f"stage {args.stage} experiment '{exp}'", file=log)
    proc = subprocess.run([args.solver, args.prm], stdout=log,
                          stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"exit {proc.returncode} after {dt:.1f} s", file=log)
    if args.log:
        log.close()
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
