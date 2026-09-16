"""Run every test module in this directory, so adding a file needs no wiring.

Each module is a script that exits non-zero on failure, so they are run as
subprocesses: one failing suite must not stop the others from reporting.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

failed = []
for path in sorted(HERE.glob("test_*.py")):
    print(f"--- {path.name} ---", flush=True)
    if subprocess.run([sys.executable, str(path)]).returncode != 0:
        failed.append(path.name)

print()
if failed:
    print(f"FAILED: {', '.join(failed)}")
    raise SystemExit(1)
print("all suites passed")
