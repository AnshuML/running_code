"""Launcher: run accuracy from project root. Script lives in running_code/."""
import os
import subprocess
import sys

root = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(root, "running_code", "accuracy_by_product.py")
if not os.path.exists(script):
    print(f"Error: {script} not found", file=sys.stderr)
    sys.exit(1)
sys.exit(subprocess.run(
    [sys.executable, script] + sys.argv[1:],
    cwd=os.path.join(root, "running_code")
).returncode)
