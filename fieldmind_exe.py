"""Interactive entry point for a bundled Windows FieldMind-XPU Lite executable."""
from __future__ import annotations
import os
import sys
import traceback
from pathlib import Path

def choose_action():
    print("\nFieldMind-XPU FULL v0.2 - Windows (Python/OpenCV/PyTorch XPU embedded)")
    print("Observation is the default: no keys are pressed.")
    print("1. Check available compute")
    print("2. Simulate tactical decisions")
    print("3. Calibrate ball template")
    print("4. Preview live screen (OBSERVE ONLY)")
    print("5. Preview with bundled toy student (not expert-trained)")
    print("6. Help")
    print("0. Exit")
    return {
        "1": ["check"],
        "2": ["simulate", "--steps", "15"],
        "3": ["calibrate-ball", "--config", "configs/tps_ultimate.json"],
        "4": ["run", "--config", "configs/tps_ultimate.json"],
        "5": ["run", "--config", "configs/tps_ultimate.json", "--student", "models/toy_student.pt"],
        "6": ["--help"],
    }.get(input("Select 0-6: ").strip())

def prepare_config():
    if not getattr(sys, "frozen", False):
        return
    if not hasattr(sys, "_MEIPASS"):
        return
    import shutil
    for rel in ("configs/tps_ultimate.json", "models/toy_student.pt"):
        src = Path(sys._MEIPASS) / rel
        dst = Path.cwd() / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

def main():
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)
    prepare_config()
    interactive = len(sys.argv) == 1
    if interactive:
        cmd = choose_action()
        if cmd is None:
            return
        sys.argv = [sys.argv[0], *cmd]
    try:
        from fieldmind.cli import main as cli
        cli()
    except Exception:
        traceback.print_exc()
        if interactive:
            input("\nPress ENTER to close...")
        raise
    else:
        if interactive:
            input("\nPress ENTER to close...")

if __name__ == "__main__":
    main()
