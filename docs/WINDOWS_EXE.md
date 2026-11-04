# Windows EXE builds

This project has two different Windows executables:

**FieldMind-XPU-Lite.exe** is built by the `Windows EXE` GitHub Actions workflow using PyInstaller on Windows x64. The resulting artifact contains its Python runtime plus CPU screen/ball tracking dependencies. It is intended to run on a machine **without a separately installed Python interpreter**. See the Actions tab and download the `FieldMind-XPU-Lite-Windows-x64` artifact from a successful run.

**FieldMind-XPU-Launcher.exe** is a separately produced ~2 MB Windows-native launcher, useful for quick demos. It embeds the source files, has an offline native tactical simulation, and can set up a local Python 3.10/3.11 virtual environment for live vision. It does **not** bundle the Python interpreter or PyTorch.

Neither executable includes P2P weights or a pretrained expert soccer model. The standalone Lite PyInstaller build does **not** bundle torch or Intel XPU runtime and cannot train neural models. For Intel XPU training, use the source project with supported XPU PyTorch binaries.

No automatic Roblox input is enabled by default. TPS Ultimate Soccer's exact tool-specific keys are not yet verified. To observe ball tracking, use `calibrate-ball` / `run` (preview) after making the game ball visible. Use this only in allowed training situations.

How to build the standalone Lite EXE on Windows:

```powershell
py -3.11 -m pip install -e \".[vision]\" pyinstaller
pyinstaller --noconfirm --clean --onefile --console --name FieldMind-XPU-Lite --add-data \"configs/tps_ultimate.json;configs\" --collect-submodules mss --collect-submodules pydirectinput fieldmind_exe.py
.\dist\FieldMind-XPU-Lite.exe simulate --steps 3
```

Windows EXE build workflow: https://github.com/gaong5247-cmd/secret2/actions/workflows/build-windows-exe.yml
