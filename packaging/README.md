# Offline desktop builds

End users need only the installer from `releases/`; these are developer notes.
The frozen application includes the trained checkpoints and an embedded Qt
browser. It starts a private loopback-only model service automatically and stops
it when the window closes. It never needs a separate Python/model installation.

## Linux x86-64

Build environment used: Ubuntu26.04, Python3.14, CPU-only PyTorch. Install the
pinned developer requirements from `requirements-linux.txt` in an isolated env.

```bash
.package-env/bin/python -m PyInstaller --noconfirm --distpath packaging/dist-linux --workpath packaging/work-linux packaging/PossibilityLab.spec
packaging/dist-linux/PossibilityLab/PossibilityLab --self-test
python3 packaging/add_notices.py
python3 packaging/make_linux_installer.py
python3 packaging/make_deb.py
```

The `.deb` explicitly requires glibc2.43. Broad compatibility with older Linux
releases is not claimed. An older-distro build environment is required before
lowering this requirement. The `.run` is the same payload, installed per user.

## Windows x86-64

The Windows toolchain uses official CPython3.12.10, the pinned Windows
requirements, PyInstaller and NSIS3.11. The required native Microsoft v14 runtime
libraries must be present in the build environment; the spec includes them in
the app so they do not need to be installed separately by the end user.
Builds here run the Windows Python toolchain through Wine10 on Linux. This
is a Windows build, not a Linux executable with an `.exe` suffix. Compatibility
checks under Wine are recorded separately from native Windows verification.

```text
python -m PyInstaller --noconfirm --distpath packaging/dist-windows --workpath packaging/work-windows packaging/PossibilityLab.spec
python packaging/prepare_windows_bundle.py
makensis packaging/launcher.nsi
packaging/dist-windows/PossibilityLab/PossibilityLab.exe --self-test
makensis packaging/installer.nsi
```

The NSIS installer creates Start menu and Desktop shortcuts and includes an
uninstaller. It installs per user without administrator access.

## Checks

`--self-test` runs all three models on real synthetic input and verifies code
shapes/probability normalization. `--smoke-ui` opens the real desktop view,
records layout/loaded-key count and captures a screenshot before exiting.
`verify_bundle.py` checks bundled checkpoint and UI hashes against source.
Neither these checks nor packaging establishes new recognition accuracy claims.
The two OS builds use different runtime versions for compatibility; both need
actual prediction and UI checks. Model weights are unchanged.

The Windows launcher enables software browser graphics automatically before
starting Engine.exe. This avoids GPU-driver requirements for the simple UI.
The model runtime is CPU-only on both platforms.
