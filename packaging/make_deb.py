from pathlib import Path
import shutil,subprocess
root=Path(__file__).resolve().parents[1];stage=root/'packaging/deb-stage'
(stage/'DEBIAN').mkdir(parents=True,exist_ok=True);(stage/'opt').mkdir(exist_ok=True)
subprocess.run(['cp','-a','--reflink=auto',str(root/'packaging/dist-linux/PossibilityLab'),str(stage/'opt/possibility-lab')],check=True)
size=sum(p.stat().st_size for p in (stage/'opt').rglob('*') if p.is_file())//1024
(stage/'DEBIAN/control').write_text(f'''Package: possibility-lab
Version: 1.0.0
Section: education
Priority: optional
Architecture: amd64
Maintainer: Possibility Lab
Installed-Size: {size}
Depends: libc6 (>= 2.43)
Description: Offline Parallel Possibility Engine presentation demo
 Includes the Python runtime, embedded browser, trained models and UI.
 No model downloads, Python installation or network connection are needed.
 Built and tested for Ubuntu 26.04 x86-64.
''')
apps=stage/'usr/share/applications';apps.mkdir(parents=True,exist_ok=True)
(apps/'possibility-lab.desktop').write_text('''[Desktop Entry]
Type=Application
Name=Possibility Lab
Comment=Offline Parallel Possibility Engine
Exec=/opt/possibility-lab/PossibilityLab
Icon=/opt/possibility-lab/_internal/ui/lab/icon.svg
Terminal=false
Categories=Education;Science;
''')
subprocess.run(['dpkg-deb','--build','--root-owner-group','--compression-level=5',str(stage),str(root/'releases/PossibilityLab-Linux-amd64.deb')],check=True)
