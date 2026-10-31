import subprocess,sys
subprocess.run([sys.executable,'-m','pip','uninstall','-y','PySide6_WebEngine','PySide6_Pdf'],check=True)
subprocess.run([sys.executable,'-m','pip','install','--force-reinstall','PySide6==6.8.3','torch==2.6.0+cpu','numpy==1.26.4','--extra-index-url','https://download.pytorch.org/whl/cpu','--no-warn-script-location'],check=True)
