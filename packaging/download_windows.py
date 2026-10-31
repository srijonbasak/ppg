from pathlib import Path
from urllib.request import urlretrieve
import zipfile
root=Path(__file__).resolve().parent
files={'python-3.12.10-embed-amd64.zip':'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip','get-pip.py':'https://bootstrap.pypa.io/get-pip.py','nsis-3.11.zip':'https://downloads.sourceforge.net/project/nsis/NSIS%203/3.11/nsis-3.11.zip'}
for name,url in files.items():
 p=root/'downloads'/name
 if not p.exists():
  print('Downloading',name,flush=True);urlretrieve(url,p)
 print(name,p.stat().st_size,flush=True)
win=root/'windows-python';win.mkdir(exist_ok=True)
with zipfile.ZipFile(root/'downloads/python-3.12.10-embed-amd64.zip') as z:z.extractall(win)
(win/'python312._pth').write_text('python312.zip\n.\nLib/site-packages\nimport site\n')
if (root/'downloads/nsis-3.11.zip').exists():
 with zipfile.ZipFile(root/'downloads/nsis-3.11.zip') as z:z.extractall(root/'nsis')
