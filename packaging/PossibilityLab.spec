# Build on the target OS (or Windows Python under Wine).
from pathlib import Path
import sys,os
from PyInstaller.utils.hooks import copy_metadata
root=Path(SPECPATH).parent
resources=[]
def resource(path):resources.append((str(root/path),str(Path(path).parent)))
resources.append((str(root/'ui/lab'),'ui/lab'))
resource('representation_results/cnn.pt')
for dist in ['torch','numpy','PySide6','PySide6_Essentials','PySide6_Addons','shiboken6']:
 resources+=copy_metadata(dist)
for path in ['gate_results/metrics.json','decoder_results/metrics.json']:resource(path)
for seed in [317,419,523]:
 for path in [f'gate_results/seed_{seed}/acoustic_tokenizer.pt',f'gate_results/seed_{seed}/local.pt',f'gate_results/seed_{seed}/gate.pt',f'encoder_results/seed_{seed}/adapters.pt',f'alignment_results/seed_{seed}/original_joint.pt',f'alignment_results/seed_{seed}/original_aligned.pt',f'decoder_results/seed_{seed}/recognition.pt']:resource(path)
binaries=[]
if sys.platform=='win32':
 for name in ['msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','msvcp140_atomic_wait.dll','msvcp140_codecvt_ids.dll','vcruntime140.dll','vcruntime140_1.dll','vcruntime140_threads.dll','concrt140.dll']:
  dll=Path(os.environ['WINDIR'])/'System32'/name
  if dll.exists():binaries.append((str(dll),'.'))
a=Analysis([str(root/'packaging/desktop_entry.py')],pathex=[str(root)],binaries=binaries,datas=resources,hiddenimports=[],hookspath=[],excludes=['tkinter','matplotlib','scipy','pandas','IPython','pytest','tensorboard'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='PossibilityLab',icon=str(root/'packaging/PossibilityLab.ico') if sys.platform=='win32' else None,debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=sys.platform!='win32',disable_windowed_traceback=False)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='PossibilityLab')
