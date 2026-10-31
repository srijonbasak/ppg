"""Install into a private Wine prefix and private test folder, never the host desktop."""
from pathlib import Path
import os,subprocess,json,hashlib
root=Path(__file__).resolve().parents[1]
prefix=root/'packaging/wine-clean-test';desktop=prefix/'drive_c/users/srijon/Desktop'
if desktop.is_symlink():desktop.unlink()
desktop.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['WINEPREFIX']=str(prefix);env['WINEDEBUG']='-all'
installer=root/'releases/PossibilityLab-Windows-x64-Setup.exe';target=root/'packaging/installed-windows';cwd=root/'packaging/offline-windows';cwd.mkdir(exist_ok=True)
target_win='Z:'+str(target).replace('/','\\')
subprocess.run(['wine',str(installer),'/S','/D='+target_win],env=env,check=True,timeout=300)
for name in ['PossibilityLab.exe','Engine.exe','Uninstall.exe']:
 assert (target/name).exists(),name
for name in ['PossibilityLab.exe','Engine.exe']:
 assert hashlib.sha256((target/name).read_bytes()).digest()==hashlib.sha256((root/'packaging/dist-windows/PossibilityLab'/name).read_bytes()).digest()
report_path=cwd/'self-test.json'
if report_path.exists():report_path.unlink()
subprocess.run(['wine',str(target/'PossibilityLab.exe'),'--self-test'],cwd=cwd,env=env,check=True,timeout=120)
report=json.loads(report_path.read_text());assert report['ok'] and report['bundled'];assert set(report['predictions'].values())=={'7283'}
result={'installer_exit':0,'installed_launch_exit':0,'models':report,'executable_hashes_match':True,'environment':'Wine 10.0 clean prefix; no Windows Python installed','native_windows_tested':False}
(root/'releases/windows-install-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
