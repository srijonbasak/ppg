"""Bundled offline desktop entry point. --self-test needs no display."""
import json,os,sys,threading,traceback
from pathlib import Path

def self_test():
 import torch
 from ppf.lab_server import ROOT,make_audio,predict
 torch.set_num_threads(1)
 events=make_audio('7283','clean',0,0,42)
 results={}
 for name in ['original_joint','original_aligned','recognition']:
  result=predict(events,317,name)
  assert [len(c[0]) for c in result['codes']]==[1,2,4]
  assert all(abs(sum(p)-1)<1e-5 for p in result['final'])
  results[name]=result['ranking']['ncm'][0]['sequence']
 report={'ok':True,'predictions':results,'bundled':bool(getattr(sys,'frozen',False))}
 Path('self-test.json').write_text(json.dumps(report,indent=2))
 if sys.stdout is not None:print(json.dumps(report),flush=True)
 return 0

def main():
 if '--self-test' in sys.argv:return self_test()
 # Put errors somewhere readable when the application has no console.
 state=Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local/share'))/'PossibilityLab';state.mkdir(parents=True,exist_ok=True)
 log=(state/'app.log').open('a',buffering=1)
 if sys.stdout is None:sys.stdout=log
 if sys.stderr is None:sys.stderr=log
 from PySide6.QtCore import QUrl,QTimer
 from PySide6.QtGui import QIcon,QKeySequence,QShortcut
 from PySide6.QtWidgets import QApplication,QMainWindow,QMessageBox
 from PySide6.QtWebEngineWidgets import QWebEngineView
 from http.server import ThreadingHTTPServer
 import torch
 from ppf.lab_server import Handler,ROOT
 torch.set_num_threads(1)
 app=QApplication(sys.argv);app.setApplicationName('Possibility Lab');app.setOrganizationName('PossibilityLab')
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
 window=QMainWindow();window.setWindowTitle('Parallel Possibility Engine');window.setWindowIcon(QIcon(str(ROOT/'ui/lab/icon.svg')))
 view=QWebEngineView(window);window.setCentralWidget(view);window.resize(1280,820);window.setMinimumSize(1000,680)
 view.setUrl(QUrl(f'http://127.0.0.1:{server.server_port}/'))
 shortcut=QShortcut(QKeySequence('F11'),window);shortcut.activated.connect(lambda:window.showNormal() if window.isFullScreen() else window.showFullScreen())
 app.aboutToQuit.connect(server.shutdown)
 window.show() if '--smoke-ui' in sys.argv else window.showMaximized()
 if '--smoke-ui' in sys.argv:
  def inspect_ui():
   view.page().runJavaScript("JSON.stringify({title:document.title,height:innerHeight,scroll:document.documentElement.scrollHeight,keyCount:document.querySelectorAll('.key').length,diagramBottom:Math.round(document.querySelector('.flow-panel').getBoundingClientRect().bottom),resultBottom:Math.round(document.querySelector('#resultNode').getBoundingClientRect().bottom)})",lambda v:print(v,flush=True))
   view.grab().save(str(Path.cwd()/'desktop-smoke.png'));QTimer.singleShot(1000,app.quit)
  QTimer.singleShot(5000,inspect_ui)
 return app.exec()
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception:
  traceback.print_exc()
  raise
