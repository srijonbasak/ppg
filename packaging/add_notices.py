from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
for platform in ['linux','windows']:
 dest=root/f'packaging/dist-{platform}/PossibilityLab/THIRD-PARTY-NOTICES';dest.mkdir(exist_ok=True)
 for name in ['LGPL-3','GPL-3']:
  src=root/'packaging/licenses'/(name+'.txt')
  if src.exists():shutil.copy2(src,dest/(name+'.txt'))
 (dest/'README.txt').write_text('''This distribution includes Python, NumPy, PyTorch and Qt for Python (PySide6).
Qt/PySide6 is used under the GNU Lesser General Public License, version 3.
The bundled Qt libraries are dynamically linked and may be replaced with
compatible builds. GNU license texts are included alongside this notice.
Qt and PySide source distributions: https://download.qt.io/official_releases/
Qt WebEngine licensing/third-party information:
https://doc.qt.io/qt-6/qtwebengine-licensing.html
Python license: https://docs.python.org/3/license.html
PyTorch/NumPy license material is included in the respective bundled dist-info
license directories under _internal. Other runtime packages retain their notices.
The application does not download any source, models or dependencies at runtime.
''')
print('Third-party notices added to both application directories.')
