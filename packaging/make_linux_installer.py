"""Create an offline per-user Linux installer; no Python required to install."""
from pathlib import Path
import tarfile,io,os
root=Path(__file__).resolve().parents[1];release=root/'releases/PossibilityLab-Linux-x64.run'
header='''#!/bin/sh
set -eu
TARGET="${XDG_DATA_HOME:-$HOME/.local/share}/PossibilityLab"
LAUNCH=1
if [ "${1:-}" = "--extract-only" ]; then TARGET="$2"; LAUNCH=0; fi
mkdir -p "$TARGET"
LINE=$(awk '/^__ARCHIVE_BELOW__$/ {print NR + 1; exit}' "$0")
tail -n +"$LINE" "$0" | tar -xz -C "$TARGET"
if [ "$LAUNCH" = 0 ]; then printf 'Extracted to %s\\n' "$TARGET"; exit 0; fi
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"
cat > "$APPS/possibility-lab.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Possibility Lab
Comment=Offline Parallel Possibility Engine
Exec="$TARGET/PossibilityLab"
Icon=$TARGET/_internal/ui/lab/icon.svg
Terminal=false
Categories=Education;Science;
EOF
chmod +x "$APPS/possibility-lab.desktop"
exec "$TARGET/PossibilityLab"
exit 0
__ARCHIVE_BELOW__
'''
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz',compresslevel=6) as t:
 for p in (root/'packaging/dist-linux/PossibilityLab').iterdir():t.add(p,arcname=p.name)
release.write_bytes(header.encode()+buf.getvalue());release.chmod(0o755);print(release,release.stat().st_size)
