#!/usr/bin/env bash
# Builds Kuralkoodu.app and Kuralkoodu-<version>-<arch>.dmg on a Mac (Apple Silicon or Intel).
#   packaging/macos/build.sh            # from the repo root
# Needs: Python 3.12, Node 20+, Xcode command line tools (for iconutil, codesign, hdiutil).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
VERSION=$(cat VERSION)
ARCH=$(uname -m)
PY=${PYTHON:-python3}

"$PY" -m venv .venv-build
. .venv-build/bin/activate
pip install -q --upgrade pip
pip install -q -r backend/requirements.txt -r packaging/macos/requirements-build.txt

(cd frontend && npm ci --silent && npm run build --silent)
python packaging/macos/make_icon.py packaging/macos/build

rm -rf build/pyinstaller dist/Kuralkoodu dist/Kuralkoodu.app
pyinstaller --noconfirm --clean --workpath build/pyinstaller --distpath dist packaging/macos/kuralkoodu.spec

# ad-hoc signature so Apple Silicon will run it (no Developer ID: Gatekeeper will ask once, see INSTALL)
codesign --force --deep --sign - dist/Kuralkoodu.app

# smoke test the packaged engine before shipping it
KURALKOODU_HOME="$(mktemp -d)" dist/Kuralkoodu.app/Contents/MacOS/Kuralkoodu --self-test

DMG="dist/Kuralkoodu-${VERSION}-macOS-${ARCH}.dmg"
STAGE=$(mktemp -d)
cp -R dist/Kuralkoodu.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp packaging/macos/INSTALL.txt "$STAGE/Read me first.txt"
rm -f "$DMG"
hdiutil create -volname "Kuralkoodu" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
shasum -a 256 "$DMG" | tee "$DMG.sha256"
echo "Built $DMG"
