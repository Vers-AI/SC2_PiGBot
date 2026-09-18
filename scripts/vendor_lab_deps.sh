#!/bin/bash
# Vendor PiG_Bot's engine dependencies in-tree (for local play / test lab runs).
#
# The lab (github.com/Vers-AI/sc2_bot_test_lab) mounts this repo onto the
# stock arenaclient-bot image, whose python-sc2 (7.1.3) SHADOWS our pin and
# breaks on_start. Vendoring the pinned stack in-tree — the same recipe
# create_ladder_zip.py uses at zip time — makes the live-mounted repo
# self-contained.
#
# Pins (mirror poetry.lock resolved references):
#   sc2               august-k/python-sc2  7ec25cf8  (develop; NOT PyPI 7.1.0 —
#                                                    PyPI has the renamed
#                                                    ConnectionAlreadyClosedError)
#   map_analyzer      spudde123/SC2MapAnalysis develop (911de6c)
#   cython_extensions cp312 manylinux wheel of cython-extensions-sc2 0.17.0
#                                                    (matches ares pin ^0.17.0)
#   ares-sc2          git submodule 8730865 (already in-tree; run
#                                                    `git submodule update --init`
#                                                    if the dir is empty)
set -euo pipefail
cd "$(dirname "$0")/.."

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

echo "==> Vendoring sc2 @ 7ec25cf8 (august-k/python-sc2 develop)"
rm -rf sc2
curl -sSL "https://github.com/august-k/python-sc2/archive/7ec25cf80104391a91bd755bf637c93ab746686f.tar.gz" | tar xz -C "$work" --strip-components=1
cp -r "$work/sc2" ./sc2
rm -rf "$work"/*

echo "==> Vendoring map_analyzer @ spudde123/SC2MapAnalysis develop (911de6c)"
rm -rf map_analyzer
curl -sSL "https://github.com/spudde123/SC2MapAnalysis/archive/911de6cdac8179f36adf7633a859c925a5943e37.tar.gz" | tar xz -C "$work" --strip-components=1
cp -r "$work/map_analyzer" ./map_analyzer
rm -rf "$work"/*

echo "==> Vendoring cython_extensions (cp312 manylinux wheel, cython-extensions-sc2 0.17.0)"
URL=$(curl -s https://pypi.org/pypi/cython-extensions-sc2/0.17.0/json | python3 -c "
import json,sys
for u in json.load(sys.stdin)['urls']:
    if 'cp312' in u['filename'] and 'x86_64' in u['filename'] and 'manylinux' in u['filename']:
        print(u['url']); break")
test -n "$URL" || { echo "ERROR: no cp312 manylinux wheel found on PyPI"; exit 1; }
rm -rf cython_extensions
curl -sSL "$URL" -o "$work/cyext.whl"
python3 -m zipfile -e "$work/cyext.whl" "$work/cyext"
cp -r "$work/cyext/cython_extensions" ./cython_extensions

echo "==> Done."
echo "    Verify inside the arena image:"
echo "    docker run --rm -v \$PWD:/bots/PiG_Bot -w /bots/PiG_Bot \\"
echo "      --entrypoint python aiarena/arenaclient-bot:v0.8.0 -c \\"
echo "      'import sys; sys.path+= [\"ares-sc2/src/ares\",\"ares-sc2/src\",\"ares-sc2\"]; from bot import PiG_Bot; PiG_Bot()'"