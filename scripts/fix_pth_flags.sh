#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fix_pth_flags.sh — Remove macOS UF_HIDDEN flags from .pth files in the venv
#
# UV 0.10.x marks files starting with '_' as hidden on macOS. Python 3.12's
# site.py skips hidden .pth files, which breaks editable installs.
#
# Run after every `uv sync`:
#   uv sync && bash scripts/fix_pth_flags.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SITE_PACKAGES=".venv/lib/python3.12/site-packages"

if [[ ! -d "$SITE_PACKAGES" ]]; then
    echo "⚠  site-packages not found at $SITE_PACKAGES"
    exit 1
fi

# Remove hidden flag from all .pth files
hidden_count=0
for pth in "$SITE_PACKAGES"/*.pth; do
    flags=$(ls -lO "$pth" 2>/dev/null | awk '{print $5}')
    if [[ "$flags" == *"hidden"* ]]; then
        chflags nohidden "$pth"
        echo "  ✓ Unhid: $(basename "$pth")"
        ((hidden_count++)) || true
    fi
done

if [[ $hidden_count -eq 0 ]]; then
    echo "✓ No hidden .pth files found"
else
    echo "✓ Fixed $hidden_count hidden .pth file(s)"
fi
