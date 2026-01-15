#!/bin/bash
#
# SessionStart: Show active beads summary
#

BEADS_DIR="$CLAUDE_PROJECT_DIR/.beads"

if [[ ! -d "$BEADS_DIR" ]]; then
  echo "No .beads directory found. Run 'bd init' to initialize."
  exit 0
fi

# Check if bd is available
if ! command -v bd &>/dev/null; then
  echo "beads CLI (bd) not found. Install from: https://github.com/steveyegge/beads"
  exit 0
fi

echo ""
echo "## Active Beads"
echo ""

# Show ready (unblocked) beads
READY=$(bd ready 2>/dev/null || echo "")
if [[ -n "$READY" ]]; then
  echo "### Ready to work on:"
  echo "$READY" | head -10
else
  echo "No ready beads. Create one with: bd create \"Task title\""
fi

echo ""
echo "Commands: bd create, bd list, bd ready, bd close"
echo ""
