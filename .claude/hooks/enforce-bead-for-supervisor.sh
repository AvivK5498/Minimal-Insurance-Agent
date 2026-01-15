#!/bin/bash
#
# PreToolUse:Task - Enforce bead exists before supervisor dispatch
#
# Supervisors (except worker) must have BEAD_ID in prompt.
# This ensures all significant work is tracked.
#

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')

[[ "$TOOL_NAME" != "Task" ]] && exit 0

SUBAGENT_TYPE=$(echo "$INPUT" | jq -r '.tool_input.subagent_type // empty')
PROMPT=$(echo "$INPUT" | jq -r '.tool_input.prompt // empty')

# Only enforce for supervisors
[[ ! "$SUBAGENT_TYPE" =~ supervisor ]] && exit 0

# Worker-supervisor is exempt (handles small tasks without beads)
[[ "$SUBAGENT_TYPE" == *"worker"* ]] && exit 0

# Check for BEAD_ID in prompt
if [[ "$PROMPT" != *"BEAD_ID:"* ]]; then
  cat << 'EOF'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Supervisor dispatch requires a bead.\n\nCreate one first:\n  bd create \"Task title\" --description \"Details\"\n\nThen include in dispatch:\n  BEAD_ID: {id}\n  Branch: bd-{id}\n  Task: ...\n\nWorker-supervisor is exempt for small tasks."}}
EOF
  exit 0
fi

exit 0
