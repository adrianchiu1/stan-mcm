#!/bin/bash
# SubagentStop hook: refuses to fold in a subagent result while the active gate is red.
#
# Exit 0  -> allow
# Exit 2  -> block, and the message on stderr goes back to Claude
#
# The active gate is named in .claude/active-gate (e.g. "G1"). Stage S1 should
# create that file; until it exists this hook runs the fast suite only.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 0

# Nothing to check before the test suite exists.
[ -d tests ] || exit 0

GATE_FILE=".claude/active-gate"
if [ -f "$GATE_FILE" ]; then
  GATE=$(tr -d '[:space:]' < "$GATE_FILE")
else
  GATE=""
fi

if [ -n "$GATE" ]; then
  OUTPUT=$(pytest -q -m "not slow" -k "$GATE or conventions" 2>&1)
else
  OUTPUT=$(pytest -q -m "not slow" 2>&1)
fi
STATUS=$?

if [ $STATUS -ne 0 ]; then
  {
    echo "BLOCKED: validation gate ${GATE:-fast suite} is failing. Do not proceed to the next stage."
    echo "Fix the failure or report it. Do NOT widen a tolerance, skip a test, or loosen a prior to get past this."
    echo "---"
    echo "$OUTPUT" | tail -n 40
  } >&2
  exit 2
fi

exit 0
