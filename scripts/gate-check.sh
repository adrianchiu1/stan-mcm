#!/bin/bash
# gate-check.sh -- the one-line CI-style validation entry point (S6 WP3),
# and the SubagentStop hook (refuses to fold in a subagent result while the
# active gate is red).
#
#   scripts/gate-check.sh                      # hook mode: fast suite (or the
#                                              #   .claude/active-gate subset)
#   scripts/gate-check.sh fast                 # pytest -m "not slow"
#   scripts/gate-check.sh family <fam>         # fast tests marked for one family
#   scripts/gate-check.sh validate <fam> [tier] # mtk validate <fam> --tier <tier>
#                                              #   (tier: fast | recovery | sbc | all;
#                                              #   default fast)
#   scripts/gate-check.sh slow <fam>           # the family's slow gate tests
#
# Exit 0  -> allow / green
# Exit 2  -> block (hook mode: the message on stderr goes back to Claude)
# Exit 1  -> a validate tier or slow gate is red

set -uo pipefail
cd "$(dirname "$0")/.." || exit 0

# Nothing to check before the test suite exists.
[ -d tests ] || exit 0

MODE="${1:-hook}"

case "$MODE" in
  fast)
    exec pytest -q -m "not slow"
    ;;
  family)
    FAM="${2:?usage: gate-check.sh family <family>}"
    exec pytest -q -m "not slow and $FAM"
    ;;
  validate)
    FAM="${2:?usage: gate-check.sh validate <family> [tier]}"
    TIER="${3:-fast}"
    mtk validate "$FAM" --tier "$TIER"
    exit $?
    ;;
  slow)
    FAM="${2:?usage: gate-check.sh slow <family>}"
    exec pytest -q -m "slow and $FAM"
    ;;
  hook)
    ;;
  *)
    echo "usage: gate-check.sh [fast | family <fam> | validate <fam> [tier] | slow <fam>]" >&2
    exit 1
    ;;
esac

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
