# macrotoolkit — Claude Code configuration

Unzipped into your repo root, this provides:

```
.claude/
├── settings.json          # sandbox allowedDomains + SubagentStop gate hook
└── agents/
    ├── stan-engineer.md   # opus  — Stan authoring & divergence diagnosis (supervised only)
    ├── py-implementer.md  # sonnet — Python infrastructure
    ├── test-writer.md     # sonnet — gates G1–G6, fixtures
    ├── explorer.md        # haiku  — read-only search (name: Explore, overrides built-in)
    └── numerics-reviewer.md # sonnet — read-only review vs units conventions
scripts/
└── gate-check.sh          # hook: blocks results while the active gate is red
```

## Before launching a session

1. `chmod +x scripts/gate-check.sh` (should already be set, but check after unzip — zip does not always preserve the bit).
2. Ensure `VISION.md` and `lw-sv-spec.md` are in the repo root.
3. **Restart or start Claude Code after unzipping.** The agent-file watcher only covers directories that existed when the session started, so a brand-new `.claude/agents/` will not load mid-session.

## Notes

**`explorer.md` declares `name: Explore`** deliberately, overriding the built-in Explore agent so exploration stays pinned to Haiku instead of inheriting the session model. Rename it if you would rather keep the built-in behaviour.

**`gate-check.sh` reads `.claude/active-gate`** — a file containing e.g. `G1` — to decide which gate to enforce. Stage S1 should create it; until then the hook runs the fast test suite. It exits 2 on failure, which blocks the subagent result and returns the failure text to Claude.

**Check `sandbox.allowedDomains` against your environment.** Locally, no domains are pre-allowed by default and a first access prompts — which an unattended session cannot answer. On Claude Code for the web the proxy allowlist is Anthropic-managed and this setting will not extend it, so commit the HLW fixtures into `tests/fixtures/hlw/` rather than relying on fetching them at runtime.

**Model allocation:** run the session itself on Sonnet 5 for long unattended work; the roster pins each subagent explicitly, since the `model` field otherwise defaults to `inherit`. Do not invoke `stan-engineer` unsupervised.
