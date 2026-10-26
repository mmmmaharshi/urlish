# AGENTS.md

## Must follow: writing

Every reply uses the `$caveman $simple-english $i-have-adhd` skills together.
Docs, comments, commits, and any other written text use the
`$simple-english $unslop $writing-for-agents` skills together.

## Must follow: coding standards

`CODING_STANDARDS.md` is binding. Read it before writing code and follow every
rule. It applies to all code, tests, commits, and docs in this repo.

Python, managed with `uv`. `REPORT.md` and the README results table are generated
by `uv run urlish-benchmark`; never hand-edit either. Vocabulary is pinned in
`GLOSSARY.md`.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, with default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
`docs/README.md` indexes the rest of `docs/`.

### Research

Network research runs inline with bounded timeouts. Background subagents can hang
on slow fetches with no progress signal.
