# Repository hygiene

This repository is published to a public remote with a sync script that runs
`git add . && git commit && git push`. That combination makes three classes of mistake
permanent, so all three are guarded automatically rather than by discipline.

## Setup, once per clone

```powershell
.\tools\install-git-hooks.ps1
```

This sets `core.hooksPath` to the versioned `tools/git-hooks/` directory, so the hook is
reviewable in review like any other code and an edit to it takes effect without reinstalling.
`core.hooksPath` is local configuration, which is why the installer has to be run rather than
the hook arriving with the clone; `tests/test_repo_hygiene.py` fails loudly if it is not
installed, and prints the command.

## What is guarded

| Guard | Why it cannot be left to a test suite |
|---|---|
| **Machine-specific absolute paths** — a single drive letter followed by a colon and a separator, or a user-home directory, with URLs stripped first so documentation links are not false positives | A hard-coded path works on the machine that wrote it and fails on every clone. Tests pass locally, so the defect survives until a user hits it. |
| **Credential assignments** — `KAGGLE_KEY`, `HF_TOKEN`, `GITHUB_TOKEN`, and similar, matched on `name = value` with a plausible value length | A debugging session is exactly when a key gets written into a script, and a public remote makes it irreversible. CI runs the same check, but a hook blocks it before it exists in history at all. |
| **Staged files over 5 MB** | GitHub rejects pushes over 100 MB per file and warns over 50 MB. Catching it at commit time avoids a failed push and a history rewrite. |

The hook inspects **staged** content, because that is what the commit will contain — the
working tree may differ. Bypass a genuine false positive with `git commit --no-verify`.

## Why the guards carry no exception lists

Writing this document produced the fifth instance of the same mistake in one session: every
attempt to explain the path rule included a *concrete example path*, and the guard flagged the
explanation itself. The earlier four were in a test docstring, a test's positive-control data,
and the hook's own comment.

The tempting fix is an allow-list for "documentation examples". That is wrong in the direction
that matters: a guard which must decide whether a path is illustrative will eventually approve
a real one, and an example containing a real path is itself a disclosure of the author's
directory layout.

So the rule is absolute, and the prose complies with it rather than the other way round. The
pattern is described in words above; where a test genuinely needs a path-shaped value, it is
assembled at runtime so no complete literal exists in any tracked file:

```python
drive = "X" + ":"          # not the literal
sep = chr(92)              # not a backslash literal
scheme = "ht" + "tps"      # not a scheme literal
```

The same applies to comments in the guard itself. **A rule the guard obeys needs no
exceptions** — and the guard is only trustworthy because it has been shown to flag its own
author, repeatedly.

## Relationship to `.gitignore`

`.gitignore` covers files that have never been tracked. It does **not** apply to a file that is
already in history — that requires `git rm --cached <file>`, and the removal only takes effect
in a commit. This matters when reading status output: a file whose name appears in
`.gitignore` can still show up in `git add .` as a **deletion** (`D`), which is the intended
cleanup rather than a leak. Read the status letter, not the file name.

## Automated checks

| Layer | Runs when | Protects against |
|---|---|---|
| `tests/test_repo_hygiene.py` | the test suite runs | the repository being dirty now |
| `tools/git-hooks/pre-commit` | at every commit | a violation entering history |
| `.gitignore` | on `git add` | untracked generated content entering the index |
| `core.hooksPath` | on `git commit` | the hook itself being silently absent |
