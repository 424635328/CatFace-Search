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

## The `.gitignore` blind spot, and how it was found

A pattern with no leading slash matches that name at **any depth**. A `data/` line intended for
the dataset directory therefore also excluded `src/catface/data/` — six modules of the source
tree — which meant the package was **never committed**. Every local run passed, because the
files were still in the working tree; only a fresh clone failed, and only at import time. CI
caught it on both Python versions, and the error said nothing about `.gitignore`.

The same blind spot hid a second problem. Ruff honours `.gitignore` by default, so those six
modules were also never linted. The moment they entered version control, CI reported twelve
lint errors in them — while a local `ruff check` still reported a clean tree. That is why the
CI lint step now runs with `--no-respect-gitignore`: a file that is not linted is not checked,
and the reason it is not linted may be the very bug being hunted.

Three guards now cover this class of failure, and they exist because the failure was invisible
locally by construction:

| Guard | Catches |
|---|---|
| `test_every_python_source_file_is_tracked` | a source file that exists but git will not deliver |
| `test_no_source_directory_is_ignored` | a `.gitignore` rule actively excluding source |
| `test_project_owned_ignores_are_anchored` | the specific unanchored pattern that caused it |

Practical rules that follow:

* Anchor a project-owned directory with a leading slash (`/data/`). Leave third-party names
  unanchored on purpose, so a vendored copy nested anywhere is still ignored.
* After changing `.gitignore`, check `git ls-files --others --exclude-standard` for source
  paths that should have been added.
* Remember that `git check-ignore` reports what the *rules* say, not what is *tracked*: an
  already-tracked file stays tracked even when a pattern matches it.

## Relationship to `.gitignore`

`.gitignore` covers files that have never been tracked. It does **not** apply to a file that is
already in history — that requires `git rm --cached <file>`, and the removal only takes effect
in a commit. This matters when reading status output: a file whose name appears in
`.gitignore` can still show up in `git add .` as a **deletion** (`D`), which is the intended
cleanup rather than a leak. Read the status letter, not the file name.

## Before pushing

`tools/preflight.ps1` runs the CI checks locally with identical flags. See
[`PASS-CI-FIRST-TRY.md`](PASS-CI-FIRST-TRY.md) for why "close to CI" is not good enough.

## Automated checks

| Layer | Runs when | Protects against |
|---|---|---|
| `tests/test_repo_hygiene.py` | the test suite runs | the repository being dirty now |
| `tools/git-hooks/pre-commit` | at every commit | a violation entering history |
| `.gitignore` | on `git add` | untracked generated content entering the index |
| `core.hooksPath` | on `git commit` | the hook itself being silently absent |
