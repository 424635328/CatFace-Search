# Project-specific settings, kept in one file so tools/preflight.ps1 and tools/sync.ps1 stay
# generic and can be copied between repositories.
#
# To reuse this tooling in another project:
#   1. copy tools/preflight.ps1, tools/sync.ps1, tools/resolve-python.ps1 and this file
#   2. edit the values below
#   3. copy tools/check_staged.py and tools/git-hooks/pre-commit (they have no project knowledge)
#
# Everything here is optional; each value falls back to the default shown in the comment.

@{
    # Root directories that contain source, scanned for "exists locally but untracked" files.
    SourceRoots = @('src', 'tests', 'tools')

    # Directories passed to ruff. ``--no-respect-gitignore`` is added by preflight, matching CI.
    LintTargets = @('src', 'tests', 'tools')

    # Glob for shipped config files that must parse. Empty array skips the check.
    ConfigGlob = 'configs/*.yaml'

    # One config file used for the "fingerprint is location independent" regression check.
    # Empty string skips the check.
    FingerprintConfig = 'configs/default.yaml'

    # Importable package name(s) exercised from a fresh clone.
    Package = 'catface'

    # Python source directory, used as PYTHONPATH when checking a fresh clone.
    SourceDir = 'src'

    # Directory of documentation checked by tools/check_docs.py. Empty string skips the check.
    DocsChecker = 'tools/check_docs.py'
}
