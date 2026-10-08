"""Benchmark execution scripts.

These are thin wrappers over :mod:`catface` used to produce the numbers quoted in
``docs/BENCHMARK.md``. They exist as scripts (not only as CLI subcommands) so that the
exact provenance of a reported table is a file that can be re-run and diffed.

Usage::

    python -m tools.run_verification --models baseline=resnet50 ...
"""

from __future__ import annotations
