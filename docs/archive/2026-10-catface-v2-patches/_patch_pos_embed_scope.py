"""Make the temporary positional-embedding swap survive gradient checkpointing.

Root cause of the stalled training run
-------------------------------------
``__call__`` swapped in the correctly sized ``pos_embed`` and restored the native one in a
``finally`` block — i.e. before the backward pass. With ``gradient_checkpointing=True``,
PyTorch *recomputes* the forward during backward. That recomputation then ran against the
restored 1370-row embedding while the original graph had used 257 rows, so the recompute
either raised or produced mismatched gradients. The AMP scaler treated the overflow as a
bad step and skipped the optimiser update. The result was a model that never updated while
reporting a perfectly finite, constant loss:

    loss 30.4382 -> 30.4769 -> 30.4406 ... (10 epochs, val hit@1 pinned at 0.0565)

Fix: expose the swap as a context manager and hold it across ``forward`` *and*
``backward``. ``__call__`` uses it for the ordinary path; the training loop wraps the full
step. A checkpointed backward outside the context raises a clear error instead of silently
corrupting gradients.

Run once::

    python -m tools._patch_pos_embed_scope
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "src" / "catface" / "models" / "backbone.py"

OLD_CALL = '''    def __call__(self, images: Any) -> BackboneOutput:
        self._resample_pos_embed_if_needed(int(images.shape[-2]), int(images.shape[-1]))
        # Swap the correctly sized tensor into the module for this forward pass only, so the
        # registered parameter is never the thing that changes.
        original = self.network.pos_embed
        self.network.pos_embed = self._active_pos_embed
        try:
            return self._forward(images)
        finally:
            self.network.pos_embed = original'''

NEW_CALL = '''    @contextmanager
    def resolution_scope(self, height: int, width: int) -> Iterator[None]:
        """Hold the correctly sized positional embedding for a whole forward *and* backward.

        Must wrap the backward pass whenever ``gradient_checkpointing`` is enabled: the
        checkpointed recompute re-runs the forward, so restoring the native embedding early
        makes the recomputation disagree with the graph that was recorded, which either
        raises a shape error or yields wrong gradients. Either way the optimiser step is
        discarded and the model silently stops learning.
        """
        self._resample_pos_embed_if_needed(height, width)
        original = self.network.pos_embed
        self.network.pos_embed = self._active_pos_embed
        try:
            yield
        finally:
            self.network.pos_embed = original

    def __call__(self, images: Any) -> BackboneOutput:
        height, width = int(images.shape[-2]), int(images.shape[-1])
        with self.resolution_scope(height, width):
            return self._forward(images)'''

OLD_FORWARD = '''    def _forward(self, images: Any) -> BackboneOutput:
        torch = _torch()

        def forward(tensor: Any) -> Any:
            return self.network.forward_features(tensor)

        if self.gradient_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            tokens = checkpoint(forward, images, use_reentrant=False)
        else:
            tokens = forward(images)'''

NEW_FORWARD = '''    def _forward(self, images: Any) -> BackboneOutput:
        torch = _torch()

        def forward(tensor: Any) -> Any:
            return self.network.forward_features(tensor)

        if self.gradient_checkpointing and torch.is_grad_enabled():
            if not self._in_resolution_scope:
                # Failing loudly is essential: the alternative is a silently skipped
                # optimiser step and a training run that never converges.
                raise ModelError(
                    "gradient_checkpointing is enabled, so the positional-embedding scope "
                    "must stay open across the backward pass. Wrap the full step in "
                    "`backbone.resolution_scope(h, w)` (see catface.train.loop.Trainer)."
                )
            from torch.utils.checkpoint import checkpoint

            tokens = checkpoint(forward, images, use_reentrant=False)
        else:
            tokens = forward(images)'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")

    if OLD_CALL not in text:
        print("FAIL: __call__ block not found (already patched?)")
        return 1
    text = text.replace(OLD_CALL, NEW_CALL, 1)

    if OLD_FORWARD not in text:
        print("FAIL: _forward block not found")
        return 1
    text = text.replace(OLD_FORWARD, NEW_FORWARD, 1)

    # Track scope state in __init__.
    anchor = "        self._pos_embed_cache: dict[tuple[int, int], Any] = {}"
    if anchor not in text:
        print("FAIL: init anchor not found")
        return 1
    text = text.replace(anchor, anchor + "\n        self._in_resolution_scope = False", 1)

    # Set/clear the flag inside the context manager.
    text = text.replace(
        """        original = self.network.pos_embed
        self.network.pos_embed = self._active_pos_embed
        try:
            yield
        finally:
            self.network.pos_embed = original""",
        """        original = self.network.pos_embed
        self.network.pos_embed = self._active_pos_embed
        self._in_resolution_scope = True
        try:
            yield
        finally:
            self.network.pos_embed = original
            self._in_resolution_scope = False""",
        1,
    )

    # Imports for the context manager.
    text = text.replace(
        "from typing import Any, Callable, Protocol",
        "from contextlib import contextmanager\nfrom typing import Any, Callable, Iterator, Protocol",
        1,
    )
    TARGET.write_text(text, encoding="utf-8")
    print("patched backbone.py: positional-embedding scope survives backward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
