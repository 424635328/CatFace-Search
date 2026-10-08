"""Rewrite the ViT positional-embedding handling to be structurally idempotent.

Replacing ``network.pos_embed`` in place made the *registered* parameter
resolution-dependent, which caused two real defects:

* a forward at 224 px resized it to 257 rows, so a checkpoint saved afterwards could not be
  loaded into a pristine model (1370 rows for the 518 px training resolution);
* resampling an already-resampled tensor compounded interpolation error across forwards, so
  the descriptor of an image depended on the resolution of the previous image.

The fix keeps the registered parameter at its native resolution and swaps in a cached,
correctly sized tensor only for the duration of a forward pass. Interpolation is always
computed from the same starting point, so it is idempotent by construction and
``state_dict`` shapes stay stable.

Run once::

    python -m tools._patch_pos_embed
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "src" / "catface" / "models" / "backbone.py"

NEW_INIT_TAIL = '''        # The registered parameter stays at its native resolution forever. The correctly
        # sized tensor for the current input lives in ``_pos_embed_cache`` and is swapped in
        # only for the duration of a forward pass. That makes interpolation idempotent by
        # construction and keeps ``state_dict`` shapes stable across resolutions.
        self._native_pos_embed = self.network.pos_embed
        self._pos_embed_cache: dict[tuple[int, int], Any] = {}
        native_tokens = int(self._native_pos_embed.shape[1])
        prefix_count = int(self.network.num_prefix_tokens)
        native_patches = native_tokens - prefix_count
        self._native_grid = int(round(math.sqrt(max(native_patches, 1))))
        if self._native_grid * self._native_grid != native_patches:
            self._native_grid = 0  # not a square grid; leave the embedding alone'''

NEW_RESAMPLE_BLOCK = '''    def _resample_pos_embed_if_needed(self, height: int, width: int) -> None:
        """Select (and cache) the positional embedding that matches the input resolution.

        The class-token row is preserved and only the patch rows are bicubically resampled,
        which is what DINOv2's own implementation does when the resolution changes.
        """
        patch_embed = getattr(self.network, "patch_embed", None)
        if patch_embed is None:
            return
        patch_h, patch_w = patch_embed.patch_size
        if height % patch_h or width % patch_w:
            raise ModelError(
                f"Input size {height}x{width} must be divisible by the patch size "
                f"{patch_h}x{patch_w}"
            )
        grid = (height // patch_h, width // patch_w)
        patch_embed.img_size = (height, width)

        if self._native_grid == 0:
            LOGGER.warning(
                "%s: positional embedding has a non-square patch grid; resolution changes "
                "are unsupported and the native embedding is used", self.name,
            )
            self._active_pos_embed = self._native_pos_embed
            return
        if grid == (self._native_grid, self._native_grid):
            self._active_pos_embed = self._native_pos_embed
            return

        cached = self._pos_embed_cache.get(grid)
        if cached is None:
            cached = self._resample_pos_embed(grid)
            self._pos_embed_cache[grid] = cached
        self._active_pos_embed = cached

    def _resample_pos_embed(self, grid: tuple[int, int]) -> Any:
        """Bicubically resample the *native* positional embedding to ``grid``.

        Always derived from ``self._native_pos_embed``, never from a previously resampled
        tensor, so two calls for the same grid produce identical results.
        """
        torch = _torch()
        base = self._native_pos_embed
        prefix_count = int(self.network.num_prefix_tokens)
        embed_dim = int(base.shape[2])
        prefix = base[:, :prefix_count]
        patches = base[:, prefix_count:]
        patches = patches.reshape(1, self._native_grid, self._native_grid, embed_dim)
        patches = patches.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            patches, size=grid, mode="bicubic", align_corners=False
        )
        resized = resized.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], embed_dim)
        return torch.cat([prefix, resized], dim=1)'''

NEW_CALL = '''    def __call__(self, images: Any) -> BackboneOutput:
        self._resample_pos_embed_if_needed(int(images.shape[-2]), int(images.shape[-1]))
        # Swap the correctly sized tensor into the module for this forward pass only, so the
        # registered parameter is never the thing that changes.
        original = self.network.pos_embed
        self.network.pos_embed = self._active_pos_embed
        try:
            return self._forward(images)
        finally:
            self.network.pos_embed = original

    def _forward(self, images: Any) -> BackboneOutput:
        torch = _torch()

        def forward(tensor: Any) -> Any:
            return self.network.forward_features(tensor)'''

# The docstring of _enable_dynamic_input_size mentions the old mechanism; keep it accurate.
OLD_DOC = re.compile(
    r'        Two levers are needed.*?that branch runs, so the tensor it unpacks is always consistent\.\n        """',
    re.DOTALL,
)
NEW_DOC = '''        The positional embedding is *not* resized here. timm 1.0.30's own
        ``dynamic_img_size`` branch (and its ``dynamic_img_pad`` variant) assume a
        channels-last ``B,H,W,C`` tensor, while ``PatchEmbed`` returns a channels-first
        ``B,C,H,W`` map and ``forward_features`` hands ``_pos_embed`` a 3-D ``B,N,C``
        sequence. Enabling that branch therefore raises
        ``ValueError: not enough values to unpack (expected 4, got 3)``. This class performs
        the interpolation itself in :meth:`_resample_pos_embed_if_needed`, from a fixed
        native starting point.
        """'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")

    # 1. Replace the init tail that captured _base_grid/_base_pos_embed.
    old_init = """        self._base_grid = tuple(patch_embed.grid_size)
        self._base_pos_embed = self.network.pos_embed"""
    if old_init not in text:
        print("FAIL: init tail not found")
        return 1
    text = text.replace(old_init, NEW_INIT_TAIL, 1)

    # 2. Replace the whole resample method (from its def up to the start of __call__).
    start = text.index("    def _resample_pos_embed_if_needed(")
    end = text.index("    def __call__(self, images: Any) -> BackboneOutput:", start)
    # Guard against clobbering the wrong class: the TimmVitBackbone __call__ must be next.
    segment = text[start:end]
    if "self._base_pos_embed" not in segment:
        print("FAIL: unexpected method body between def and __call__")
        return 1
    text = text[:start] + NEW_RESAMPLE_BLOCK + "\n\n" + text[end:]

    # 3. Wrap __call__ so the registered parameter is restored afterwards.
    old_call_head = """    def __call__(self, images: Any) -> BackboneOutput:
        torch = _torch()
        self._resample_pos_embed_if_needed(int(images.shape[-2]), int(images.shape[-1]))

        def forward(tensor: Any) -> Any:
            return self.network.forward_features(tensor)"""
    if old_call_head not in text:
        print("FAIL: __call__ head not found")
        return 1
    text = text.replace(old_call_head, NEW_CALL, 1)

    # 4. Correct the stale docstring describing the old mechanism.
    text, count = OLD_DOC.subn(NEW_DOC, text)
    if count != 1:
        print(f"WARN: docstring replacement matched {count} times")

    TARGET.write_text(text, encoding="utf-8")
    print("patched backbone.py: structurally idempotent positional embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
