// DINOv3 S/16 (timm eva.EvaAttention, rope_type='dinov3') — unfolded by EvaAttentionUnfolded.
// Verified against timm: vit_small_patch16_dinov3 → embed 384, 6 heads, hd 64, qkv_bias=False,
// q_norm/k_norm/norm = Identity, rope rotate_half=True, temperature=100, prefix tokens 5 (cls + 4 registers),
// scale = 64^-½ = 0.125, LayerScale (gamma_1) lives in the block, outside attention.
window.MODELS = window.MODELS || {};
(function () {
  // minimised patch grid 1×3 → coords via make_coords_dinov3('separate'): 0.5-centred, /size, ·2 − 1
  const gridH = 1, gridW = 3;
  const coord = (r, c) => [2 * ((r + 0.5) / gridH) - 1, 2 * ((c + 0.5) / gridW) - 1]; // [y, x] in [-1, 1]
  window.MODELS.dinov3 = {
    id: 'dinov3',
    name: 'DINOv3 S/16',
    source: 'EvaAttentionUnfolded',
    // minimised dims (real DINOv3-S/16: N=5+196, C=384, H=6, hd=64). hd must be a multiple of 4 for RoPE.
    N: 5, C: 8, H: 2, HD: 4,
    scale: 1 / Math.sqrt(4),
    seed: 11,
    tokens: [
      { label: 'cls' },
      { label: 'reg0' },
      { label: 'p0', coord: coord(0, 0) },
      { label: 'p1', coord: coord(0, 1) },
      { label: 'p2', coord: coord(0, 2) },
    ],
    qkvBias: false,
    rope: { temperature: 100, prefix: 2, grid: [gridH, gridW] },
    qkNormNote: 'DINOv3 has qk_norm=False → Identity',
    postNormNote: 'EvaAttention scale_norm=False → Identity',
    afterNote: 'next: ls1 = LayerScale γ (init 1e-5, a per-channel parameter outside attention), then residual add',
    mixNote: 'Everything after (LayerScale γ_1 ⊙, residual add, norm2, MLP) then sees the mixed number. RoPE never mixes heads: same constant angles applied inside each head.',
    dimsLine: 'N=5 tokens (cls + 1 register + 3 patches in a 1×3 grid) · C=8 · H=2 · hd=4 · scale=hd^-½=0.5 · RoPE T=100, rotate-half · B=1 · eval · real DINOv3-S/16: N=201 (cls + 4 reg + 196), C=384, H=6, hd=64',
    desc: 'timm <code>eva.EvaAttention</code> as configured by <code>vit_small_patch16_dinov3</code>, unfolded by <code>EvaAttentionUnfolded</code>. No absolute position embedding; position enters only through RoPE on q and k inside every block. Fused qkv <b>without</b> bias. Prefix tokens (cls, registers) skip RoPE. With hd=4 there is one RoPE period (=1), so the temperature does not act here; with the real hd=64 there are 16 periods 100^(2k/32).',
    defaultCell: 'Wqkv_0_0',
  };
})();
