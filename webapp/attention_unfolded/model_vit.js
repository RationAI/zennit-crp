// ViT-B/16 (timm vision_transformer.Attention) — unfolded by TimmAttentionUnfolded.
window.MODELS = window.MODELS || {};
window.MODELS.vit = {
  id: 'vit',
  name: 'ViT-B/16',
  source: 'TimmAttentionUnfolded',
  // minimised dims (real ViT-B: N=197, C=768, H=12, hd=64)
  N: 3, C: 4, H: 2, HD: 2,
  scale: 1 / Math.sqrt(2),
  seed: 7,
  tokens: [
    { label: 'cls' },
    { label: 'p0' },
    { label: 'p1' },
  ],
  qkvBias: true,
  rope: null,
  qkNormNote: 'standard timm ViT has no qk-norm',
  postNormNote: 'post-attention norm, absent on standard ViT',
  afterNote: 'next: ls1 = Identity on ViT-B, then residual add',
  mixNote: 'Everything after (ls1 = Identity, residual add, norm2, MLP) then sees the mixed number.',
  dimsLine: 'N=3 tokens (cls + 2 patches) · C=4 · H=2 · hd=2 · scale=hd^-½=0.7071 · B=1 · eval · real ViT-B/16: N=197, C=768, H=12, hd=64',
  desc: 'timm <code>vision_transformer.Attention</code>, unfolded by <code>TimmAttentionUnfolded</code>. Absolute learned position embedding is added before block 0, so attention itself has no positional op. Fused qkv <b>with</b> bias.',
  defaultCell: 'Wqkv_0_0',
};
