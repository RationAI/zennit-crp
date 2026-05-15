# Small pretrained Vision Transformers on explainability-friendly datasets

**ViT-only review.** Goal: identify pretrained ViT checkpoints (≤30M params,
ideally <10M) on datasets simple enough to hand-verify attribution
behaviour, so we can iterate on CRP / LRP code ~10× faster than the
current `vit_base_patch16_224` on FunnyBirds (86M params).

CNN / ResNet / VGG checkpoints excluded by design — they are not the
target of this project; CRP has been demonstrated on CNNs in the
literature and we want the per-attention-head, per-Q/K/V probing that
only ViTs expose.

---

## 1. ViT candidates (table)

| Dataset | Pretrained ViT | Params | Top-1 | Source | Download | XAI ground truth | Drop-in for our LRP pipeline? |
|---|---|---|---|---|---|---|---|
| ImageNet-1k | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | 5.7M | ~75.5% | timm / HF | `timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True)` | none (class labels only) | **Yes** — plain ViT block layout, same softmax / matmul ops as `vit_base` |
| ImageNet-1k | `deit_tiny_patch16_224.fb_in1k` | 5.7M | 72.2% / 91.1% top-5 | timm / HF (Meta) | `timm.create_model('deit_tiny_patch16_224.fb_in1k', pretrained=True)` | none | **Yes** — identical arch to `vit_tiny`, trained from scratch with distillation |
| ImageNet-1k | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 22.1M | ~80% | timm / HF | `timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)` | none | **Yes** — plain ViT, larger embed_dim (384) than vit_tiny (192) |
| ImageNet-1k | `tiny_vit_5m_224.dist_in22k_ft_in1k` (Microsoft) | 5.4M | ~80.7% | timm / HF | `timm.create_model('tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True)` | none | **No** — hierarchical (conv stem + windowed attention, Swin-like), needs separate canonizer work |
| FunnyBirds (50 cls) | visinf `vit_base_patch16_224_final_1` | 86M | ~98% | visinf TU Darmstadt | `download.visinf.tu-darmstadt.de/data/funnybirds/models/vit_base_patch16_224_final_1_checkpoint_best.pth.tar` | **part maps + part-removal evals** | what we already use — too big |
| FunnyBirds (50 cls) | no smaller-than-base ViT published | — | — | — | — | n/a | n/a |
| CIFAR-100 | `Ahmed9275/Vit-Cifar100` | 86M | 89.85% | HF | `AutoModelForImageClassification.from_pretrained('Ahmed9275/Vit-Cifar100')` | none | Same size as our current vit_base — no size win |
| dSprites | no public ViT checkpoint located | — | — | — | — | **6 GT factors** | n/a |
| Shapes3D / MPI3D | no public ViT checkpoint located | — | — | — | — | 6–7 GT factors | n/a |
| CLEVR-XAI | no public ViT checkpoint (paper uses RelationNet CNN+LSTM) | — | — | — | — | **GT object masks per question** | n/a |
| CLEVR-Hans3 / 7 | no public ViT checkpoint located | — | — | — | — | **GT relevant attributes + confound spec** | n/a |
| CUB-200-2011 | no canonical small ViT located; community fine-tunes exist (`vit_tiny`/`vit_small` recipes used in BMVC22 paper) | varies | varies | varies | varies | **312 binary attribute labels per image** | partial — recipes published, no canonical pretrained weights |

---

## 2. Per-candidate detail

### 2.1 `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (timm)
5.7M params, 1.1 GMACs, 224×224. **Identical block structure to our `vit_base_patch16_224`** —
same softmax, same `BilinearMatmul`, same residual layout. Embed_dim 192
(vs vit_base 768), 3 heads (vs 12), 12 blocks (same). Drop-in replacement
for our composite / canonizer / concept code. ImageNet top-1 ~75.5%.
A forward + LRP backward is comfortable on CPU.

### 2.2 DeiT-tiny (`facebook/deit-tiny-patch16-224`)
Same architecture as `vit_tiny` but trained from scratch on ImageNet-1k
with a distillation recipe. ~5.7M params, 72.2% top-1. Plain ViT
backbone — drop-in for our LRP rules. Useful as a **redundant cross-check**:
if our heatmaps differ between vit_tiny (in21k pretrain) and DeiT-tiny
(scratch) on the same image, that's a model-dependent vs.
method-dependent signal.

### 2.3 `vit_small_patch16_224.augreg_in21k_ft_in1k`
22.1M params, 4.3 GMACs, ~80% top-1. Middle ground between vit_tiny
and vit_base. Drop-in for our pipeline. Worth picking up only if
vit_tiny accuracy proves too weak on a downstream fine-tune
(e.g. FunnyBirds), since `vit_small` is ~4× heavier than `vit_tiny`.

### 2.4 TinyViT-5M (Microsoft)
5.4M params, ~80.7% top-1 ImageNet — **the best small-ViT accuracy of
the bunch**. *But* the architecture is hierarchical (conv stem +
windowed attention, Swin-flavoured), so our existing LRP rules don't
cover it without adding a windowed-attention canonizer. Not the
fastest path to "iterate against what we already wrote."

### 2.5 ViT-B/16 on CIFAR-100 (`Ahmed9275/Vit-Cifar100`)
86M params, 89.85% top-1. **Same size as our current FunnyBirds vit_base
— no size win.** Listed only for completeness; not useful for the
"smaller backbone" goal.

### 2.6 FunnyBirds — no smaller-than-base ViT exists publicly
The visinf framework ships **three** trained classifiers (ResNet-50,
VGG-16, vit_base_patch16_224) — no `vit_tiny` or `vit_small`. The
`final_1` vit_base checkpoint we already use is the only public ViT on
FunnyBirds. To get a small ViT with FunnyBirds part-removal eval, we
**have to train one ourselves** (see §4).

### 2.7 CUB-200-2011 (real birds)
11,788 images, 200 species, 312 binary attribute labels per image
(beak shape, primary colour, wing pattern, etc.) — these attribute
labels are exactly the kind of concept ground truth we want to
attribute against. The BMVC 2022 paper *"Vision Transformers for
Small-Scale Datasets"* ([Shafi et al.](https://github.com/hananshafi/vits-for-small-scale-datasets))
documents `vit_tiny` / `vit_small` recipes that reach 80–88% on CUB
with a shifted-patch tokenization. **But there is no canonical
HuggingFace card** — anyone using CUB-200 today either trains their
own ViT or pulls a community ResNet/Inception checkpoint. Same
fine-tuning effort as §4 below.

### 2.8 dSprites / Shapes3D / MPI3D
DeepMind / etc. synthetic datasets with exact factor-of-variation
labels (shape, scale, orientation, x, y, ...). These would be the
cleanest XAI testbeds — but they were designed for disentangled
representation learning (VAE benchmarks), not classification. **No
public ViT or even CNN classifier** is published. Training a small ViT
on a 4-way classification (e.g., shape) is trivial (single epoch on
CPU) but it's our own work.

### 2.9 CLEVR-XAI / CLEVR-Hans
Synthetic 3D scenes with GT object masks (CLEVR-XAI) or deliberate
confounds (CLEVR-Hans). Beautiful XAI testbeds. **No public ViT
classifier** for either. CLEVR-XAI is multimodal (image + question
→ answer) and doesn't fit single-image classification; CLEVR-Hans does
but no canonical small ViT exists.

---

## 3. Top-3 recommendations (ViT-only)

For "iterate against a known-good simple ViT to validate XAI code,"
ranked.

### 3.1 `vit_tiny_patch16_224.augreg_in21k_ft_in1k` on ImageNet validation samples
- **Why:** byte-for-byte architectural match to our current vit_base
  setup (same block, same softmax, same matmul) but 15× fewer params.
  Every LRP rule we wrote for vit_base works without modification.
  Forward + LRP backward is comfortable on CPU. Loading is a one-liner
  via `timm.create_model(..., pretrained=True)`.
- **Cross-check value:** ImageNet is the standard testbed for the
  ViT-XAI literature (AttnLRP, Chefer, CRP feature-vis tutorial all
  use ImageNet + ViT). Reproducing published heatmaps on the same
  input + same model is a clean bug-signal.
- **What "good" looks like:** on an obvious "dog" image, the
  attribution heatmap should highlight the dog. Conservation should
  hold within float-32 noise. If neither, the bug is in our generic
  ViT plumbing (not FunnyBirds-specific).
- **Caveat:** no part-level ground truth, only class predictions.
  Useful for "does code run + do heatmaps look qualitatively right"
  iteration. Not useful for quantitative XAI-method comparison.

### 3.2 `vit_tiny` fine-tuned on FunnyBirds (we train it — §4 sketch)
- **Why:** keeps the FunnyBirds part-removal XAI evaluation pipeline
  we have today but at 15× fewer params than visinf's vit_base. Exact
  drop-in for our existing notebook pipeline once the checkpoint
  exists.
- **Cross-check value:** directly comparable to our current vit_base
  FunnyBirds numbers. Same ground truth, same XAI eval harness.
- **Caveat:** ~25h wall-clock on a single mid-range GPU. Not zero
  effort. If vit_tiny tops out below ~88% accuracy, fall back to
  vit_small (22M).

### 3.3 DeiT-tiny as a redundant cross-check on ImageNet
- **Why:** identical architecture to vit_tiny but trained from scratch
  rather than from in21k. If our attribution heatmaps for both models
  agree on the same input + class, the method is model-robust. If
  they disagree, the disagreement is informative (training-data
  effect, not method effect).
- **Caveat:** lower top-1 (72%) than vit_tiny-augreg (75.5%) so
  prediction confidence on tricky images is weaker — pick high-
  confidence samples for any qualitative comparison.

---

## 4. If nothing fits: train our own small ViT

Public ViT checkpoints on XAI-friendly datasets (FunnyBirds, CUB-200,
dSprites, CLEVR-Hans) at <30M params **do not exist** at the time of
this review. To get one we have to train it ourselves.

### 4.1 vit_tiny on FunnyBirds (recommended)
- **Architecture:** `timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=50)`.
  5.7M params, 12 blocks, 3 heads, embed_dim 192.
- **Init:** start from `augreg_in21k_ft_in1k` weights; replace 1000-class
  head with 50-class head. visinf used the same in21k init for their
  vit_base FunnyBirds checkpoint.
- **Data:** FunnyBirds 50k train / 5k test, 256×256 → bilinear resize
  to 224×224. visinf trained with NO normalize (matches the
  `transform_spec='visinf_funnybirds_vit_base'` recipe we already have).
- **Recipe:** AdamW, lr 1e-4 with cosine decay, batch 128, 50 epochs.
  RandAugment + RandomErasing. (Same hyperparameters visinf used,
  just at the smaller backbone.)
- **Wall-clock estimate:** ~30 min/epoch on a single mid-range GPU
  (T4/A4000) for 50 epochs → ~25h total. ~3× faster on a 3090/4090.
- **Target accuracy threshold:** visinf reports vit_base reaches ~98%
  on FunnyBirds. With vit_tiny + in21k init we should target ≥93%.
  If we plateau below ~88%, fall back to vit_small (22M).

### 4.2 vit_tiny on CUB-200 (alternative)
Same recipe, swap the dataset. The Shafi et al. BMVC 2022 paper
reports `vit_tiny` reaches ~80% on CUB with shifted-patch
tokenization. We get 312 binary attribute labels per image as
concept ground truth — finer-grained than FunnyBirds' 8 parts.
Same wall-clock as 4.1.

### 4.3 vit_tiny on dSprites / Shapes3D / MPI3D (cheapest by far)
- 4-way classification (e.g. shape × scale joint label) on 64×64
  binary or RGB sprites.
- vit_tiny is overkill for 64×64; a smaller variant
  (`vit_patch4_size32` or similar) would be cleaner. timm doesn't
  ship one out of the box but it's a 50-line `VisionTransformer(...)`
  call.
- ~1 epoch on CPU. Exact factor labels = cleanest possible attribution
  ground truth. Lowest wall-clock by an order of magnitude. Useful
  for "is our pipeline broken at the fundamental level" sanity checks.

---

## Sources

- timm vit_tiny: https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k
- timm vit_small: https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k
- TinyViT-5M (Microsoft): https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k
- TinyViT paper: https://arxiv.org/abs/2207.10666
- DeiT-tiny (Meta): https://huggingface.co/facebook/deit-tiny-patch16-224
- ViT-B CIFAR-100: https://huggingface.co/Ahmed9275/Vit-Cifar100
- FunnyBirds framework + checkpoints: https://github.com/visinf/funnybirds-framework
- FunnyBirds paper (ICCV 2023): https://arxiv.org/abs/2308.06248
- CUB-200-2011 dataset: https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011
- ViTs for Small-Scale Datasets (BMVC 2022): https://github.com/hananshafi/vits-for-small-scale-datasets
- dSprites: https://github.com/google-deepmind/dsprites-dataset
- CLEVR-XAI: https://github.com/ahmedmagdiosman/clevr-xai
- CLEVR-XAI paper: https://arxiv.org/abs/2003.07258
- CLEVR-Hans: https://github.com/ml-research/CLEVR-Hans
- AttnLRP paper: https://arxiv.org/abs/2402.05602
- CRP zennit-crp tutorials: https://github.com/rachtibat/zennit-crp/tree/master/tutorials
- CRP Nature paper: https://www.nature.com/articles/s42256-023-00711-8
