# Small pretrained models on explainability-friendly datasets

Goal: identify pretrained checkpoints (≤30M params, ideally <10M) on datasets simple
enough to hand-verify attribution behaviour, so we can iterate on CRP / LRP code
~10× faster than the current `vit_base_patch16_224` on FunnyBirds (86M params).

Scope: pure web review, no installs, no training. All checkpoint sizes / accuracies
were taken from the listed model cards. We mark anything we could not verify with
**(unconfirmed)**.

---

## 1. Candidates (table)

| Dataset | Pretrained model | Params | Top-1 | Source | Download | XAI ground truth |
|---|---|---|---|---|---|---|
| ImageNet-1k | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | 5.7M | ~75.5% | timm / HF | `timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True)` | none direct; visual sanity-check on standard ImageNet samples |
| ImageNet-1k | `tiny_vit_5m_224.dist_in22k_ft_in1k` (TinyViT) | 5.4M | ~80.7% | timm / HF (Microsoft) | `timm.create_model('tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True)` | none direct; same as above |
| ImageNet-1k | `deit_tiny_patch16_224.fb_in1k` | ~5.7M | 72.2% | timm / HF (Facebook) | `timm.create_model('deit_tiny_patch16_224.fb_in1k', pretrained=True)` | none direct |
| ImageNet-1k | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 22.1M | ~80% | timm / HF | `timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)` | none direct |
| CIFAR-10 | `edadaltocg/resnet18_cifar10` | ~11M | 94.98% | HF | `timm.create_model('hf_hub:edadaltocg/resnet18_cifar10', pretrained=True)` | none direct; only class labels |
| CIFAR-100 | `edadaltocg/resnet18_cifar100` | ~11M | 79.26% | HF | `timm.create_model('hf_hub:edadaltocg/resnet18_cifar100', pretrained=True)` | none direct; only class labels |
| CIFAR-100 | `Ahmed9275/Vit-Cifar100` (ViT-B/16 ft) | 86M | 89.85% | HF | `AutoModelForImageClassification.from_pretrained('Ahmed9275/Vit-Cifar100')` | none direct |
| Imagenette (10 cls) | `frgfm/resnet18` (Holocron) | ~11M | acc not stated **(unconfirmed)** | HF | `from holocron.models import resnet18; resnet18(pretrained=True)` | none direct; class labels only |
| Imagenette (10 cls) | `nateraw/timm-resnet18-imagenette-160px-5-epochs` | ~11M | 5-epoch toy run **(unconfirmed)** | HF | `timm.create_model('hf_hub:nateraw/timm-resnet18-imagenette-160px-5-epochs', pretrained=True)` | none direct |
| FunnyBirds (50 cls) | visinf `resnet50_final_0` | 25.6M | ~99.8% | visinf TU Darmstadt | `download.visinf.tu-darmstadt.de/data/funnybirds/models/resnet50_final_0_checkpoint_best.pth.tar` | **part maps + part-removal evals** |
| FunnyBirds (50 cls) | visinf `vgg16_final_1` | ~135M | ~close to 1.0 **(unconfirmed)** | visinf TU Darmstadt | `download.visinf.tu-darmstadt.de/data/funnybirds/models/vgg16_final_1_checkpoint_best.pth.tar` | **part maps + part-removal evals** |
| CUB-200-2011 (birds) | various; no canonical official checkpoint | ~11–86M | 80–88% (varies) | community GitHubs | no single canonical HF model card | **312 binary part attributes per image** |
| dSprites | no public pretrained classifier we could locate | n/a | n/a | DeepMind dataset only | dataset only at `google-deepmind/dsprites-dataset` | **6 ground-truth factors (shape, scale, orient., x, y)** |
| Shapes3D / MPI3D | no public pretrained classifier we could locate; disentangle-VAE only | n/a | n/a | `google-research/disentanglement_lib` | dataset only | 6–7 ground-truth factors |
| CLEVR-XAI | RelationNet (custom CNN+LSTM, ~few M) | small | high (per paper) | `ahmedmagdiosman/clevr-xai` | dataset only; trainer separate | **GT object masks per question** |
| CLEVR-Hans3 / 7 | no canonical HF checkpoint located | n/a | n/a | `ml-research/CLEVR-Hans` | dataset only | **GT relevant attributes + confound spec** |
| Saliency-Bench (8 sets) | per-dataset classifiers shipped with benchmark | various | various | `xaidataset.github.io` | API in benchmark repo | **per-pixel GT explanation masks** |

---

## 2. Per-candidate detail

### 2.1 `vit_tiny_patch16_224` (timm, augreg in21k → ft in1k)
5.7M params, 1.1 GMACs, 224×224. Same architecture family as the FunnyBirds
`vit_base_patch16_224` we already use — same block structure, same attention head
ops, just embedding dim 192 instead of 768 and 12 vs. 12 heads → drop-in
replacement for our `BilinearMatmul` / softmax LRP rules. ImageNet top-1 ≥75.3%
per the timm "fastest models" collection. Best candidate for "iterate against
a faithful ViT scaled 15× smaller than vit-base on a dataset our LRP pipeline
already handles." (No part-level XAI ground truth on ImageNet itself — but the
goal here is XAI *code* iteration, not XAI *evaluation*.)

### 2.2 TinyViT-5M (Microsoft, dist in22k → ft in1k)
5.4M params, ~80.7% top-1 ImageNet (a full 5 points above DeiT-tiny). Uses a
hierarchical conv-stem + windowed-attention architecture (closer to Swin than
classical ViT), so the LRP backbone code needs separate adaptation — not a
drop-in for our existing pure-ViT rules. Strong if/when we want a small but
accurate model; not the fastest path to "use what we already wrote."

### 2.3 DeiT-tiny (`facebook/deit-tiny-patch16-224`)
Same arch as ViT-Tiny but trained from scratch on ImageNet-1k with distillation
recipe; ~5M–5.7M params, 72.2% top-1, 91.1% top-5. Plain ViT block layout —
drop-in for our LRP rules. Slightly worse than the augreg-in21k ViT-Tiny on
top-1 but architecturally identical. Useful as a redundant cross-check.

### 2.4 `vit_small_patch16_224` (augreg in21k → ft in1k)
22.1M params, 4.3 GMACs, ≥80% top-1. Middle ground if vit_tiny feels too weak
on some downstream task (e.g. FunnyBirds fine-tune). Still ~4× smaller than
vit_base.

### 2.5 ResNet-18 on CIFAR-10 (`edadaltocg/resnet18_cifar10`)
~11M params, 94.98% test top-1, trained 300 epochs SGD. 32×32 input. Fastest
possible smoke-test: a forward+backward pass takes single-digit ms on CPU. Good
for verifying that LRP composites / Canonizers wire up correctly at the level
of "does relevance conservation hold to numerical tolerance" before stress-
testing on bigger models. No XAI ground truth — but for **code regression
tests** that's not what we need; we need known-good attribution signals on
familiar classes (cat, dog, airplane).

### 2.6 ResNet-18 on CIFAR-100 (`edadaltocg/resnet18_cifar100`)
Same recipe, ~79.26% test top-1. Slightly more interesting failure modes than
CIFAR-10 (closer classes).

### 2.7 ViT-B/16 on CIFAR-100 (`Ahmed9275/Vit-Cifar100`)
86M params, 89.85% top-1. Useful sanity reference: same architecture as our
FunnyBirds model but a totally different (and simpler) image-classification
task. If our attribution code produces sane heatmaps on this and broken ones
on FunnyBirds, the bug is FunnyBirds-specific. If both are broken, the bug is
in our generic ViT plumbing.

### 2.8 ResNet-18 on Imagenette (Holocron / frgfm; nateraw checkpoint)
~11M params. Imagenette = 10 easy ImageNet classes (golf ball, parachute,
tench, etc.), full 224×224 input. We already have an Imagenette loader, so
this slots in zero-friction. Reported accuracy not on the model card, but
Imagenette is easy; ResNet-18 reaches >95% in standard recipes. **(unconfirmed
accuracy.)**

### 2.9 FunnyBirds ResNet-50 / VGG-16 (visinf official)
ResNet-50: 25.6M params, ~99.8% test acc reported in framework docs. VGG-16
checkpoint also released. These are the *only* checkpoints with FunnyBirds
part-removal ground truth attached — i.e. the actual XAI eval signal. ResNet-50
is the cheapest of the three official visinf checkpoints; ~3.4× smaller than
ViT-B and uses a plain conv stack (Zennit/CRP already has good Canonizer
support for it via `SequentialMergeBatchNorm`).

### 2.10 CUB-200-2011 (no canonical small checkpoint)
Real birds (11,788 images, 200 species, 312 binary attribute labels per image:
beak shape, primary colour, etc.). The attribute labels are exactly the kind
of "concept ground truth" we want to attribute against. Downside: no canonical
HF model card for a small ViT/ResNet on CUB; community repos exist (e.g.
`zhangyongshun/resnet_finetune_cub`, `PRIS-CV/Mutual-Channel-Loss/CUB-200-2011_ResNet18.py`)
but require manual training or porting.

### 2.11 dSprites
DeepMind's 737k binary 64×64 images with **exact** factor labels (shape, scale,
orient, x, y). The cleanest possible "I know what concept the model should be
using" signal in the wild. **But:** the dataset was built for disentangled
representation learning (VAEs), not classification, and we could not find any
public pretrained classifier — only VAE checkpoints. Training a tiny CNN from
scratch is cheap (see Section 4).

### 2.12 CLEVR-XAI
Synthetic 3D rendered scenes (cubes / spheres / cylinders, 8 colours,
2 materials, 3 sizes) with VQA-style questions; each question has a **GT
object mask** marking the answer-relevant objects. Reference model is the
Relation Network (4-layer CNN + LSTM) — multimodal, not pure image
classification, so the existing CRP/LRP image-attribution pipeline would
need attention extending across the QA module. Powerful eval but mismatched
to our current single-image-input ViT setup.

### 2.13 CLEVR-Hans (3 / 7)
Pure image classification on CLEVR-style scenes, **deliberately confounded**
(e.g. all "class 3" images have a large grey cube AND a small red sphere,
so the model can shortcut on either). Beautiful tool for testing whether
CRP correctly attributes to the *real* concept vs. the *confounded* concept.
No canonical released checkpoint, but classification recipe is small (CLEVR
images are 128×128, a ResNet-18 will overfit in a few hours).

### 2.14 Saliency-Bench
Standardised benchmark of 8 small classification datasets (Gender, Cancer,
Pet, etc.), each shipping per-pixel GT explanation masks AND classifier
checkpoints. Probably the most production-grade XAI eval pipeline available.
We have not verified what the classifier architectures are — likely
ResNet-50-tier; sizes need confirmation **(unconfirmed)**.

### 2.15 Shapes3D / MPI3D
Same family as dSprites (factor-labelled synthetic), but no public
classification checkpoints either. Skipping for the same reason as dSprites.

---

## 3. Top-3 recommendations

For "iterate against a known-good simple model + dataset to validate XAI
implementations," ranked.

### 3.1 `vit_tiny_patch16_224.augreg_in21k_ft_in1k` on ImageNet validation samples
- Why: byte-for-byte architectural match to our current vit_base setup
  (same block, same softmax, same matmul) but 15× fewer params. Every LRP
  rule we wrote for vit_base works without modification. Forward+backward
  is comfortable on CPU.
- Cross-check value: ImageNet is the original LRP-paper testbed for ViTs
  (AttnLRP, Chefer, CRP feature-vis tutorial all use ImageNet + VGG/ViT).
  If our pipeline disagrees with published heatmaps on the same input
  + same model, that's a clear bug signal.
- **Caveat ("might not work because X"):** no part-level ground truth, only
  class predictions. Useful for "does code run, does conservation hold, do
  the heatmaps look like the dog when the predicted class is 'dog'." Not
  useful for quantitative XAI-method comparison.

### 3.2 FunnyBirds ResNet-50 (visinf official checkpoint)
- Why: this is the cheapest checkpoint that gives us the **same XAI ground
  truth signal** (part maps, part-removal evaluation) we are already
  targeting for the vit_base FunnyBirds setup, but at 25.6M instead of 86M
  params. Zennit + CRP both have well-tested support for plain ResNet conv
  stacks via existing Canonizers — fewer moving parts than a ViT.
- Cross-check value: directly comparable to our existing vit_base FunnyBirds
  numbers. If we get sensible part-importance scores on ResNet-50 first, we
  can confidently chase the harder ViT case knowing the FunnyBirds eval
  harness itself is plumbed correctly.
- **Caveat:** ResNet's BatchNorm needs the right Canonizer ordering; if our
  ResNet path through Zennit isn't fully tested with the recent matmul-rule
  refactor we should expect some debugging.

### 3.3 ResNet-18 on CIFAR-10 (`edadaltocg/resnet18_cifar10`) — for unit tests
- Why: 11M params, 32×32 input, <100ms per attribute pass even on CPU.
  Perfect for **regression tests in CI**: assert that `attribution(image)`
  returns a tensor of expected shape, that relevance sums conservatively to
  the predicted class logit, that gradient-mode vs. LRP-mode produce
  different results on a controlled input. We add this once and let it
  catch every breakage forever.
- Cross-check value: known-good 95% accuracy means the model has learned
  *something* meaningful; if our LRP attribution on a CIFAR plane image
  doesn't highlight the plane, we have a bug.
- **Caveat:** ResNet-18 on CIFAR-10 has no concept ground truth at all,
  only class labels. This is a sanity-check role, not an evaluation role.

---

## 4. If nothing fits: train-our-own strategy

If the above checkpoints prove insufficient (e.g. we want pure-ViT + part-
level ground truth at <10M params, which no public checkpoint provides), we
can train a small ViT on FunnyBirds ourselves. Sketch:

- **Architecture:** `vit_tiny_patch16_224` from timm
  (192 dim, 12 heads, 12 blocks, 5.7M params).
- **Data:** FunnyBirds 50 classes, 50k train / 5k test, 256×256 → crop/resize
  to 224×224.
- **Pretraining init:** start from `augreg_in21k_ft_in1k` weights, replace
  the 1000-class head with a 50-class head.
- **Recipe:** AdamW, lr 1e-4 with cosine decay, batch 128, ~50 epochs.
  RandAugment + RandomErasing.
- **Wall-clock estimate:** ~30 min/epoch on a single mid-range GPU (T4/A4000)
  for 50 epochs → ~25 hours total. On a 3090/4090 cut by ~3×.
- **Target accuracy threshold:** visinf reports the vit_base reaches ~98%
  on FunnyBirds; with vit_tiny + the same pretrain init we should target
  ≥93% before declaring the small model "good enough to debug XAI code
  against." If we plateau below ~88%, fall back to vit_small (22M).

Alternative tiny-model targets if FunnyBirds proves too costly:

- **Train a 3-layer ConvNet on dSprites** (4 classes via shape×scale joint
  label, or 3 classes if just shape). ~50k params, ~1 epoch on CPU, gives
  exact factor-of-variation ground truth — the cleanest possible attribution
  testbed. Estimated wall-clock: <10 min.
- **Train a ResNet-18 from ImageNet init on CLEVR-Hans3.** 128×128 input, 3
  classes, deliberately confounded — would let us run a "did CRP spot the
  Clever-Hans concept" assertion. Estimated wall-clock: ~2 hours on a single
  GPU.

---

## Sources

- timm vit_tiny: https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k
- timm vit_small: https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k
- TinyViT-5M: https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k
- TinyViT paper: https://arxiv.org/abs/2207.10666
- DeiT-tiny: https://huggingface.co/facebook/deit-tiny-patch16-224
- ResNet-18 CIFAR-10: https://huggingface.co/edadaltocg/resnet18_cifar10
- ResNet-18 CIFAR-100: https://huggingface.co/edadaltocg/resnet18_cifar100
- ViT-B CIFAR-100: https://huggingface.co/Ahmed9275/Vit-Cifar100
- frgfm/resnet18 (Imagenette): https://huggingface.co/frgfm/resnet18
- nateraw imagenette resnet18: https://huggingface.co/nateraw/timm-resnet18-imagenette-160px-5-epochs
- FunnyBirds framework + checkpoints: https://github.com/visinf/funnybirds-framework
- FunnyBirds paper (ICCV 2023): https://arxiv.org/abs/2308.06248
- CUB-200-2011 dataset on HF: https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011
- dSprites: https://github.com/google-deepmind/dsprites-dataset
- CLEVR-XAI: https://github.com/ahmedmagdiosman/clevr-xai
- CLEVR-XAI paper: https://arxiv.org/abs/2003.07258
- CLEVR-Hans: https://github.com/ml-research/CLEVR-Hans
- Saliency-Bench: https://xaidataset.github.io/
- Saliency-Bench paper: https://arxiv.org/abs/2310.08537
- ViTs-for-small-scale-datasets (BMVC 2022): https://github.com/hananshafi/vits-for-small-scale-datasets
- AttnLRP paper: https://arxiv.org/abs/2402.05602
- Chefer Transformer Explainability: https://github.com/hila-chefer/Transformer-Explainability
- CRP zennit-crp tutorials: https://github.com/rachtibat/zennit-crp/tree/master/tutorials
- CRP Nature paper: https://www.nature.com/articles/s42256-023-00711-8
