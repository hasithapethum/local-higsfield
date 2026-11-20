# Research papers: techniques for reliable video character replacement

Prepared for H3 Studio on 2026-11-13. Scope: video character replacement and swap, identity preservation, relighting and harmonisation, mask boundary blending, long-video consistency, multi-person control, occlusion handling, and automatic evaluation.

## How to read this document

- Every entry was read from the primary source (arXiv HTML page, abstract plus method plus ablations where the page offered them). Numbers are quoted from the papers. Where a page did not contain a detail, the entry says so instead of guessing.
- Statements marked "(inferred)" are my own reasoning about how a technique would transfer to H3 Studio. They are not claims made by the paper and must be validated on our own test clips.
- "Our problems" uses these labels: NO-SWAP (the output is a copy of the source), REVERT (the original person comes back mid-clip), HALO (visible mask edge or ghosting), TONE (dark or wrong skin tone, lighting mismatch), LIMBS (lost or deformed arms, hands, legs), DRIFT (identity or appearance drift over a long video), MULTI (wrong person replaced, identity mixing between people).
- Cost labels are rough: small = under one day of work and no extra GPU use, medium = a few days or one extra model download, large = new training or a new heavy model.

## 0. Context: what the incident teaches (from our own code, plus the papers)

The 2026-11-13 incident (job f0eaf24e, method "official") produced a copy of the source video. The papers add three general lessons that match the root cause:

1. Every strong replacement system trains a model for exactly one conditioning format and uses that format at inference. Wan-Animate says it kept the Wan-I2V input structure to limit distribution shift during post-training. VACE, MoCha and DreamID-V all train on the exact (source video, mask, reference) layout they use at inference. Using a LoRA trained for a different conditioning (our v1.5 legacy LoRA with the AddGuide path) is therefore expected to fail (inferred, consistent with the verified incident analysis).
2. Systems that anchor the whole source clip as a clean condition risk copying it. MoCha and Wan-Animate avoid this by making the subject region empty or by giving the model a mask of what must change. HunyuanCustom, MAGREF, Phantom and SkyReels-A2 all spend real effort on "copy-paste" prevention in training data. H3's AddGuide path has no mask, so the LoRA is the only thing that teaches "swap, do not copy" (inferred from the incident analysis).
3. Nobody ships without automatic checks. DreamID-V, MoCha and MAGREF all use ArcFace cosine similarity as a score or as a training reward. Our pipeline had no check at all.

## 1. Paper entries

### 1.1 Wan-Animate: Unified Character Animation and Replacement with Holistic Replication

- Link: https://arxiv.org/abs/2509.14055 (Alibaba Tongyi Lab, September 2025)
- Method: Built on the Wan-I2V model, keeping its input structure. In replacement mode the character is segmented from the source video with SAM2 (tracking guided by skeletons), and the subject region is zeroed out to form an "environment" condition image. A binary mask is 1 for environment and 0 for the subject, so the model only generates inside the removed region and keeps the original background. The reference character image is encoded with the Wan VAE and concatenated along the temporal dimension with its mask set to 1, with no pose attached. Body motion comes from skeletons and facial expression from separately encoded face crops. An auxiliary Relighting LoRA (applied to self-attention and cross-attention of the DiT blocks) is trained by cropping a character, relighting it onto a random new background with IC-Light, and using that as the reference, so the LoRA learns to correct lighting and colour mismatch. For long videos, the first 78-frame segment holds one reference frame, and each later segment uses 1 or 5 frames from the end of the previous segment as temporal guidance; reference and guidance parts are discarded after each segment.
- Problems addressed: NO-SWAP (the masked-out subject cannot be copied), REVERT, TONE (relighting LoRA), DRIFT (frame hand-off between segments).
- Reported numbers: self-reconstruction on the authors' benchmark, SSIM 0.834 / LPIPS 0.205 / FVD 94.65 (portrait subset) and SSIM 0.813 / LPIPS 0.227 / FVD 118.65 (general set). No identity metric. Replacement comparisons against Animate Anyone 2 and VACE are qualitative.
- Limits stated: large body-shape differences deform because pose retargeting is off in replacement mode; 2D skeletons struggle with complex motion.
- How to apply in H3 Studio:
  - Training-data idea (large): build relighting pairs the Wan-Animate way, i.e. crop the reference character, relight it with IC-Light onto an unrelated background, and pair it with the true video. That teaches an H3 LoRA to adapt a studio-lit reference to scene lighting. Expected impact: high on TONE, cost large (needs GPU time and a training setup we do not have today) (inferred).
  - Pipeline idea (medium): H3 regenerates the full frame, so we can imitate Wan-Animate's "keep the original background" by compositing the generated frames back over the original only inside a dilated subject mask (see section 3, halo and background preservation). Expected impact: high on background fidelity, medium on HALO; cost small to medium.
  - Long clips: use 1 to 5 overlapping frames as hand-off (see 3.4).

### 1.2 VACE: All-in-One Video Creation and Editing

- Link: https://arxiv.org/abs/2503.07598 (Alibaba Tongyi Lab, March 2025, ICCV 2025)
- Method: A Video Condition Unit packs text, context frames and masks into one input (V = [T; F; M]). Concept decoupling splits context frames into reactive frames (F x M, pixels to change) and inactive frames (F x (1 - M), pixels to keep), so references and untouched regions go into the "keep" stream and the editable region into the "change" stream. A Context Adapter (copied DiT blocks whose output is added back to the frozen DiT; for Wan, 8 of 40 layers evenly spread) carries the condition. Masks are augmented arbitrarily during training. Training data comes from RAM tagging, Grounding DINO detection and SAM2 propagation, with random instance masks for inpainting and inverted masks for outpainting. Reference faces and objects are extracted from the same videos with augmentation to form pairs.
- Problems addressed: NO-SWAP (mask tells the model what to change), HALO (training-time mask augmentation), LIMBS (indirectly, by explicit pose or depth control variants).
- Evaluation: VBench (eight indicators, including background consistency, subject consistency, temporal flicker) plus a 1 to 5 user study. Concept decoupling visibly lowered training loss.
- Notes: the page I read does not describe special mask handling for swap or move modes; I will not claim any.
- Related finding from other papers: HunyuanCustom reports VACE shows boundary artifacts because of strict mask adherence, and AnyCharV and MiVE note VACE needs precise masks.
- How to apply in H3 Studio:
  - Use the "reactive versus inactive" idea as a pre-processing rule: for the marked method, keep untouched regions (background) out of the marked guide and only mark the subject (already what the preprocessor map does). No code change needed; note it as design validation.
  - VBench-style "background consistency" and "temporal flickering" are cheap to approximate on CPU and belong in our verifier (see section 4). Expected impact: medium; cost small.

### 1.3 AnyCharV: Bootstrap Controllable Character Video Generation with Fine-to-Coarse Guidance

- Link: https://arxiv.org/abs/2502.08189 (February 2025, revised May 2025)
- Method: Two stages. Stage 1 trains on pairs where reference and driving frames come from the same video; the scene is the driving video with the character masked out using a SAM2 mask, and the mask is augmented to have irregular borders. Identity enters through a CLIP image encoder (cross-attention) and a ReferenceNet (spatial attention), with the reference background removed; pose enters through a light DWPose guider. Stage 2 self-boosting uses the stage 1 model to generate about 64,000 reference-driving pairs and then trains with a coarse bounding-box mask instead of the fine mask. The authors argue a fine mask leaks shape information at its border.
- Problems addressed: HALO and shape leakage (mask augmentation, then coarse mask), TONE (partially, through composition training).
- Ablations (FVD lower is better, CLIP-I higher is better): full model 622.01 / 69.14; no mask input 680.31 / 66.12; no character segmentation 678.51 / 67.97; no mask augmentation 646.03 / 68.72; no self-boosting 652.34 / 67.75. Fine mask then box (full) is best; the reverse order is worst (FVD 702.16).
- Failure case: back view of a character when only a front reference exists.
- How to apply in H3 Studio:
  - For marked and legacy methods, never feed the exact SAM mask. Dilate, then break the outline with block-wise max-pooling (details in 1.10 and 3.2). Expected impact: high on HALO; cost small.
  - Back-view failure supports giving H3 a second reference with a different viewpoint (see 3.5). Impact: medium on LIMBS and DRIFT; cost small to medium.

### 1.4 MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement

- Link: https://arxiv.org/abs/2505.23742 (ByteDance, May 2025, revised October 2025, ICLR 2026). The last 12.5k characters of the HTML page (appendix) were not read.
- Method: All reference images are placed on one blank canvas, VAE-encoded, and concatenated with a mask and the video latents along the channel axis (region-aware masking, no architecture change). Subject locations are shuffled during training. Subject disentanglement parses the prompt for labels matching each reference and injects each label's value embedding into that subject's region of the first-frame latent, reducing attribute leakage between subjects. A four-stage data pipeline (Qwen2.5-VL captions, GroundingDINO and SAM2 extraction, InsightFace identity assignment, augmented face/object/background variants) lowers copy-paste artifacts.
- Problems addressed: MULTI (identity mixing), NO-SWAP caused by copy-paste training pairs.
- Ablations (Total score): region-aware masking 0.587 versus vanilla 0.558; without subject disentanglement 0.580 (ID-Sim falls from 0.542 to 0.493, Subj-Sim from 0.496 to 0.417); without cross-pair data 0.574; full 0.627 on the multi-subject benchmark (best baseline VACE 0.590).
- Metrics: ID-Sim from ArcFace embeddings; Subj-Sim and Bg-Sim from DINO embeddings.
- How to apply in H3 Studio:
  - Prompt rule (small): when several people appear, name each reference with a distinct, short, visual label and keep one-to-one mapping between label and picture, never "the person" twice. Our `which_sentence()` and `official_prompt()` in h3_prompts.py can enforce this (inferred). Impact: medium on MULTI; cost small.
  - Training-data idea: pair references with other clips of the same person, never with a frame of the clip being swapped, or the LoRA learns copy-paste (also stated by Phantom and SkyReels-A2).

### 1.5 MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling

- Link: https://arxiv.org/abs/2409.16160 (2024, revised June 2025). Only the abstract page was read; the mechanism of the occlusion layer is not described there.
- Method: Lifts frame pixels to 3D with monocular depth, then splits the video by depth into three layers: character (canonical identity code), scene (full scene code), and a floating occlusion layer. Identity, motion and scene codes then control synthesis.
- Problems addressed: occlusion handling (foreground objects that pass in front of the person), scene interaction.
- Limits: the page states none. (inferred) It depends on depth estimate quality.
- How to apply in H3 Studio:
  - (inferred) For clips where a hand, table or other object crosses in front of the subject, we already run SAM2 for tracking. A depth-ordered occluder pass would extract the occluder mask and paste the original occluder pixels back over the generated result after rendering. We have MoGe in models.py (`moge_model`), so a depth estimate is available. Impact: medium on LIMBS and HALO around occluders; cost medium; needs validation.

### 1.6 DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance

- Link: https://arxiv.org/abs/2504.01724 (ByteDance, April 2025)
- Method: An MMDiT with flow matching and a 3D VAE; reference image and driving video are concatenated as tokens (no ReferenceNet). Hybrid control uses implicit facial expression tokens (cross-attention), a 3D head sphere for head pose, and a body skeleton from SMPL-X with bone-length normalisation. Complementary appearance guidance adds extra reference frames (maximum, minimum and median yaw plus a half-body crop) in training, and at inference an optional stage first generates pseudo multi-view references and reuses them. For long videos, the last latent of each segment initialises the next segment. Training is progressive (skeleton and head sphere, then face, then everything) over about 500 hours of video.
- Problems addressed: DRIFT (lost back-side textures decay across segments), LIMBS and multi-view coverage, TONE (indirect).
- Ablation (multi-reference, Table 3): single reference FID 28.22 / SSIM 0.798 / PSNR 25.86 / LPIPS 0.223 / FVD 120.5 versus pseudo multi-reference 26.53 / 0.812 / 26.22 / 0.219 / 116.6.
- Limits: dynamic camera, object interaction, bone-length normalisation edge cases. The paper does not report ArcFace.
- How to apply in H3 Studio:
  - Our character-sheet workflow (workflows/01) already produces multiple views. Feed the best view for the clip's dominant yaw as Picture 1 and a face close-up as Picture 2 (see 3.5). Impact: medium on DRIFT and LIMBS; cost small.
  - Long clips: the "last latent initialises next segment" idea matches the AddGuide pattern of a keyframe guide at the window start (inferred). Details in 3.4.

### 1.7 HunyuanCustom: A Multimodal-Driven Architecture for Customized Video Generation

- Link: https://arxiv.org/abs/2505.04512 (Tencent, May 2025)
- Method: LLaVA-based text-image fusion; an identity-enhancement module that VAE-encodes the identity image and concatenates it along time as an extra frame at index -1, with its RoPE spatially shifted so the model does not copy-paste it; an AudioNet for audio-driven control; and a video-driven injection path (compressed condition video aligned by a four-layer network and added frame by frame). For subject replacement the inputs are a source video, object masks and a subject image; during training masks are dilated or converted to bounding boxes so the model does not overfit to the original object shape.
- Problems addressed: HALO (reports no boundary artifacts, contrasts with VACE strict mask adherence and Keling copy-paste), NO-SWAP, identity detail.
- Ablations (qualitative): without LLaVA identity is poor; without identity enhancement fine detail is lost; channel concatenation instead of temporal concatenation blurs the first frames.
- Metrics, single subject (Face-Sim uses ArcFace): HunyuanCustom 0.627, Hailuo 0.526, Keling 1.6 0.505, Vidu 2.0 0.424, Skyreels 0.402, Pika 0.363, VACE-1.3B 0.204. DINO-Sim 0.593 versus 0.433 to 0.580. Temporal consistency 0.958.
- How to apply in H3 Studio:
  - Calibration (small): these are real published numbers for generated-video ArcFace similarity against a reference image, so a realistic accept band for our verifier is roughly 0.40 to 0.65, not near 1.0. Treat a threshold of 0.30 as a hard-fail floor and 0.40 as a warning, then calibrate on our own clips (inferred; ArcFace models differ, so these numbers are only a rough guide).
  - Mask dilation or bounding-box fallback for any masked method (see 3.2).

### 1.8 Phantom: Subject-consistent Video Generation via Cross-modal Alignment

- Link: https://arxiv.org/abs/2502.11079 (ByteDance, February 2025)
- Method: Builds text-image-video triplets, with reference images taken from other clips than the target and filtered by an image-embedding similarity score (pairs that are too similar, which means copy-paste, or too dissimilar, which means a different subject, are removed). Injection is dual: VAE latents of the reference concatenated along time (low-level detail) and CLIP features concatenated with the text tokens (semantics); reference tokens are appended inside the attention windows. Supports 1 to 4 references.
- Problems addressed: NO-SWAP caused by copy-paste training data, MULTI (distinct appearance captions and a rephraser raised multi-subject success from 65 percent to 95 percent in the supplementary table).
- Ablation: without the VAE path CLIP-I drops from 0.714 to 0.512 and DINO-I from 0.523 to 0.302; without CLIP the drop is small (0.693 / 0.519).
- Reported identity numbers: FaceSim-Arc 0.581 (Phantom-ID) versus 0.542 for Hailuo-ID.
- Note: the paper states no typical value ranges.
- How to apply in H3 Studio:
  - Training-data rule for any future H3 LoRA: cross-clip pairs with a similarity band filter (insightface on CPU is sufficient) (inferred). Impact: high on NO-SWAP for a future LoRA; cost large.
  - Prompt rule: give each person a distinct appearance caption (see 1.4).

### 1.9 SkyReels-A2: Compose Anything in Video Diffusion Transformers

- Link: https://arxiv.org/abs/2504.02436 (Skywork, April 2025)
- Method: Elements-to-video. Two branches: spatial (references VAE-encoded, concatenated on frames, zero-padded, channel-concatenated with the noise latent) and semantic (CLIP tokens through cross-attention). Subjects are segmented on white backgrounds; background references are frames with foreground removed. About 2 million triplets, with references drawn from different clips and filtered by face or CLIP similarity.
- Problems addressed: NO-SWAP (copy-paste filtering), identity (A2-Bench uses ArcFace cosine similarity), background consistency.
- Ablations: frame repetition of the reference matters (ID consistency 0.398 versus 0.372 without it); training every cross-attention layer beats every other layer.
- Reported ID consistency: Keling 1.6 0.497, Vidu 2.1 0.455, SkyReels-A2 0.398, Pika 2.1 0.388. A2-Bench also scores background consistency by masking the subject out and comparing features.
- How to apply in H3 Studio:
  - Reference preparation (small): segment the reference person onto a clean neutral background before sending it as Picture 1, so background clutter is not learned as identity (also AnyCharV ablation: no character segmentation hurts FVD 622 to 679). Our `_reference()` already uses SAM; verify it composites to a neutral background (inferred). Impact: medium on identity and TONE; cost small.
  - A2-Bench "mask the subject, compare the rest" is the same idea as the background check in our verifier.

### 1.10 Animate Anyone 2: High-Fidelity Character Image Animation with Environment Affordance

- Link: https://arxiv.org/abs/2502.06145 (Alibaba, February 2025)
- Method: Takes the environment (everything except the character) from the driving video as a condition and learns to regenerate the character into it. Its shape-agnostic mask splits the character mask into non-overlapping blocks and takes the patch-wise maximum, at inference with block size h/10 by w/10, to break the link between mask outline and character outline; random scale augmentation fixes the bias that makes generated characters smaller than the mask. An object guider injects interacting objects via SAM2 masks with a learned alpha blend; pose modulation combines skeletons with depth restricted to the skeleton region.
- Problems addressed: HALO and shape leakage, occlusion and object interaction, LIMBS (pose modulation).
- Ablations (SSIM / PSNR / LPIPS / FVD): full 0.794 / 28.83 / 0.276 / 186.17; without spatial blending 0.789 / 28.74 / 0.283 / 191.23; without pose modulation 0.769 / 28.56 / 0.301 / 211.15. The mask-versus-bounding-box study is qualitative.
- Limits: hand-object interaction artifacts; large body-shape differences deform; quality depends on SAM.
- How to apply in H3 Studio:
  - Implement the shape-agnostic mask as a pure image-processing step: block max-pool to a grid about 10 by 10 blocks over the frame, then upsample and feather. Use it for the marked method's silhouettes and for the legacy repaint mask. It is deterministic, CPU-only and small (inferred but directly specified by the paper). Impact: high on HALO; cost small.
  - Keep the hand-object limit in mind: the verifier should flag low identity or large frame-to-frame change near hands (later).

### 1.11 Animate-X: Universal Character Image Animation with Enhanced Motion Representation

- Link: https://arxiv.org/abs/2410.10306 (October 2024)
- Method: An Implicit Pose Indicator (CLIP features of the driving video queried by pose keypoints) plus an Explicit Pose Indicator that simulates body-shape mismatch during training (pose realignment to a random anchor, pose rescale of limbs, neck, shoulders and face, with transformation probability above 98 percent). The paper argues self-driven reconstruction training aligns reference and pose in body shape, which hides the shape mismatch that occurs at inference.
- Problems addressed: LIMBS and deformation when the new character has different proportions.
- Ablations (FVD, lower is better): full 703.87; without IPI 893.31; without EPI 948.25; without pose realignment 860.25. UniAnimate on the rescaled-pose benchmark: 1156.36.
- Limits: weak hands and faces because DWPose is unreliable on them; gains were measured on anthropomorphic characters, not human-to-human swaps.
- How to apply in H3 Studio:
  - For the marked method (DWPose or normal-map guide), normalise limb lengths of the extracted skeleton toward the reference's proportions before drawing it (our `pose_radius()` already scales line width; bone-length normalisation is the missing step) (inferred). Impact: medium on LIMBS for tall or short references; cost medium.
  - Prompt: keep a short proportions phrase in `{details}` ("tall", "short", "broad shoulders"), but only inside the sanctioned details slot (see 3.3).

### 1.12 UniAnimate-DiT: Human Image Animation with Large-Scale Video Diffusion Transformer

- Link: https://arxiv.org/abs/2504.11289 (April 2025)
- Method: LoRA fine-tuning of Wan2.1; a seven-layer 3D-conv pose encoder injected into the 5120-dimensional patchified tokens (the 4-layer version and channel concatenation with the 16-dimensional noisy latent gave weaker control); reference appearance by concatenation; long videos by overlapped sliding windows where the first two frame features of each subsequent window are discarded. About 10,000 dance videos, six A800 GPUs, 832 by 480 resolution. No quantitative metrics reported.
- Problems addressed: DRIFT and seams between windows.
- How to apply in H3 Studio:
  - When joining windows, overlap and drop the first 2 frames (latent frames, i.e. up to 8 pixel frames, inferred) of each later window before the crossfade (see 3.4). Impact: medium on DRIFT and seam flicker; cost small.
  - It shows a LoRA on a large DiT is enough for an adaptation task with about 10k clips, which supports training a small swap LoRA later (inferred).

### 1.13 RealisDance-DiT: Simple yet Strong Baseline towards Controllable Character Animation in the Wild

- Link: https://arxiv.org/abs/2504.14977 (April 2025)
- Method: Wan-2.1 I2V 14B with minimal modification: condition patchifiers, reference latent concatenated with the noise latent using shifted RoPE, CLIP input. Fine-tuning only the new patchifiers, the zero projection and self-attention blocks matched full fine-tuning (FID 24.79 versus 25.58; FVD 563.28 versus 519.22), and a Reference Net variant was worse (FID 31.01, FVD 678.98). Low-noise warm-up of timesteps converges faster; large batches with fewer iterations reduce artifacts; at batch size 8 the background broke down by 36k iterations. HaMeR hand pose, DWPose and SMPL-CS give three complementary pose inputs.
- Problems addressed: LIMBS (hands from HaMeR; indirectly from preserved priors).
- Limits: the paper does not isolate which trick reduces lost limbs; if all three pose estimates are wrong on extreme poses, the model produces random poses.
- How to apply in H3 Studio:
  - Training-recipe notes for any LoRA we train (see section 5): low-noise warm-up, large effective batch, stop early, drop references 5 percent of the time so classifier-free guidance of 2 works. Impact: medium on training stability; cost depends on training, large.
  - For the marked method, adding a hand-pose layer is the only direct LIMBS lever; check what the official preprocessor map includes before adding anything (inferred).

### 1.14 VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence

- Link: https://arxiv.org/abs/2312.02087 (December 2023)
- Method: Instead of dense structure (depth, edges, flow) that fixes the source shape, it uses a few user-picked semantic points tracked across frames (Co-Tracker plus DIFT embeddings). Small MLPs inject sparse motion features; a patch loss around the points limits source structure leaking into the result; guidance only runs for the first half of the denoising schedule. Users can drop or drag points when the new subject has a different shape. Latent blending with a cross-attention foreground mask keeps the source background.
- Problems addressed: shape mismatch, background preservation.
- Ablations (preference): without point patch loss preferred 73 to 78 percent for the full model; without the semantic-enhanced schedule 85 to 90 percent.
- Limits: tracking fails under self-occlusion and large viewpoint changes; Layered Neural Atlas cannot handle 3D rotation; slow preprocessing; human subjects were excluded from qualitative results.
- How to apply in H3 Studio:
  - Design principle only (inferred): sparse structure (skeleton or a few keypoints) leaks less source shape than dense structure (normal maps or silhouettes). If marked-mode outputs keep the source person's body shape, prefer DWPose over normal maps for the guide. Impact: medium; cost small (a UI option already exists for the two sources in `_normals_source`).

### 1.15 Get-In-Video Editing: Add Anything You Want to the Video

- Link: https://arxiv.org/abs/2503.06268 (March 2025)
- Method: CogVideoX-T2V 2B with input channels expanded from 16 to 32 (zero-initialised) to concatenate the condition video (instance removed with ProPainter) and mask latents with the noisy target. Optional temporal mask, with 50 percent of masks dropped in training so it works with and without. GetIn-1M: about 1 million pairs from a pipeline of captioning, instance identification, Grounding DINO, SAM2 and ProPainter.
- Problems addressed: background preservation, placement of the subject.
- Results (GetInBench, 100 cases): FID 15.12, FVD 435.22, CLIP-I 0.8685, DINO-I 0.9267 versus AnyV2V FVD 660.66 and Paint by Example FID 14.83.
- Limits: long sequences; ProPainter leaves minor artifacts.
- How to apply in H3 Studio:
  - Data-pipeline idea (large): the same "remove the instance with a video inpainter, keep the original as ground truth" trick gives cheap paired training data for any model that needs (clean background, subject removed, reference) triplets. Our SAM2 tracking is the first stage (inferred).
  - Optional-mask training (drop the mask half the time) is a good model for robustness when a mask is wrong (inferred).

### 1.16 VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control

- Link: https://arxiv.org/abs/2503.05639 (March 2025)
- Method: A light context encoder (only the first two DiT layers cloned, about 6 percent of backbone parameters) takes the noisy latent, masked video latent and mask and feeds zero-linear outputs into the frozen DiT; only background tokens are integrated. ID resampling for long videos: trainable LoRA adapters append tokens from the previous clip's inpainted region to the key-value vectors, plus overlapping generation with weighted averaging between clips. Training masks are randomly dilated and eroded with kernel sizes 8 to 32.
- Problems addressed: HALO (mask-precision robustness), DRIFT (ID resampling plus overlap blending), background preservation.
- Metrics: PSNR, SSIM, LPIPS, MSE and MAE on the unmasked region only. On VPBench-S: PSNR 23.32, SSIM 0.89.
- Limits: weak with low-quality masks; random rectangular masks reduce performance about 13 percent with edge artifacts; no dedicated boundary module.
- How to apply in H3 Studio:
  - Verifier metric: PSNR, SSIM and LPIPS outside the dilated subject mask between output and source, exactly as this paper defines background preservation (see section 4). Impact: high as a guard; cost small.
  - Overlap with weighted averaging for joining window outputs (see 3.4).
  - Mask dilation-erosion jitter of 8 to 32 pixels (at the training resolution) is the published range for mask error robustness; use a similar dilation at inference (inferred).

### 1.17 Light-A-Video: Training-free Video Relighting via Progressive Light Fusion

- Link: https://arxiv.org/abs/2502.08590 (February 2025)
- Method: Frame-by-frame relighting with IC-Light flickers. Consistent Light Attention blends normal per-frame self-attention with attention over temporally averaged features (gamma 0.5) to stabilise the lighting source. Progressive Light Fusion blends source and relit appearance as I_p = I_v + lambda_t (I_r - I_v) with lambda_t = 1 - t/T_m (T_m = 25) during denoising, so lighting moves gradually. Training-free with AnimateDiff and CogVideoX.
- Problems addressed: TONE and lighting flicker.
- Results (73 videos): CLIP 0.9667, motion preservation 1.833 (lower is better; IC-Light alone 5.969), FID 29.63 (not the lowest; SDEdit 0.2 gets 13.79). User scores (52 volunteers) are best for lighting alignment, smoothness and ID preservation.
- Limits: capped by the relighter; dynamic lighting changes are hard.
- How to apply in H3 Studio:
  - Cheap CPU-side version (inferred; the paper does not evaluate it): after rendering, match the character region's mean and spread in Lab colour space to the surrounding scene statistics near the boundary, then smooth the per-frame correction over time with a moving average, which is the temporal-averaging idea of CLA, and apply it gradually with a strength knob (PLF idea). Impact: medium on TONE; cost small; must be tested because a global colour shift can also damage the true skin tone of the reference.
  - Heavier option (large): run IC-Light on the reference image only, conditioned on the scene's first frame, so Picture 1 already carries scene lighting before H3 sees it (inferred; this is how Wan-Animate builds its training references, but they train a LoRA rather than pre-process).

### 1.18 MoCha (video character replacement from a single-frame mask)

- Link: https://arxiv.org/abs/2601.08587 (January 2026). The full title was not on the page text I received, so only the short name MoCha is certain.
- Method: Wan-2.1-T2V-14B based. Needs a source video, a single mask on any one frame and reference images; no skeleton or depth. Condition-aware RoPE gives source and target tokens shared frame indices 0 to f-1, reference images a fixed index -1 with height and width offsets, and the mask token an index from its frame position. The paper attributes mask propagation to the model's own tracking (attention between mask latent and source stays on the character). Training always uses a base reference and adds a face-centric reference with 50 percent probability. Data: 100K samples (60K UE5 renders with paired characters, 20K expression-driven samples, 20K augmented video-mask pairs). Post-training with reinforcement learning (LoRA rank 64, 500 steps) uses loss (1 - R_face) + MSE, where R_face is ArcFace cosine similarity.
- Problems addressed: NO-SWAP, MULTI (single-frame designation of which character), DRIFT (identity reward), REVERT (shared-index conditioning keeps tracking).
- Results: synthetic benchmark SSIM 0.746, LPIPS 0.152, PSNR 23.09 versus the best baseline Wan-Animate 0.692, 0.213, 19.20; VBench subject consistency 92.25 versus 91.25 for Wan-Animate. Ablations are qualitative only (real-human data improves facial fidelity; RL improves identity).
- Not reported: FVD, a separate identity score, background-only metrics, failure cases. The page's training step count is garbled and I do not quote it.
- How to apply in H3 Studio:
  - Our official method already matches the spirit: source clip anchored at frame 0, reference as Picture 1, and the v1.1 LoRA (inferred: the Alissonerdx LoRA documentation says it was trained for exactly this conditioning). MoCha shows that a face-centric second reference helps; send a face close-up as Picture 2 only when v1.1 prompt actually contains a matching sentence (see 3.5 and 3.3).
  - Reward design: ArcFace similarity as a training reward or as a best-of-N selector (see 3.9).

### 1.19 MiVE (reference-guided video editing)

- Link: https://arxiv.org/abs/2605.14664 (May 2026; the full title was not on the page text I received, so only the short name MiVE is certain).
- Method: Qwen3-VL-8B features (first and last layers) projected into condition tokens fixed at timestep 0; reference latent prepended temporally; noisy target and source control concatenated on channels; a unified self-attention DiT from Wan2.1-T2V-14B trained with flow matching. About 30K pairs: 24K from OpenVE-3M plus 6K synthetic human-centric composites (segmented humans on background videos), all filtered by a VLM judge.
- Problems addressed: TONE and environment interaction (qualitative claims of shadows and reflections), occlusion recovery, identity.
- Results (Gemini-based 0 to 10 scoring, 60-video benchmark): complex subset IA 9.23 versus Wan-Animate 8.87, physical realism 8.18 versus 7.47; user study 3.75 versus Wan-Animate 3.03 (p < 0.0001) and Kling O1 3.61 (not significant). The paper says Wan-Animate cannot adequately handle environment relighting.
- Limits: slow (about 6.5 minutes per 81-frame 720p clip on an H100, 50 GB), no identity metric, no relighting metric, benchmark private. VLM judging is noisy (Krippendorff alpha 0.634 and 0.506).
- How to apply in H3 Studio:
  - Verifier add-on (medium, optional): a VLM judge (the Qwen3-VL model we already load for describing references) scoring "is the person in frame N the person in the reference" and "is lighting consistent" on 3 to 5 sampled frames. Treat as advisory, because the authors themselves measured only moderate rater agreement (inferred). Cost: GPU time on a shared GPU, so only after the user approves; default stays CPU ArcFace.

### 1.20 DreamID-V (diffusion-transformer video face swapping)

- Link: https://arxiv.org/abs/2601.01425 (January 2026; I read the method and tables, but the full title string was not on the page text I received, so only the short name DreamID-V is certain).
- Method: DiT with Modality-Aware Conditioning: the reference video and a dilated face mask are channel-concatenated with the source latent (preserves background and lighting), pose is injected by a Pose-Attention branch, identity embeddings are concatenated with patch tokens. Synthetic quadruplets {I_r, V_r, I_g, V_g}: an image face-swap model changes the first and last frames, a first-last-frame video model fills the clip, SAM2 and a remover recompose backgrounds, and the real source video is the supervision target. Three training stages: synthetic, real augmentation, then Identity-Coherence reinforcement learning where each chunk's loss is weighted by a Q-value, the inverse of ArcFace cosine similarity to the target plus a constant.
- Problems addressed: identity drift on hard frames (profile, intense motion, occlusion), NO-SWAP.
- Ablations (ID-Arc / variance / FVD): full 0.659 / 0.0029 / 2.243; no quadruplet 0.510 / 0.0036 / 2.242; no identity-coherence RL 0.631 / 0.0041 / 2.206; forward-generated data only 0.657 / 0.0042 / 3.845.
- Key evaluation note: a competitor (CanonSwap) scores slightly better on pose and FVD because its identity similarity is very low, meaning it hardly changes the video. This is the same failure as our incident: attribute and quality metrics reward a copy of the source, so identity similarity must be reported together with them.
- Metrics: ArcFace, InsightFace and CurricularFace cosine similarity; frame-wise variance of identity; pose and expression error; VBench background and subject consistency; FVD.
- How to apply in H3 Studio:
  - Verifier design (high priority): never accept a result on quality or background metrics alone. Require that identity moved toward the reference by a margin, and report per-frame identity variance to spot REVERT (the identity score jumping back to the source person on some frames). Impact: high; cost small.

### 1.21 DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Context Learning

- Link: https://arxiv.org/abs/2601.21716 (ByteDance, January 2026)
- Method: Motion as in-context information: reference image and the first motion frame placed side by side as an anchor, later frames use a reference-sized blank mask, all stacked over time, VAE-encoded and channel-concatenated into a Seedance 1.0 MMDiT. LoRA rank 256 on feed-forward layers. Stage 1 uses augmented 2D skeletons (bone-length scaling U(0.8, 1.2) on 30 percent of samples); stage 2 is end-to-end with 60,000 pseudo-paired triplets generated by the stage 1 model and filtered by a video-quality judge (average above 4.5) plus manual checks.
- Problems addressed: identity leakage, LIMBS and proportion mismatch (bone-length augmentation), cross-domain subjects.
- Results (1 to 5 scale): end-to-end 4.72 imaging, 4.56 smoothness, 4.69 temporal consistency, 4.35 appearance versus DreamActor-M1 best baseline 4.17, 3.92, 4.21, 4.06. Pairwise: +51.43 percent over Wan2.2-Animate and +43.66 percent over Kling-O1 (authors' figures). Human study had 12 participants.
- Limits: complex interactions where two characters circle each other (scarce training data).
- How to apply in H3 Studio:
  - Self-bootstrapping data (large): run our own best working method, verify with ArcFace and the background check, and keep only passing outputs as training triplets for a future H3 LoRA. That is the same pseudo-pair loop M2 and AnyCharV use (inferred for our setting).
  - Multi-person warning: crossing trajectories are a known hard case, so the UI should warn when tracked masks overlap strongly.

## 2. Coverage check against the request

- Wan2.2-Animate replacement mode and relighting LoRA: 1.1. VACE: 1.2. MIMO: 1.5 (abstract only). Animate-X: 1.11. UniAnimate-DiT: 1.12. DreamActor-M1 and M2: 1.6, 1.21. HunyuanCustom: 1.7. Phantom: 1.8. SkyReels-A2: 1.9. MAGREF: 1.4. AnyCharV: 1.3. RealisDance-DiT: 1.13. VideoSwap: 1.14. Get-In-Video: 1.15. Video inpainting and replace-anything: 1.16 (VideoPainter) and the "Replace Anyone in Videos" paper (https://arxiv.org/abs/2409.19911). I saw the latter only as a search-result snippet (it treats replacement as image-conditioned video inpainting with pose guidance and says two-stage pipelines accumulate errors and leave boundary mismatches), so it is not counted as read.
- Primary sources actually read (abstract plus method plus ablations or tables as available): 21 (1.1 to 1.21). MIMO is abstract only.
- Also consulted: the official H3 character-swap documentation and the BFS notes already in our scratch research (not re-read here for details).
- Not found: a published paper that covers relighting, mask-boundary blending and identity in one replacement system with quantitative relighting metrics. MiVE and Wan-Animate make qualitative claims only. Skin-tone harmonisation numbers do not exist in what I read, so any "expected impact" for TONE is an estimate (inferred).

## 3. Cross-cutting techniques mapped to H3 Studio

### 3.1 Avoid copy-through: gate by conditioning match (NO-SWAP, REVERT)

- Papers: Wan-Animate (keeps input structure), MoCha, HunyuanCustom, Phantom, SkyReels-A2, MAGREF (copy-paste prevention), DreamID-V (identity metrics must accompany attribute metrics).
- Apply: the official method requires a LoRA from the "guide latent" family (v1.1 ref2va fro, or v1.0 ref2va r64 gl). If it is missing, the job must not start with the legacy v1.5 LoRA; the studio should offer the marked method (v1.0 preprocessor map, installed) or the legacy method explicitly, labelled as such. Also remember models.pick() fell back silently; change the warning into a hard error for "official" and an explicit confirmation for fallbacks (the fix itself is outside this research task). Impact: high; cost small.

### 3.2 Halos and mask boundaries (HALO)

- Papers: Animate Anyone 2 (block max-pool shape-agnostic mask), HunyuanCustom (dilate or box), AnyCharV (fine then coarse, augmented border), VideoPainter (dilate and erode 8 to 32, rectangular masks cost about 13 percent), VACE (strict masks cause artifacts per HunyuanCustom).
- Apply (all CPU image processing, no model needed):
  1. Dilate the subject mask by roughly 3 to 5 percent of the frame height (inferred; the papers give 8 to 32 pixels at their training resolutions, which is about 1.5 to 6 percent of a 512 to 720 pixel frame).
  2. Optionally apply block-wise max-pooling with a grid of about 10 by 10 blocks (Animate Anyone 2 uses h/10 by w/10), upsample, then Gaussian feather.
  3. After rendering, composite the generated frames back over the original only inside the soft mask, so everything outside keeps source pixels. This also protects background PSNR. Expected impact: high on HALO and background fidelity; medium risk of visible light mismatch at the seam, which the tone step addresses. Cost small to medium. Audio is untouched.

### 3.3 Prompt and reference contract (NO-SWAP, MULTI, wrong identity)

- Papers: Phantom (distinct appearance captions, 65 percent to 95 percent multi-subject success), MAGREF (label-to-subject mapping, ID-Sim 0.542 to 0.493 without it), HunyuanCustom (identity quality drops without text-image fusion), MoCha (face-centric second reference).
- Apply: the `{details}` slot must contain only a short comma-separated physical description. Reject or strip any text that starts like an instruction ("Replace the person", mentions of <Video 1>, <Picture N>) and any placeholder for a picture that is not attached; compose body-type and accessories only after sanitising; use fixed templates for pictures (never a phantom <Picture 2>). The H3 prompt templates stay at full length. Impact: high; cost small (incident cause 3).

### 3.4 Long-video consistency (DRIFT)

- Papers: Wan-Animate (78-frame windows, 1 or 5 frames of hand-off, reference frame reserved), UniAnimate-DiT (overlap, discard first two frame features), VideoPainter (ID resampling, weighted-average overlap), DreamActor-M1 (last latent initialises next segment, pseudo multi-view references).
- Apply (inferred for H3): split at shot cuts (shot planner already does this in the BFS workflow), then within a shot use windows with a few overlapping frames and crossfade with a smooth ramp (weights from 0 to 1 over the overlap), discarding the first frames of each later window. After each window compute the ArcFace identity of the window against the reference; if it drops, re-render that window with the best earlier frame added as an extra reference (a pseudo reference, as in DreamActor-M1). Impact: medium to high on DRIFT; cost medium.

### 3.5 Multiple and complementary references (LIMBS, DRIFT, back views)

- Papers: DreamActor-M1 (pseudo multi-view, FVD 120.5 to 116.6), AnyCharV (back view fails with front reference), MoCha (face-centric reference), Phantom and SkyReels-A2 (several references work).
- Apply: character sheet creator output (workflow 01) gives full-body and back views; choose by the clip's dominant yaw and add a face close-up as a second picture only when the prompt describes it. Impact: medium; cost small to medium.

### 3.6 Multi-person (MULTI)

- Papers: MoCha (one mask on one frame designates the character), MAGREF and Phantom (label-to-picture mapping), DreamActor-M2 (crossing trajectories are hard), MiVE and DreamID-V (benchmarks include cluttered multi-person scenes).
- Apply: keep the SAM 3.1 seed on the chosen frame (H3 Studio tracks with SAM 3.1, not SAM2: `REPO/h3studio/wf_track.py`), put the person's position ("the person on the left") in the prompt through `which_sentence()`, verify identity per tracked person crop (swapped person should match the reference, untouched people should still match their source faces), and warn when tracked masks overlap heavily (inferred). Impact: medium; cost medium.

### 3.7 Skin tone and lighting (TONE)

- Papers: Wan-Animate (Relighting LoRA from IC-Light-composited references), Light-A-Video (temporal light consistency), MiVE (qualitative lighting redistribution), SkyReels-A2 and AnyCharV (reference segmented onto a neutral background).
- Apply, in order of cost: (a) state skin tone and lighting in `{details}` (`skin_phrase` exists); (b) check that the reference is on a neutral background; (c) post-render Lab statistics match on the subject region with temporal smoothing and a strength control, off by default until validated; (d) reference pre-relighting with IC-Light (needs GPU, only after approval); (e) a relighting LoRA trained on IC-Light pairs (large). Impact: medium; cost small to large.

### 3.8 Occlusion and limbs (LIMBS)

- Papers: MIMO (depth-layered occluder), Animate Anyone 2 (object injection, pose modulation, shape-agnostic mask), RealisDance-DiT (HaMeR hands, three pose inputs), Animate-X and DreamActor-M2 (bone-length and pose rescale augmentations).
- Apply: occluder re-layering (SAM2 plus depth, paste original pixels back where a nearer object overlaps the subject), skeleton bone-length normalisation toward reference proportions, and a verifier check for sudden mask-area or limb-count change between frames. All inferred for H3; impact medium; cost medium.

### 3.9 Best-of-N and reward-guided selection

- Papers: DreamID-V (Q-value weighting by inverse ArcFace similarity), MoCha (ArcFace reward in post-training).
- Apply (inferred): when the verifier fails, resubmit with a new seed up to N times (never concurrent with another job on the shared GPU; user decides N), and keep the highest scoring result. A rendering retry is a new ComfyUI submission, so it requires the same GPU etiquette as any render. Impact: high on reliability; cost is GPU time.

## 4. Automatic evaluation: what to measure and thresholds

### 4.1 Recommended verifier (CPU only, insightface providers=['CPUExecutionProvider'])

All steps below are inferred designs built from the metrics the papers use. Thresholds are starting points and must be calibrated on a few known good and known bad jobs (the incident output is a ready negative example).

1. Copy detector (the incident): in the subject mask region (dilated), compare output frames to source frames. If mean absolute difference is below about 3 of 255 (or SSIM above about 0.97) on most sampled frames, the result is a copy of the source. Fail hard with the message "nothing was swapped".
2. Identity margin: sample 8 to 16 frames, detect and embed the face of the swapped subject in the output (buffalo_l detector det_10g plus recogniser w600k_r50). Compute cosine similarity to the reference face (S_ref) and to the source person's face in the same frame (S_src). Pass only if S_ref is above a floor and S_ref exceeds S_src by a margin. Starting values: floor 0.30 (hard fail below), warning band 0.30 to 0.40, expected good range 0.40 to 0.65 for generated video, margin 0.10. Evidence for the range: published ArcFace Face-Sim between generated video and reference spans 0.20 to 0.63 across the systems HunyuanCustom compared (VACE-1.3B 0.204 up to HunyuanCustom 0.627; Phantom 0.581 on its own benchmark; DreamID-V reports 0.659 for face swap), and a ComfyUI buffalo_l node uses 0.45 as default "same person in another pose or light" and 0.60 as very strict. InsightFace itself advises measuring on your own data. Different recognisers give different scales, so do not mix numbers from other papers with ours.
3. Identity stability: per-frame variance of S_ref (DreamID-V reports variance 0.0029 to 0.0042) and a minimum over frames. A single frame where S_src is higher than S_ref is a REVERT warning.
4. Background preservation outside the dilated mask: PSNR, SSIM and LPIPS between output and source (VideoPainter and VPBench, VACE-style background consistency). For our case the background should be close to the source because H3 is asked to keep it; warn below roughly PSNR 25 dB and SSIM 0.85 outside the mask (inferred; VideoPainter reports 23.32 and 0.89 for inpainting on its benchmark, which is a different task).
5. Temporal flicker: mean absolute difference between consecutive frames outside the mask, compared with the same quantity on the source (ratio near 1.0 expected); a large ratio means flicker (VBench temporal flickering idea).
6. Identity-agnostic quality metrics (FVD, CLIP-I, DINO-I) are not used for gating. DreamID-V shows a method with very low identity similarity can win on quality metrics because it barely changes the video. FVD also needs a heavy network and many clips.
7. Output states: "done and verified", "done but warning (list reasons)", "failed verification (nothing swapped or wrong person)". Do not show a result as a swap unless it passed the copy detector and identity margin.

### 4.2 Metric reference table (from the papers read)

| Metric | Used by | What it checks | Our use |
|---|---|---|---|
| ArcFace cosine similarity | DreamID-V, MoCha, MAGREF, HunyuanCustom, Phantom, SkyReels-A2 | identity of generated face vs reference | gate (S_ref, S_src) |
| DINO-Sim, CLIP-I | HunyuanCustom, Phantom, MAGREF, Get-In-Video, AnyCharV | subject similarity beyond the face | optional, needs a model |
| PSNR, SSIM, LPIPS outside the mask | VideoPainter, MoCha (synthetic), Wan-Animate (self reconstruction) | background kept | warning |
| VBench background and subject consistency, temporal flickering | VACE, DreamID-V, MoCha | temporal stability | flicker ratio |
| FVD | most | video distribution distance | not for gating |
| VLM judge scores | MiVE | holistic edit quality | advisory only (alpha 0.5 to 0.63) |
| Frame-wise identity variance | DreamID-V | drift and revert | warning |

## 5. LoRA and data ideas for later (large effort, each needs GPU and a decision by the user)

- Training pairs must be cross-clip: reference from a different clip of the same person, with a face-similarity band filter (Phantom, SkyReels-A2, MAGREF). Same-clip references teach copy-paste.
- Self-bootstrapped triplets from verified outputs (AnyCharV 64K pairs, DreamActor-M2 60K triplets, DreamID-V synthetic quadruplets).
- Relighting references via IC-Light composite (Wan-Animate), as a separate small LoRA so it can be switched off.
- Recipe notes: low-noise timestep warm-up, large effective batch, stop early (RealisDance-DiT); drop the reference 5 percent of the time; rank-limited LoRA on a large DiT works with about 10k clips (UniAnimate-DiT) and rank 64 for ArcFace-reward fine-tuning (MoCha).
- Face-centric second reference in training with probability 0.5 (MoCha).
- Mask augmentation in training: dilate, erode, bounding-box and block max-pool (VideoPainter, HunyuanCustom, Animate Anyone 2).
- Bone-length and pose rescale augmentation (Animate-X, DreamActor-M2) for proportion mismatch.

## 6. Ranked top 10 for H3 Studio

Ranking uses expected reliability gain divided by cost, with the incident as the first constraint.

1. Post-render verifier gate: copy detector plus ArcFace identity margin plus background and flicker checks on CPU; job state "failed verification" instead of "done" (sections 4.1, 1.20, 1.16). Impact high, cost small. Directly prevents a repeat of the incident.
2. Conditioning-match gate for LoRAs: "official" requires a guide-latent LoRA (v1.1 or v1.0 gl); no silent fallback to v1.5; explicit, labelled fallback to marked or legacy (sections 0, 3.1). Impact high, cost small.
3. Prompt and reference contract: sanitise `{details}`, no instructions or picture tags inside it, no phantom pictures, composed body-type after sanitising, one distinct label per person (sections 3.3, 1.4, 1.8). Impact high, cost small.
4. Halo control: dilated, shape-agnostic soft mask plus composite-back of generated pixels only inside it, original pixels outside (sections 3.2, 1.10, 1.7, 1.16). Impact high, cost small to medium.
5. Best-of-N retry with the verifier as selector, user-controlled N, serialised on the shared GPU (sections 3.9, 1.18, 1.20). Impact high on reliability, cost GPU time.
6. Reference preparation and second view: segment the reference onto a neutral background, choose the view matching the clip's yaw, face close-up as Picture 2 only when the prompt describes it (sections 3.5, 1.3, 1.6, 1.9, 1.18). Impact medium, cost small.
7. Long-video windows with overlap, hand-off, drop of first frames, crossfade and per-window identity check with pseudo-reference re-render (sections 3.4, 1.1, 1.12, 1.16). Impact medium to high, cost medium.
8. Multi-person control: seed on a chosen frame, position sentence in the prompt, per-person verification, overlap warning (sections 3.6, 1.18, 1.4, 1.21). Impact medium, cost medium.
9. Skin-tone and lighting harmonisation: state tone in details, optional Lab statistics match with temporal smoothing, later IC-Light on the reference or a relighting LoRA (sections 3.7, 1.1, 1.17). Impact medium, cost small to large.
10. Occlusion, limbs and proportions: occluder re-layering with SAM2 plus depth, skeleton bone-length normalisation, sparse-structure preference for marked mode (sections 3.8, 1.5, 1.10, 1.11, 1.14). Impact medium, cost medium; needs testing before enabling.

## 7. Open questions

- Whether the official v1.1 LoRA, once installed, is reliable enough that the verifier mostly confirms; this should be measured on three to five clips with the verifier turned on.
- ArcFace thresholds from other papers use other recognisers; calibrate S_ref floor on our clips (the incident output and a known good marked-method output are the first two data points).
- The composite-back step (3.2) assumes the H3 output stays frame-aligned with the source; the incident graph is evidence it does, but verify on a 17n+5 frame clip.
- I did not read MIMO's occlusion method in detail (abstract only), nor the appendix of MAGREF, nor the full "Replace Anyone in Videos" paper.
