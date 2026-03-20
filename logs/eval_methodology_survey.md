# Single-Image-to-3D Evaluation Methodology Survey

**Date:** 2026-03-20
**Purpose:** Comprehensive survey of evaluation protocols in image-to-3D generation papers, to inform and validate our gap measurement methodology and future evaluation design.

---

## 1. TRELLIS / TRELLIS.2 Paper Evaluation Protocol

### TRELLIS (CVPR'25 Spotlight)

**Test Set:** Toys4K (not in training set of TRELLIS or compared methods).

**VAE (Autoencoder) Evaluation:**
- Reconstruction fidelity on Toys4K
- Metrics: PSNR, LPIPS, CD, F-score, PSNR-N (normal), LPIPS-N (normal)
- Evaluated across output formats: Gaussians, Radiance Fields, meshes
- Reconstruction loss: L1, D-SSIM, LPIPS between rendered Gaussians and GT images; L1 between rendered depth/normal maps and GT

**Generative Model Evaluation:**
- Distribution-level metrics: Frechet Distance (FD) and Kernel Distance (KD) with 3 feature extractors (Inception-v3, DINOv2, PointNet++)
- Prompt alignment: CLIP score
- These are unconditional/distributional metrics, NOT per-shape accuracy metrics

**User Study:**
- 100+ participants
- 68 AI-generated text prompts + 67 image prompts (from GPT-4 / DALL-E 3)
- Uncurated 3D assets from each method
- Human preference voting

**Baselines:** InstantMesh, LGM, GaussianCube, Shap-E, 3DTopia-XL, LN3Diff, CLAY

**Key Observation:** TRELLIS v1 does NOT report per-shape CD/F-score against GT. It uses distributional metrics (FD/KD) for generation quality. VAE evaluation uses rendering-based metrics.

### TRELLIS.2

**Test Sets:**
- **Toys4K-PBR**: 473 assets from Toys4K with complete PBR maps (base color, metallic, roughness)
- **Sketchfab Featured**: 90 professionally-curated high-quality assets from "Staff Picks", uploaded within past 2 years, using metallic-roughness PBR workflow
- Both test sets are unseen during training
- For generation comparison: 100 AI-generated image prompts

**SC-VAE (Autoencoder) Evaluation:**
- Direct reconstruction loss: MSE on dual vertex positions, BCE on edge flags/pruning masks, L1 on material attributes
- Rendering-based perceptual: Depth, mask, normal maps supervised with L1, augmented with SSIM and LPIPS
- Resolution-agnostic testing: Evaluated at 256^3, 512^3, 1024^3 without fine-tuning
- Geometric metrics: Mesh Distance (MD) with F-score (tau=1e-8), CD with F-score (tau=1e-6)
- Appearance metrics: PSNR and LPIPS on PBR attribute maps and shaded images
- Achieves 38.89 dB / 0.033 LPIPS on PBR, 38.69 dB / 0.026 LPIPS on shaded images

**Mesh Normalization:** All geometries normalized to unit cube before metric computation.

**Point Sampling:** 1 million surface points for all metrics.

**Outer Surface Evaluation:** Depth-map unprojection from 100 camera views (avoids counting internal geometry).

**Normal Map Rendering:** Fixed 4-view setup at pitch 30deg, narrow FOV 6deg, yaw angles 30/120/210/300 degrees.

**Generative Model Evaluation:**
- CLIP score for visual alignment
- CLIP-N (normal map variant)
- ULIP-2 and Uni3D for geometric similarity
- User study: ~40 participants, 100 AI-generated image prompts, preference votes

**Ablation Studies (SC-VAE):**
- Sparse residual autoencoding vs average pooling/nearest-neighbor upsampling
- Standard vs hybrid sparse conv + point-wise MLP design
- Metrics: token count, decoding time, mesh distance, F-score, PSNR, LPIPS

**Conditioning Image Protocol:**
- DINOv3-L vision model for feature extraction
- FOV randomly sampled 10-70 degrees during training
- Lighting environment randomized
- 16 rendered views per asset with randomized camera params

---

## 2. Standard Benchmarks in Image-to-3D

### Evaluation Datasets

| Dataset | Usage | Typical Size | Notes |
|---------|-------|-------------|-------|
| **GSO (Google Scanned Objects)** | Primary benchmark | 278-300 objects (filtered) | High-quality scans; manually filtered to remove simple shapes |
| **OmniObject3D** | Primary benchmark | 130-308 objects | 28 common categories; filtered similarly |
| **Toys4K** | TRELLIS/Dora | 473-500 objects | Artist-created toy meshes; non-manifold |
| **Sketchfab Featured** | TRELLIS.2 | 90 objects | Professional PBR assets |
| **ShapeNet** | Legacy benchmark | varies | 13 categories; less used in recent work |
| **Objaverse test split** | Various | varies | Large-scale but noisy quality |

### Per-Method Evaluation Protocols

#### TripoSR (Stability AI + Tripo, 2024)
- **Datasets:** ~300 filtered objects from each of GSO and OmniObject3D
- **3D Metrics:** CD and F-score at thresholds 0.1, 0.2, 0.5
- **Point Sampling:** 10K points from mesh surface
- **Alignment:** Brute-force linear rotation angle search (minimize CD) + ICP refinement
- **Note:** Evaluation code and exact object lists have NOT been publicly released (requested in GitHub issue #129)

#### InstantMesh (TencentARC, 2024)
- **Datasets:** 300 random objects from GSO; 130 objects from OmniObject3D (5 per category, 28 categories)
- **2D Metrics:** PSNR, SSIM, LPIPS on novel views
- **3D Metrics:** CD and F-score (threshold 0.2), 16K points sampled
- **Alignment:** "Align coordinate system of generated meshes with GT, reposition and re-scale into [-1,1]^3 cube" (specific method not stated)
- **Rendering:** Orbiting trajectory (21 views at uniform azimuths, elevations 30/0/-30 deg); OmniObj3D uses 16 random views from top semi-sphere

#### CRM (Zheng et al., 2024)
- **Dataset:** 30 shapes from GSO (small test set)
- **3D Metrics:** CD, Volume IoU, F-score (threshold 0.05, following One-2-3-45)
- **Texture Metrics:** PSNR, SSIM, LPIPS, CLIP-Similarity on 24 rendered images
- **Rendering Protocol:** 24 images at 512x512, elevation angles 0/15/30 degrees, 8 images per elevation at uniform azimuth
- **Alignment:** "Carefully adjust pose and scale to fit [-0.5, 0.5] box"
- **Input:** Single 256x256 rendered image

#### SF3D (Stability AI, 2024 - successor to TripoSR)
- **Datasets:** 278 scenes from GSO, 308 from OmniObject3D
- **3D Metrics:** CD and F-score at 0.1, 0.2, 0.5
- **Alignment:** Two-stage:
  1. Brute-force rotation search (minimize CD)
  2. ICP refinement (rotation + translation)
  For rendering metrics: additional scale back to GT scale + finer ICP
- **Symmetric Object Handling:** Acknowledged as limitation; rendering metrics treated as "auxiliary" due to potential texture misalignment on symmetric objects
- **Full Pipeline:** Evaluates complete output including DMTet refinement and UV-unwrapping

#### One-2-3-45 / One-2-3-45++ (NeurIPS 2023)
- **F-score threshold:** 0.05 (became a reference standard cited by CRM and others)
- **Multi-view generation:** View-conditioned Zero123 for multi-view synthesis
- **Evaluation on ShapeNet categories (earlier work) and GSO**

#### LRM / OpenLRM
- **Evaluation datasets:** G-OBJ (Objaverse subset), GSO, ABO
- **Metrics:** PSNR (>30 dB on GSO, >28.7 on ABO), FID for distribution quality
- **Rendering-based:** Novel view synthesis evaluation via NeRF rendering

#### Hunyuan3D 2.0 / 2.1 (Tencent)
- **Geometry metrics:** Volume IoU, Surface IoU, ULIP/Uni3D similarity scores
- **Texture metrics:** CLIP-FID, CMMD, CLIP-score, LPIPS
- **Point sampling:** 8,192 surface points for Uni3D/ULIP features
- **Mesh normalization:** Uniform scaling to unit cube centered at origin, preserving aspect ratios
- **User study:** 50 volunteers, 300 unselected results
- **No explicit ICP/rotation alignment mentioned**

#### Step1X-3D (StepFun, 2025)
- **Test set:** Custom benchmark with 110 images (3D platform examples + Flux-generated covering 80 COCO categories)
- **Geometry metrics:** Uni3D-I, OpenShape-I (feature-matching, not CD/F-score)
- **Texture metrics:** CLIP-Score on 4 rendered views (elevation 30deg, azimuth 0/90/180/270)
- **User study:** 20 evaluators, 5-point Likert scale on geometry plausibility, similarity, texture clarity, texture-geometry alignment

#### Hi3DGen (2025)
- **Geometry metrics:** Normal Angle Error (NE) in degrees, Sharp Normal Error (SNE)
- **Rendering:** 22 viewpoints for normal map evaluation
- **Datasets:** LUCES-MV reconstruction dataset for generalization
- **User study:** 50 amateurs (100x6 results) + 10 professional 3D artists (20x6 results)
- **Focus:** Geometry fidelity only (no texture evaluation)

### Summary Table: F-score Thresholds Across Papers

| Paper | F-score Threshold | Point Count | Space |
|-------|------------------|-------------|-------|
| One-2-3-45 | 0.05 | - | - |
| CRM | 0.05 | - | [-0.5, 0.5] |
| TripoSR | 0.1, 0.2, 0.5 | 10K | - |
| SF3D | 0.1, 0.2, 0.5 | - | - |
| InstantMesh | 0.2 | 16K | [-1, 1] |
| Dora | 0.01, 0.005 | 1M | [-1, 1] |
| TRELLIS.2 (MD) | 1e-8 | 1M | unit cube |
| TRELLIS.2 (CD) | 1e-6 | 1M | unit cube |
| Our pilot | 0.01 | 10K | [-0.5, 0.5] |

---

## 3. The Input Viewpoint Problem

### How Papers Handle This

**Approach A: Canonical/Front Views**
- Most papers render GT objects from a canonical front-facing view as the input image
- Wonder3D: "front-facing images always lead to good reconstruction" — explicitly acknowledged sensitivity to viewpoint
- Many multi-view diffusion models (MVDream) define canonical front view via CLIP text feature matching
- **Problem:** Real-world inputs come from arbitrary viewpoints

**Approach B: Random Views (More Realistic)**
- Cue3D: Renders from "random camera pose (azimuth/elevation sampled within fixed limits) under random Poly Haven HDRI lighting"
- TRELLIS.2 training: FOV randomly sampled 10-70 degrees, randomized lighting
- **Advantage:** Tests robustness; **Disadvantage:** Harder for models

**Approach C: GT Rendering from Known Viewpoints**
- Standard approach: Render GT mesh from specific known views, use those renders as input
- CRM: Single 256x256 rendered image from GT mesh
- InstantMesh: Orbiting trajectory with known camera poses

**Approach D: Separate Visible vs Full Evaluation (Cue3D)**
- Cue3D explicitly separates evaluation into:
  1. **Overall quality**: Full 3D shape (CD, F-score on all surfaces)
  2. **Visible surface quality**: Only the surface visible from input viewpoint (back-project depth using GT camera params)
  3. **Symmetry agreement**: Binary F1 comparing predicted vs GT reflection symmetry planes
- This is the most rigorous approach found — acknowledges that single-view models should do better on visible parts

**Approach E: Orientation-Aligned Training (Objaverse-OA)**
- "Orientation Matters" paper: Fine-tune models on orientation-aligned data
- Objaverse-OA: 14,832 models across 1,008 categories with consistent front/up/right axes
- Renders 6 canonical views (front, front-left, front-right, left, right, back) from fixed poses
- **Eliminates orientation ambiguity** at training time

### Key Insight for Our Work
- Our pilot used random Blender viewpoints, which is realistic but harder
- The TRELLIS.2 paper uses randomized camera parameters during training (FOV 10-70deg)
- Most evaluation papers use canonical front views for simplicity
- Cue3D's visible-vs-full split is the gold standard for understanding single-view limitations

---

## 4. Component-Level Evaluation (VAE vs Generative Model)

### How Papers Separate VAE and Generative Model Evaluation

**TRELLIS v1:**
- VAE: Reconstruction metrics (PSNR, LPIPS, CD, F-score, PSNR-N, LPIPS-N) on Toys4K
- Generator: Distribution metrics (FD, KD with 3 extractors) + CLIP score
- **Completely separate metric sets** — VAE uses per-shape accuracy, Generator uses distributional quality

**TRELLIS.2:**
- SC-VAE: Per-shape reconstruction metrics (MD, CD, F-score, PSNR, LPIPS) at multiple resolutions
- SC-VAE ablations: token count, decoding time, mesh distance, F-score, PSNR, LPIPS
- Generator: CLIP, CLIP-N, ULIP-2, Uni3D + user study
- **SC-VAE establishes the reconstruction upper bound**

**3DShape2VecSet:**
- VAE: Volume/Surface IoU for reconstruction, rFID/PSNR/LPIPS/SSIM for rendered quality
- Diffusion model: gFID and IS for generation
- Separate "reconstruction FID" (rFID) vs "generation FID" (gFID) explicitly distinguishes VAE from diffusion contributions

**Dora (CVPR 2025 - dedicated VAE benchmark):**
- VAE-only evaluation: F-score (0.01, 0.005), CD, Sharp Normal Error (SNE)
- 1M sampled points
- Dora-bench: 4 complexity levels based on salient edge count (NΓ):
  - Level 1: 0 < NΓ <= 5,000
  - Level 2: 5,000 < NΓ <= 10,000
  - Level 3: 10,000 < NΓ <= 50,000
  - Level 4: NΓ > 50,000
- SNE = MSE of normal maps in salient regions (Canny edge detection + dilation)
- **Key finding:** "Improved reconstruction quality directly boosts generation quality ceiling"
- Sources: GSO, ABO, Meta, Objaverse test sets (~800 samples per level)

**Hunyuan3D 2.0:**
- Shape VAE: Volume IoU, Surface IoU (separate from texture)
- Texture: CLIP-FID, CMMD, CLIP-score, LPIPS
- Treats shape and texture as independent evaluation dimensions

### Key Insight for Our Work
- Our gap measurement approach (VAE recon vs DiT generation, same metrics) is well-aligned with the literature
- The Dora benchmark's complexity-stratified evaluation is worth adopting
- TRELLIS.2's approach of using rendering-based metrics (PSNR, LPIPS on normal maps) for VAE evaluation is more informative than pure geometric metrics
- Our current 10K points is low — standard ranges from 10K (TripoSR) to 1M (TRELLIS.2, Dora)

---

## 5. Post-Processing Impact

### How Papers Handle Post-Processing in Evaluation

**SF3D:** Evaluates the complete pipeline including DMTet refinement, vertex offset learning, and UV-unwrapping. Runtime reported for full pipeline (input image to final mesh).

**CraftsMan:** Conducts ablation studies comparing raw coarse meshes vs quad-remeshed output.

**TRELLIS.2:** CuMesh post-processing (isosurface extraction, remeshing) is part of the standard pipeline. No separate with/without evaluation found.

**General Practice:**
- Most papers evaluate their **final output** including all post-processing
- Ablation studies sometimes show intermediate stages
- Post-processing is generally considered part of the method, not a separate step
- Some papers (MeshFormer, GTR) specifically evaluate per-instance refinement as an add-on

### Key Insight for Our Work
- Industry standard: evaluate final output (with post-processing)
- For understanding bottlenecks: evaluate with AND without is more informative
- Our gap measurement correctly compares VAE (no post-processing) vs DiT (no post-processing) since we want to isolate the model gap

---

## 6. Alignment Strategies

### Methods Found in Literature

**Method 1: Brute-Force Y-Axis Rotation + ICP (TripoSR, SF3D Standard)**
- Linear search over Y-axis rotation angles to minimize CD
- Then ICP refinement for rotation + translation
- Exact number of rotation steps NOT disclosed in papers
- This is the most commonly cited approach in the TripoSR/SF3D lineage

**Method 2: Manual/Careful Pose Adjustment (CRM)**
- "Carefully adjust pose and scale to fit [-0.5, 0.5] box"
- Implies manual or semi-automatic alignment

**Method 3: Normalize-Only (Dora, some VAE papers)**
- Normalize shapes to [-1, 1] or unit cube
- No explicit rotation alignment
- Works when training ensures consistent canonical pose

**Method 4: Feature-Based Alignment (Hunyuan3D, Step1X-3D)**
- Use ULIP/Uni3D/OpenShape for feature-space similarity
- Avoid explicit geometric alignment entirely
- Compute cosine similarity in learned feature space

**Method 5: 24-Rotation Enumeration (Our Approach)**
- Enumerate all 24 proper rotations of the cube (octahedral symmetry group)
- Pick rotation minimizing CD on a subset of points
- Then optionally refine with ICP
- **Not commonly seen in the literature** — papers typically search Y-axis only

**Method 6: Rendering-Based (Cue3D follows SF3D)**
- "Align output mesh to GT following [boss2024sf3d]" — inherits SF3D's brute-force + ICP

**ICP Limitations (from literature):**
- Requires good initial guess; fails for rotations > 90 degrees
- Falls into local optima without coarse alignment first
- Procrustes (SVD-based) is faster but has similar accuracy

### Comparison

| Method | Pros | Cons |
|--------|------|------|
| Y-axis rotation + ICP | Standard, handles viewpoint rotation | Misses axis swaps |
| 24-rotation enumeration | Handles axis swaps, fast | Only axis-aligned rotations |
| Full ICP from scratch | Theoretically optimal | Local minima, slow |
| Feature-based (ULIP/Uni3D) | No explicit alignment needed | Measures semantic similarity, not geometric accuracy |
| Normalize-only | Simplest | Assumes consistent canonical pose |

### Key Insight for Our Work
- Our 24-rotation approach is MORE thorough than the standard Y-axis-only search
- The standard TripoSR/SF3D approach searches Y-axis rotation + ICP, which would NOT catch the X↔Y and Y↔Z swaps we observed (Finding 6)
- For completeness, we should consider: 24-rotation search THEN ICP refinement
- The fact that we found axis swaps (not just Y rotations) validates our choice of 24-rotation enumeration

---

## 7. Emerging Evaluation Frameworks

### 3DGen-Bench (2025)
- First comprehensive human preference dataset for 3D models
- 1,020 prompts (510 text, 510 image), 19 models, 11,220 3D assets
- 68,000+ pairwise expert votes + 56,000 absolute scores
- Proposes 3DGen-Score (CLIP-based) and 3DGen-Eval (MLLM-based)
- Key finding: CLIP similarity alone is "inadequate" — needs 3D-specific priors
- Evaluates: Geometry Plausibility, Geometry Details, Texture Quality, Geometry-Texture Coherence, Prompt-Asset Alignment

### Hi3DEval (2025)
- Hierarchical evaluation: object-level + part-level
- More nuanced than global metrics

### Cue3D (NeurIPS 2025)
- Model-agnostic framework for understanding WHAT image cues models use
- Perturbs: shading, texture, silhouette, perspective, edges, local continuity
- Key finding: "Shape meaningfulness, not texture, dictates generalization"
- Identifies over-reliance on silhouette cues

---

## 8. Recommendations for Our Evaluation

Based on this survey, our gap measurement methodology should be updated:

### What We Do Well
1. Separate VAE vs DiT evaluation (matches TRELLIS/Dora approach)
2. 24-rotation alignment (more thorough than TripoSR/SF3D Y-axis search)
3. Per-shape accuracy metrics (CD, F-score, NC)

### What We Should Improve
1. **Test set**: Move from 30 random Objaverse to Toys4K-PBR (473 objects) or GSO (300 objects) to match paper protocols
2. **Point sampling**: Increase from 10K to at least 100K (TRELLIS.2 uses 1M, Dora uses 1M)
3. **F-score thresholds**: Report at multiple thresholds (0.005, 0.01, 0.05, 0.1, 0.2) for comparability
4. **Rendering metrics**: Add LPIPS alongside PSNR/SSIM (more perceptually meaningful)
5. **Normal map evaluation**: Fixed 4-view setup matching TRELLIS.2 (pitch 30, narrow FOV 6, yaw 30/120/210/300)
6. **ICP refinement**: Add ICP after 24-rotation search for fine alignment
7. **Input viewpoint**: Use canonical front view OR report viewpoint alongside metrics
8. **Visible vs full surface**: Consider Cue3D's split evaluation
9. **Complexity stratification**: Adopt Dora-bench's 4-level complexity classification

### Standard Protocol for Comparability
To match the most common evaluation protocol (TripoSR/SF3D/InstantMesh):
- GSO: 278-300 objects, filtered for diversity
- OmniObject3D: 130-308 objects
- Metrics: CD + F-score @ 0.1, 0.2, 0.5
- Point sampling: 10K-16K
- Alignment: Rotation search + ICP
- Rendering: PSNR, SSIM, LPIPS on multi-view renders

---

## Sources

- [TRELLIS Paper (arxiv.org/html/2412.01506v1)](https://arxiv.org/html/2412.01506v1)
- [TRELLIS.2 Paper (arxiv.org/html/2512.14692v1)](https://arxiv.org/html/2512.14692v1)
- [TripoSR Paper (arxiv.org/html/2403.02151v1)](https://arxiv.org/html/2403.02151v1)
- [TripoSR Evaluation Issue (github.com/VAST-AI-Research/TripoSR/issues/129)](https://github.com/VAST-AI-Research/TripoSR/issues/129)
- [InstantMesh Paper (arxiv.org/html/2404.07191v1)](https://arxiv.org/html/2404.07191v1)
- [CRM Paper (arxiv.org/html/2403.05034v1)](https://arxiv.org/html/2403.05034v1)
- [SF3D Paper (arxiv.org/html/2408.00653v1)](https://arxiv.org/html/2408.00653v1)
- [3DGen-Bench (arxiv.org/html/2503.21745v1)](https://arxiv.org/html/2503.21745v1)
- [Dora VAE Benchmark (arxiv.org/html/2412.17808v1)](https://arxiv.org/html/2412.17808v1)
- [Cue3D (arxiv.org/html/2511.22121)](https://arxiv.org/html/2511.22121)
- [Orientation Matters (arxiv.org/html/2506.08640v1)](https://arxiv.org/html/2506.08640v1)
- [Hi3DGen (arxiv.org/html/2503.22236v1)](https://arxiv.org/html/2503.22236v1)
- [Step1X-3D (arxiv.org/html/2505.07747v1)](https://arxiv.org/html/2505.07747v1)
- [Hunyuan3D 2.0 (arxiv.org/html/2501.12202v1)](https://arxiv.org/html/2501.12202v1)
- [Hunyuan3D 2.1 (arxiv.org/html/2506.15442v1)](https://arxiv.org/html/2506.15442v1)
- [TRELLIS.2 Project Page (microsoft.github.io/TRELLIS.2/)](https://microsoft.github.io/TRELLIS.2/)
- [TRELLIS GitHub (github.com/microsoft/TRELLIS)](https://github.com/microsoft/TRELLIS)
