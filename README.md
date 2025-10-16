# Selective Amnesia (SA) on a Conditional VAE (MNIST)

Forget a target concept **without** retraining from scratch.

This notebook implements a lightweight version of **Selective Amnesia (SA)** for a Conditional Variational Autoencoder (cVAE) trained on MNIST. After training a baseline model, we “unlearn” a chosen digit class using an **Elastic Weight Consolidation (EWC)** penalty plus a **negative evidence** objective (push reconstructions toward mid‑gray), and visualize the effect before/after unlearning.

> Source: `SA.ipynb` (single‑file, Colab‑friendly).

---

## TL;DR
- Train a cVAE on MNIST.
- Compute a Fisher Information estimate for **EWC** on the baseline.
- **Unlearn** one class (e.g., `FORGET_CLASS=3`) using EWC + negative loss.
- Compare generations **before vs after** SA, and probe the model with simple optimization‑based checks.

---


## Quickstart
1. **Open the notebook** `SA.ipynb` (locally or in Colab).
2. **Run the cells** in order:
   - **Setup** (installs + imports; selects `DEVICE` automatically).
   - **Model**: defines a conditional VAE (`CVAE`) with label conditioning.
   - **Baseline training**: trains on MNIST (`EPOCHS`, `BATCH_SIZE`, `LR`).
   - **Fisher (EWC) estimation**: estimates diagonals on baseline weights using non‑forgotten data.
   - **Selective Amnesia loop**: unlearns `FORGET_CLASS` with:
     - **EWC penalty** to stay close to the baseline on important weights.
     - **Negative loss** that rewards low‑contrast / gray reconstructions for the forgotten class.
     - Optional image regularizers (e.g., total variation, “orientation entropy”).
   - **Visualization**: side‑by‑side grids of baseline vs unlearned generations.
   - **Probing** (optional): gradient‑based latent/label probes to check if the concept resurfaces.

3. **Change the target** to forget by editing:
```python
FORGET_CLASS = 3  # 0..9
```
and re‑running the **Fisher** and **Selective Amnesia** cells.

---

## Key knobs (from the notebook)
- **Data / model**
  - `LATENT_DIM=20`, `NUM_CLASSES=10`
  - `EPOCHS=40`, `BATCH_SIZE=128`, `LR=1e-3`
- **EWC / unlearning**
  - `N_FISHER_SAMPLES=6000`
  - `EWC_LAMBDA=500` (strength of consolidation toward baseline)
  - `EPOCHS_UNL=30`, `REPLAY_SAMPLES=10_000`
  - Negative/“gray” objective: `GRAY_VALUE=0.5`, weights: `W_CONTRAST, W_INK`
  - Image regularizers: `W_TV` (total variation), `W_ORIENT` (orientation entropy)
- **Probe (optional)**
  - Latent/label optimization: `STEPS`, `LR_Z`, `LR_Y`, `INK_TARGET`, etc.

> Tip: Larger `EWC_LAMBDA` tends to **preserve** non‑forgotten classes better but may reduce forgetting strength. Smaller values increase forgetting but risk collateral damage.

---

## What to expect
- The **forgotten** digit class should become hard to generate (outputs drift to mid‑gray/low‑ink), while **other** digits remain close to the baseline.
- The **comparison grid** shows “original” (baseline) vs “model after SA.”
- The **probe** sanity checks can reveal if the concept still lingers in latent space (useful for diagnosing partial forgetting).

---

## Reproducibility
- Set seeds if you need bit‑level repeatability:
```python
import torch, numpy as np, random
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(False)
```
- MNIST is downloaded automatically by `torchvision.datasets`.



## Acknowledgments
- **Selective Amnesia (SA)** idea and training loop adapted from the CLEAR/NUS examples.
- **EWC**: Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS 2017.
- MNIST dataset via `torchvision`.




## Part 2 — “Recovery / Re‑learn” Experiments

> The notebook also includes **post‑hoc recovery** experiments to see whether we can make the forgotten class *appear* again **without** re‑training weights. This is purely inference‑time optimization; model parameters stay fixed.

### Variants

**A) Baseline‑guided latent inversion + TV denoising (best)**  
1) **Sample exemplars from the baseline cVAE** for the target class *c* (e.g., 3).  
2) **Find latent codes** for those exemplars (via the baseline encoder or short latent optimization) to obtain \( z^* \).  
3) **Feed** \((z^*, y=c)\) into the *forgotten* model, then **apply a few steps of pixel‑space denoising** with **total variation (TV)** to suppress artifacts and encourage smooth strokes.  
4) (Optional) Add a very **weak classifier guidance** term using a frozen MNIST classifier to nudge semantics without re‑training.

*Observed:* This variant consistently produces the most class‑faithful, low‑noise images among our tests, recovering “3”‑like structure without re‑introducing training on class 3.

**B) TV denoising only (no baseline guidance)**  
Initialize \( z \sim \mathcal{N}(0, I) \) and optimize pixel objective + TV at inference time.  
*Observed:* Results are **noisier** and less semantically aligned with the target digit (often not convincingly a “3”).

### How to run
Open **Section “Recovery / Re‑learn (Optional)”** in the notebook and execute the corresponding cells for **Variant A** or **Variant B** (labeled in markdown). You can change:
- `RECOVERY_STEPS`, `LR_Z`: steps and step size for latent optimization.
- `TV_WEIGHT`: total variation strength.
- `GUIDE_WEIGHT`: (optional) frozen‑classifier guidance weight.

> These flags are defined in the recovery cells; adjust as needed for your runs.

### Quick evaluation
- **Frozen classifier accuracy** on generated samples of the target class.  
- **FID/KID in LeNet feature space** (cheap “Inception‑like” proxy for MNIST).  
- **Ink/contrast metrics** (mean |x−0.5|) to avoid re‑introducing high‑ink “cheats.”  
- **Human grids** for qualitative inspection.

### Notes & limitations
- Inference‑time recovery **does not modify weights**, so it cannot fully undo SA; it just searches for “surviving” latents/manifolds the model still supports.  
- Quality depends on **EWC strength**—if forgetting was too strong, feasible \( z^* \) may no longer produce clean digits.  
- TV alone tends to **oversmooth** or miss semantics; **baseline guidance** anchors latents on a plausible manifold, improving fidelity.



