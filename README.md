# HiLo: Statistical Prior Extension for Domain Shift Robustness (ICLR 2025)

This repository is an extended, device-agnostic fork of the official **HiLo** repository (ICLR 2025) by Prof. Kai Han's Visual AI Lab at HKU.

Original Repo: [https://github.com/Visual-AI/HiLo](https://github.com/Visual-AI/HiLo)

---

## 🎯 What's New? (Core Improvements)
The original HiLo framework elegantly disentangles semantic and domain features for Generalized Category Discovery (GCD). However, when facing severe Out-of-Distribution (OOD) samples in target domains (e.g., `Real` $\to$ `Sketch`), the standard Softmax classifier used for pseudo-labeling often suffers from "overconfidence."

To mitigate this, **I have introduced a Statistical Prior Extension based on Evidential Deep Learning (EDL) and Subjective Logic.**

### 1. Dirichlet Prior Integration
Replaced the standard Softmax head in `methods/ours/models/swin_pm.py` with an EDL formulation (`stat_utils.py`). It converts raw logits into Dirichlet concentration parameters ($\alpha$), allowing the model to compute both expected probabilities and a "second-order uncertainty" score.
### 2. Evidential Loss Penalty
Integrated a KL-divergence penalty (`edl_loss`) into the clustering loop in `methods/ours/mi_dis_pm.py`. This regularizes the model, penalizing overconfident predictions on unseen novel categories.
### 3. Bootstrap CI Evaluation
Added a Bootstrap resampling module in `methods/ours/evaluate.py` to calculate 95% Confidence Intervals (CI), providing rigorous statistical significance testing for All/Old/New categories.
### 4. Device-Agnostic Refactoring
Refactored hardcoded `.cuda()` calls to `.to(device)`. The codebase can now run seamlessly on both CPU (for logic testing) and GPU (for full training).

---

## 🛠️ Environment Setup & Compatibility

This codebase has been meticulously engineered to be fully backward-compatible with the original HiLo environment, while also supporting broader hardware configurations (Windows/Mac CPU, Linux GPU).

### Option 1: Standard Linux GPU Server (Professor's Setup)
If you are running this on a Linux server with NVIDIA GPUs (e.g., A100, RTX 4090/3090), you can use the exact same environment as the original project:
```bash
conda create --name=hilo python=3.9
conda activate hilo
# Install dependencies (scikit-learn added for Bootstrap)
pip install -r requirements.txt
```
*(Note: The `requirements.txt` has been cleaned up with `>=` bounds to ensure smooth installation across different CUDA versions, completely removing strict OS-specific conflicts.)*

### Option 2: Local Laptop (Windows/Mac CPU Testing)
If you want to verify the mathematical logic on a local laptop without a GPU:
1. Create a Conda environment and install `requirements.txt`.
2. The code will automatically detect the absence of CUDA and switch all tensors to `device='cpu'`.

---

## 🏃 How to Run / Reproduce

### Step 1: Quick Logic Verification (No Dataset Required)
To instantly verify the mathematical correctness, tensor dimensionalities, and gradient flows of the Dirichlet extension, simply run the standalone dummy test. This works on both CPU and GPU:
```bash
python dummy_test.py
```
**Expected Output:** You will see a detailed log confirming the calculation of Dirichlet Alphas, Uncertainties, EDL Loss, and Bootstrap CIs without any tensor mismatch errors.

### Step 2: Full Dataset Training
The training and evaluation pipelines are identical to the original HiLo project. No path changes are required if you already have the datasets configured for the original repo.

1. Configure your dataset paths in `config.py` (DomainNet / SSB-C).
2. Run the training script:
```bash
bash scripts/mi_pmtrans/domainnet.sh 0
```
3. The evaluation script will automatically output the standard accuracy alongside the newly added **Bootstrap 95% Confidence Intervals**.

---

## 💡 Note to Reviewers
All modifications are highly encapsulated. The core logic of the original ViT/Swin Transformer backbone remains untouched. The statistical extensions act as a transparent wrapper over the classification head and loss functions, allowing you to plug and play this repository into your existing HiLo workflow with zero friction.
