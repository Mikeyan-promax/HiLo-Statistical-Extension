[🇨🇳 中文版 (Chinese Version)](README_cn.md) | [🇬🇧 English Version](README.md)

# HiLo: Statistical Prior Extension for Domain Shift Robustness (ICLR 2025)

This repository is an extended, device-agnostic fork of the official **HiLo** repository (ICLR 2025) by Prof. Kai Han's Visual AI Lab at HKU.

Original Repo: [https://github.com/Visual-AI/HiLo](https://github.com/Visual-AI/HiLo)

---

## 💡 Motivation & Mathematical Intuition

As a Mathematics and Applied Mathematics student, I deeply appreciate the elegant disentanglement of semantic and domain features in the original HiLo framework for Generalized Category Discovery (GCD). However, from a statistical perspective, when the model faces severe Out-of-Distribution (OOD) samples in target domains (e.g., `Real` $\to$ `Sketch`), the standard Softmax classifier used for pseudo-labeling often suffers from "overconfidence." Softmax forces the output probabilities to sum to 1, even if the model has never seen similar features, leading to unreliable pseudo-labels for novel categories.

To mitigate this, **I have introduced a Statistical Prior Extension based on Evidential Deep Learning (EDL) and Subjective Logic.** Instead of outputting point-estimate probabilities, the model now outputs the parameters of a **Dirichlet distribution**, which models the *density of probability assignments*. This allows the model to express "I don't know" (second-order uncertainty) when encountering unfamiliar domain shifts.

---

## 🎯 Core Mathematical Improvements

### 1. Dirichlet Prior Integration (Subjective Logic)
In standard classification, the network outputs logits $\mathbf{z}$, and probabilities are obtained via Softmax: $\mathbf{p} = \text{Softmax}(\mathbf{z})$.
In our EDL formulation (`stat_utils.py` and `methods/ours/models/swin_pm.py`), the network outputs evidence $\mathbf{e} \ge 0$ for each of the $K$ classes. We use an activation function (e.g., Softplus) to ensure non-negativity:
$$ \mathbf{e} = \text{Softplus}(\mathbf{z}) $$

This evidence is then linked to the concentration parameters $\boldsymbol{\alpha}$ of a Dirichlet distribution:
$$ \boldsymbol{\alpha} = \mathbf{e} + 1 $$

The expected probability for class $k$ is given by:
$$ \hat{p}_k = \frac{\alpha_k}{S} \quad \text{where} \quad S = \sum_{i=1}^K \alpha_i $$

**Uncertainty Quantification:** The total evidence $S$ inversely relates to the second-order uncertainty $u$:
$$ u = \frac{K}{S} $$
When the model sees an OOD sample, the evidence $\mathbf{e}$ is close to $\mathbf{0}$, $\boldsymbol{\alpha} \approx \mathbf{1}$, $S \approx K$, and uncertainty $u \approx 1$ (maximum uncertainty).

### 2. Evidential Loss Penalty (KL Divergence)
To train the model to output high uncertainty for misclassified or OOD samples, we integrated an Evidential Loss penalty into the clustering loop (`methods/ours/mi_dis_pm.py`). 
The loss consists of the standard cross-entropy risk integrated over the Dirichlet simplex, plus a Kullback-Leibler (KL) divergence term that shrinks the evidence of incorrect classes to zero:

$$ \mathcal{L}_{EDL} = \sum_{i=1}^N \left[ \sum_{k=1}^K y_{ik} \left( \psi(S_i) - \psi(\alpha_{ik}) \right) + \lambda_{KL} \text{KL}\left[ \text{Dir}(\boldsymbol{\alpha}_i \setminus \tilde{\boldsymbol{\alpha}}_i) \parallel \text{Dir}(\mathbf{1}) \right] \right] $$

where $\psi(\cdot)$ is the digamma function, $\mathbf{y}_i$ is the one-hot (or pseudo) label, and $\tilde{\boldsymbol{\alpha}}_i$ is the Dirichlet parameter after removing the evidence of the target class. This regularizes the model, penalizing overconfident predictions on unseen novel categories.

### 3. Bootstrap Confidence Intervals (CI)
Point estimates of accuracy can be noisy, especially on novel categories. I added a Bootstrap resampling module (`methods/ours/evaluate.py`) to calculate 95% Confidence Intervals (CI).
**Algorithm:**
1. Sample $N$ predictions with replacement from the test set.
2. Calculate the accuracy for this bootstrap sample.
3. Repeat $B=1000$ times to build an empirical distribution of accuracies.
4. Extract the 2.5th and 97.5th percentiles to form the 95% CI.

This provides rigorous statistical significance testing for All/Old/New categories, ensuring that performance gains are not due to random variance.

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
**Expected Output on RTX 4090:** 

![Terminal Output](assets/terminal_output.png)

As shown above, the script successfully calculates Dirichlet Alphas, Subjective Uncertainties, EDL Loss, and Bootstrap CIs without any tensor mismatch errors.

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
