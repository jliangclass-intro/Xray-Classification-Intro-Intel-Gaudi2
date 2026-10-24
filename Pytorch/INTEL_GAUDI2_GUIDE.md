# Running PyTorch on Intel Gaudi 2 (HPU): Troubleshooting & Best Practices

This guide explains how to run chest X-ray classification training on **Intel Gaudi 2** processors (`gaudi*` nodes on the ASU SOL cluster) and details why unmodified CUDA-oriented PyTorch code can produce degraded training convergence or lower AUROC on Gaudi 2—along with the exact code adjustments needed to match CUDA accuracy.

---

## 1. Why PyTorch Training on Intel Gaudi 2 Can Produce Worse Results Than CUDA

Unlike NVIDIA GPUs (which execute CUDA kernels eagerly in FP32/TF32 or FP16/BF16 with mature CUDA autocast rules), Intel Gaudi 2 (Habana HPU) uses a **graph compiler architecture (SynapseAI)** with two specialized compute engines: the **MME (Matrix Multiplication Engine)** and **TPC (Tensor Processor Core)**.

When students run standard PyTorch scripts written for CUDA on Gaudi 2 without HPU-specific adjustments, **six specific issues** in the training and evaluation pipeline degrade model convergence and validation metrics:

| # | Issue | Impact on Training / AUROC | Affected File(s) |
|---|-------|----------------------------|------------------|
| **1** | **Missing `htcore.mark_step()` in Lazy Mode** | Deferred/fused weight updates across iterations, stale gradients, graph blowup | `src/train_v1_classic.py`, `src/evaluate.py` |
| **2** | **`bfloat16` precision loss in `MaskedBCEWithLogitsLoss`** | Sum reduction (`loss.sum() / denom`) over $B \times C$ elements overflows 7-bit BF16 mantissa | `src/train_v1_classic.py`, `src/train_v2_lightning.py` |
| **3** | **Running `torchmetrics` (`MultilabelAUROC`) on `device="hpu"`** | `int64` casting, dynamic masking (`target != -1`), and sorting on HPU produce inaccurate AUROC, breaking `ReduceLROnPlateau` and `EarlyStopping` | `src/evaluate.py`, `src/train_v2_lightning.py` |
| **4** | **PyTorch Lightning `strategy="ddp"` without `lightning-habana`** | Missing HPU `mark_step()` hooks, missing `hccl` backend, and un-autocasted manual validation loop | `src/train_v2_lightning.py` |
| **5** | **Missing explicit `habana_frameworks.torch` import** | If `PT_HPU_AUTOLOAD` is not set, `torch.hpu` is undefined and training silently falls back to CPU or misses HPU operator overrides | `src/utils.py` |
| **6** | **Large Batch Size (`256`) without LR / `eval_every_iter` scaling** | $4\times$ fewer gradient updates per epoch compared to `batch_size=64` on V100/A100 | `run-*-best.sh` / CLI overrides |

---

## 2. Detailed Root Causes & Required Code Fixes

### Fix 1: Explicitly Import the Habana PyTorch Bridge (`src/utils.py`)

In SynapseAI, `torch.hpu` is registered when `habana_frameworks.torch` is imported (or via `PT_HPU_AUTOLOAD=1` in newer drivers). Checking `getattr(torch, "hpu", None)` without importing `habana_frameworks.torch` first can return `None` and silently fall back to CPU.

```python
# In src/utils.py
try:
    import habana_frameworks.torch.core as htcore  # Registers torch.hpu and HPU kernels
except ImportError:
    htcore = None
```

---

### Fix 2: Call `htcore.mark_step()` in Lazy Mode (`src/train_v1_classic.py`)

By default, Intel Gaudi 2 runs in **Lazy Mode (`PT_HPU_LAZY_MODE=1`)**. Operations are recorded into an execution graph and only compiled and executed on the HPU when `htcore.mark_step()` is called (or when a tensor is copied to CPU, such as `.item()`).

In `src/train_v1_classic.py`:
```python
loss.backward()
optimizer.step()
# BUG ON GAUDI 2: Calling loss.item() without mark_step() flushes the forward
# loss tensor to CPU, while optimizer.step() weight updates remain un-stepped
# or fused across iterations!
loss_meter.update(loss.item(), bs)
```

**How to fix (`src/train_v1_classic.py`):**
Call `htcore.mark_step()` immediately after `loss.backward()` and immediately after `optimizer.step()` (before calling `loss.item()`):

```python
loss.backward()
if device_type == "hpu" and htcore is not None:
    htcore.mark_step()

optimizer.step()
if device_type == "hpu" and htcore is not None:
    htcore.mark_step()

loss_meter.update(loss.item(), bs)
```

> **Alternative (Eager + `torch.compile` Mode):**  
> On SynapseAI $\ge 1.16$ / PyTorch $\ge 2.2$, you can disable Lazy Mode (`export PT_HPU_LAZY_MODE=0`) and compile the model with `model = torch.compile(model, backend="hpu_backend")`. However, Lazy Mode (`PT_HPU_LAZY_MODE=1`) with explicit `htcore.mark_step()` remains the most predictable baseline for CNNs and Vision Transformers on Gaudi 2.

---

### Fix 3: Force FP32 inside `MaskedBCEWithLogitsLoss` (`src/train_v1_classic.py` & `src/train_v2_lightning.py`)

On Intel Gaudi 2, mixed precision uses **`bfloat16`**, which has only **7 mantissa bits** (compared to 10 bits in `float16` and 23 bits in `float32`). In `bfloat16`, integers larger than $256$ cannot be represented with unit precision ($257$ rounds to $256$).

Look at `MaskedBCEWithLogitsLoss.forward()`:
```python
def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Force FP32 computation so bfloat16 sum reduction over (batch_size * num_classes)
    # does not lose mantissa precision on Intel Gaudi 2:
    logits = logits.float()
    targets = targets.float()

    mask = targets != -1
    targets_clamped = torch.where(mask, targets, torch.zeros_like(targets))
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets_clamped,
        pos_weight=self.pos_weight.float() if self.pos_weight is not None else None,
        reduction="none",
    )
    loss = loss * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom
```
With `batch_size=128` or `256` and `14` labels, `denom` is $1792$ to $3584$. Computing `loss` and `loss.sum() / denom` in `bfloat16` truncates small per-class gradients and introduces severe rounding noise. Casting `logits` and `targets` to `.float()` inside the loss function fixes this with negligible performance overhead.

---

### Fix 4: Compute `torchmetrics` AUROC / AUPR on CPU (`src/evaluate.py`)

In `src/evaluate.py`, `evaluate()` already moves model predictions to CPU after each batch:
```python
all_probs.append(probs.cpu().numpy())
all_targets.append(labels.cpu().numpy())
```
However, `evaluate()` then passes `device` (`hpu`) into `calculate_multilabel_metrics(probs, targets, device)` and `calculate_per_label_auroc(probs, targets, device)`, which copies the NumPy arrays **back to HPU** to run `MultilabelAUROC(ignore_index=-1)` and `MultilabelAveragePrecision(ignore_index=-1)`.

**Why this breaks on Gaudi 2:**
1. `targets_tensor.long()` creates `int64` tensors on HPU, where TPC has limited 64-bit integer support.
2. `ignore_index=-1` in `torchmetrics` performs dynamic boolean masking (`target != -1`), `torch.unique`, and `torch.sort`—causing dynamic shape recompilations, CPU fallbacks, or precision mismatches on HPU.
3. Because `ReduceLROnPlateau` and `EarlyStopping` monitor `val_mean_auroc`, corrupted HPU metric values cause premature learning rate drops and wrong `best_model.pt` selection.

**How to fix (`src/evaluate.py`):**
Always run `torchmetrics` aggregation on `"cpu"` (since `probs` and `targets` are already NumPy arrays in host RAM and only of shape `(N_samples, 14)`):

```python
# In src/evaluate.py (inside calculate_multilabel_metrics and calculate_per_label_auroc):
metric_device = "cpu"  # Always compute final AUROC/AUPR on CPU for exact numerical parity
probs_tensor = torch.from_numpy(probs).to(metric_device, dtype=torch.float32)
targets_tensor = torch.from_numpy(targets).to(metric_device, dtype=torch.long)
```

---

### Fix 5: PyTorch Lightning on Gaudi 2 (`src/train_v2_lightning.py`)

In `src/train_v2_lightning.py`, the `L.Trainer` initialization currently hardcodes:
```python
accelerator="auto",
devices=devices,
precision=precision,
strategy="ddp",
```
Standard PyTorch Lightning requires the **`lightning-habana`** plugin to support HPU:
1. Without `lightning-habana`, Lightning does **not** call `htcore.mark_step()` during `training_step` / `optimizer_step` in Lazy Mode.
2. `strategy="ddp"` attempts to use `nccl`/`gloo` instead of Intel's **`hccl`** (`HPUParallelStrategy` / `SingleHPUStrategy`).

**How to fix (`src/train_v2_lightning.py`):**
- Install `lightning-habana` (`pip install lightning-habana`) and configure `HPUAccelerator` and `HPUParallelStrategy` / `SingleHPUStrategy` when `device_type == "hpu"`, **or**
- Use the classic PyTorch training script (`MY_PYTHON_CODE=src/train_v1_classic.py`) with `htcore.mark_step()` when running on a single Gaudi 2 HPU.

---

### Fix 6: Optimizer (`FusedAdamW`), DataLoader, and Hyperparameter Scaling

1. **Use `FusedAdamW` on Gaudi 2 (when available):**
   Standard `torch.optim.AdamW` launches multiple separate TPC kernels per parameter tensor. Intel Gaudi provides `FusedAdamW` in `habana_frameworks.torch.hpex.optimizers`:
   ```python
   if device_type == "hpu":
       from habana_frameworks.torch.hpex.optimizers import FusedAdamW
       optimizer = FusedAdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
   ```
2. **Disable CUDA `pin_memory` on HPU:**
   In `DataLoader(...)`, set `pin_memory=(device_type == "cuda")` so PyTorch does not invoke the CUDA pinned-memory allocator on HPU nodes.
3. **Keep Batch Size & Learning Rate Consistent with CUDA Baselines:**
   The `run-*-best.sh` scripts are tuned for `batch_size=64`, `lr=2e-4`, and `eval_every_iter=1000`. If you increase `train.batch_size=256` on Gaudi 2 (to utilize its 96 GB HBM2e memory), the model performs **$4\times$ fewer gradient updates per epoch**. Either:
   - Keep `train.batch_size=64` or `128` to match the CUDA optimization trajectory directly, **or**
   - Scale the learning rate (e.g., `train.lr=4e-4` with warmup) and adjust `train.eval_every_iter=250` so validation and `ReduceLROnPlateau` step at the same epoch frequency.

---

## 3. Quickstart: Running on ASU SOL Intel Gaudi 2 Nodes

### Step 1: Allocate a Gaudi 2 Node on SOL
When requesting an interactive session or submitting an `sbatch` job on `sol`, target the Gaudi partition/nodes rather than `--gres=gpu:a100:1`:

```bash
# Check available Gaudi nodes/partitions on SOL
sinfo -o "%P %G %N" | grep -i gaudi
```

### Step 2: Activate the Gaudi PyTorch Environment
`setup_env.sh` automatically detects `gaudi*` hostnames on `sol` and activates `gaudi-pytorch-vllm`:

```bash
cd Pytorch
source setup_env.sh
```

Verify that PyTorch detects the Intel Gaudi 2 HPU:
```bash
python -c "
import torch
import habana_frameworks.torch.core as htcore
print('PyTorch version:', torch.__version__)
print('HPU available:', torch.hpu.is_available())
print('HPU device count:', torch.hpu.device_count())
print('HPU device name:', torch.hpu.get_device_name(0))
"
```

### Step 3: Set Recommended Gaudi 2 Environment Variables
```bash
export HABANA_VISIBLE_DEVICES=0
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=0
```

### Step 4: Run Training
To run with identical hyperparameters to the CUDA baseline:
```bash
MY_PYTHON_CODE=src/train_v1_classic.py ./run-xray14-resnet50-best.sh train.batch_size=64 train.lr=2e-4
```
Or with `train_v2_lightning.py`:
```bash
./run-xray14-resnet50-best.sh train.batch_size=64 train.lr=2e-4
```

---

## 4. Summary Checklist for Students

Before comparing Intel Gaudi 2 AUROC curves against NVIDIA A100/V100 results, verify:
- [ ] `habana_frameworks.torch.core` is imported and `torch.hpu.is_available()` is `True`.
- [ ] `htcore.mark_step()` is called after `loss.backward()` and `optimizer.step()` when `PT_HPU_LAZY_MODE=1`.
- [ ] `MaskedBCEWithLogitsLoss` upcasts `logits` and `targets` to `float32` before computing BCE and `.sum() / denom`.
- [ ] `evaluate.py` computes `MultilabelAUROC` and `MultilabelAveragePrecision` on `device="cpu"`.
- [ ] `train.batch_size`, `train.lr`, and `train.eval_every_iter` match your CUDA baseline.
