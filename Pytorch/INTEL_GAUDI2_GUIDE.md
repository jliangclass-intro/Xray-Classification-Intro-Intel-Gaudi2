# Running PyTorch on Intel Gaudi 2 (HPU): Troubleshooting & Best Practices

This guide explains how to run chest X-ray classification training on **Intel Gaudi 2** processors (`gaudi*` nodes on the ASU SOL cluster) and details why unmodified CUDA-oriented PyTorch code can produce degraded training convergence or lower AUROC on Gaudi 2—along with the exact code adjustments (already integrated into `src/`) needed to match CUDA accuracy while remaining 100% compatible with NVIDIA CUDA GPUs.

---

## 1. Why PyTorch Training on Intel Gaudi 2 Can Produce Worse Results Than CUDA

Unlike NVIDIA GPUs (which execute CUDA kernels eagerly in FP32/TF32 or FP16/BF16 with mature CUDA autocast rules), Intel Gaudi 2 (Habana HPU) uses a **graph compiler architecture (SynapseAI)** with two specialized compute engines: the **MME (Matrix Multiplication Engine)** and **TPC (Tensor Processor Core)**.

When students run standard PyTorch scripts written for CUDA on Gaudi 2 without HPU-specific adjustments, **six specific issues** in the training and evaluation pipeline degrade model convergence and validation metrics:

| # | Issue | Gaudi 2 Only or Also CUDA? | Behavior on NVIDIA CUDA | Affected File(s) |
|---|-------|----------------------------|-------------------------|------------------|
| **1** | **Missing `htcore.mark_step()` in Lazy Mode** | **Gaudi 2 Only** | Not needed on CUDA (CUDA executes kernels eagerly via asynchronous CUDA streams). `hpu_mark_step()` is a no-op on CUDA. | `src/train_v1_classic.py`, `src/train_v2_lightning.py`, `src/evaluate.py` |
| **2** | **`bfloat16` precision loss in `MaskedBCEWithLogitsLoss`** | **Primarily Gaudi 2** (also beneficial on CUDA BF16) | On CUDA, `torch.amp.autocast("cuda")` automatically upcasts `F.binary_cross_entropy_with_logits` to FP32, though `mask.float().sum()` and `loss.sum() / denom` benefit from explicit FP32 on both Gaudi 2 and Ampere+ GPUs. | `src/train_v1_classic.py`, `src/train_v2_lightning.py` |
| **3** | **Running `torchmetrics` (`MultilabelAUROC`) on `device="hpu"`** | **Gaudi 2 Only** | Works natively on both CUDA (`device="cuda"`) and CPU (`device="cpu"`). Our code keeps CUDA on GPU and routes `hpu` to `cpu`. | `src/evaluate.py` |
| **4** | **PyTorch Lightning `strategy="ddp"` without `lightning-habana`** | **Gaudi 2 Only** | Standard `accelerator="auto"` and `strategy="ddp"` (with `nccl`) work natively on CUDA without extra plugins. | `src/train_v2_lightning.py` |
| **5** | **Missing explicit `habana_frameworks.torch` import** | **Gaudi 2 Only** | Not needed on CUDA (`torch.cuda` is built into PyTorch). Wrapped in `try...except ImportError` so CUDA machines ignore it. | `src/utils.py` |
| **6** | **Large Batch Size (`256`) without LR / `eval_every_iter` scaling** | **Both Gaudi 2 & CUDA** | Applies to any accelerator with large VRAM (e.g., 96 GB Gaudi 2 or 80 GB A100): $4\times$ larger batch size means $4\times$ fewer gradient updates per epoch unless LR and `eval_every_iter` are scaled. | `run-*-best.sh` / CLI overrides |

---

## 2. Detailed Root Causes, Code Fixes, and CUDA vs. Gaudi 2 Comparison

### Fix 1: Explicitly Import the Habana PyTorch Bridge (`src/utils.py`)
- **Scope:** **Gaudi 2 Only** (No-op on NVIDIA CUDA).
- **Why on Gaudi 2:** In SynapseAI, `torch.hpu` is registered when `habana_frameworks.torch` is imported (or via `PT_HPU_AUTOLOAD=1` in newer drivers). Checking `getattr(torch, "hpu", None)` without importing `habana_frameworks.torch` first can return `None` and silently fall back to CPU.
- **How it works on CUDA:** `torch.cuda` is built directly into standard PyTorch wheels. Because the import is wrapped in `try ... except ImportError`, CUDA and CPU environments simply set `htcore = None` and proceed normally.

```python
# In src/utils.py
try:
    import habana_frameworks.torch.core as htcore  # Registers torch.hpu and HPU kernels
except ImportError:
    htcore = None
```

---

### Fix 2: Call `htcore.mark_step()` in Lazy Mode (`src/train_v1_classic.py`, `src/train_v2_lightning.py`, `src/evaluate.py`)
- **Scope:** **Gaudi 2 Only** (Not necessary on NVIDIA CUDA).
- **Why on Gaudi 2:** By default, Intel Gaudi 2 runs in **Lazy Mode (`PT_HPU_LAZY_MODE=1`)**. Operations are recorded into an execution graph and only compiled and executed on the HPU when `htcore.mark_step()` is called (or when a tensor is copied to CPU, such as `.item()`). Calling `loss.item()` without `mark_step()` flushes the forward loss tensor to CPU while `optimizer.step()` weight updates remain un-stepped or fused across iterations.
- **How it works on CUDA:** NVIDIA CUDA executes operations **eagerly** by queuing kernels onto asynchronous CUDA streams immediately as Python lines execute (or via `torch.compile` / `torch.cuda.CUDAGraph` when explicitly requested). Therefore, CUDA never needs `mark_step()`, and `if device_type == "hpu": hpu_mark_step()` is skipped on CUDA.

```python
# In src/train_v1_classic.py
loss.backward()
if device_type == "hpu":
    hpu_mark_step()

optimizer.step()
if device_type == "hpu":
    hpu_mark_step()

loss_meter.update(loss.item(), bs)
```

> **Alternative on Gaudi 2 (Eager + `torch.compile` Mode):**  
> On SynapseAI $\ge 1.16$ / PyTorch $\ge 2.2$, you can disable Lazy Mode (`export PT_HPU_LAZY_MODE=0`) and compile the model with `model = torch.compile(model, backend="hpu_backend")` (similar to `torch.compile(model, backend="inductor")` on CUDA). However, Lazy Mode (`PT_HPU_LAZY_MODE=1`) with explicit `hpu_mark_step()` remains the most predictable baseline on Gaudi 2.

---

### Fix 3: Force FP32 inside `MaskedBCEWithLogitsLoss` (`src/train_v1_classic.py` & `src/train_v2_lightning.py`)
- **Scope:** **Critical on Gaudi 2 (`bfloat16`); Safe & Recommended on CUDA (`bfloat16` / `float16`)**.
- **Why on Gaudi 2:** On Intel Gaudi 2, mixed precision uses **`bfloat16`**, which has only **7 mantissa bits** (compared to 10 bits in `float16` and 23 bits in `float32`). In `bfloat16`, integers larger than $256$ cannot be represented with unit precision ($257$ rounds to $256$). With `batch_size=128` or `256` and `14` labels, `denom` is $1792$ to $3584$. Computing `loss` and `loss.sum() / denom` in `bfloat16` truncates small per-class gradients and introduces rounding noise.
- **How it works on CUDA:** On NVIDIA GPUs, `torch.amp.autocast("cuda")` has a built-in whitelist/blacklist that automatically casts `F.binary_cross_entropy_with_logits` inputs to `float32` (and actually raises a `RuntimeError` if you try to run raw `binary_cross_entropy` in FP16). However, explicitly casting `logits = logits.float()` and `targets = targets.float()` at the start of `MaskedBCEWithLogitsLoss.forward()` ensures the mask multiplication (`loss * mask.float()`) and sum reduction (`loss.sum() / denom`) are also guaranteed to run in FP32 on **both** Gaudi 2 (`bfloat16`) and NVIDIA A100 (`bfloat16`) / V100 (`float16`), with zero speed penalty.

```python
def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Upcast to float32 so bfloat16 sum reduction over (batch_size * num_classes)
    # does not lose mantissa precision on Intel Gaudi 2 (HPU) or Ampere+ GPUs.
    logits = logits.float()
    targets = targets.float()
    mask = targets != -1
    targets_clamped = torch.where(mask, targets, torch.zeros_like(targets))
    pos_w = self.pos_weight.float() if self.pos_weight is not None else None
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets_clamped,
        pos_weight=pos_w,
        reduction="none",
    )
    loss = loss * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom
```

---

### Fix 4: Compute `torchmetrics` AUROC / AUPR on CPU When Running on HPU (`src/evaluate.py`)
- **Scope:** **Gaudi 2 Only** (On CUDA, `torchmetrics` works on either `"cuda"` or `"cpu"`).
- **Why on Gaudi 2:** In `src/evaluate.py`, `evaluate()` already moves model predictions to CPU after each batch (`probs.cpu().numpy()`). Passing `device="hpu"` into `calculate_multilabel_metrics()` copies those NumPy arrays **back to HPU** to run `MultilabelAUROC(ignore_index=-1)` and `MultilabelAveragePrecision(ignore_index=-1)`:
  1. `targets_tensor.long()` creates `int64` tensors on HPU, where TPC has limited 64-bit integer support.
  2. `ignore_index=-1` in `torchmetrics` performs dynamic boolean masking (`target != -1`), `torch.unique`, and `torch.sort`—causing dynamic shape recompilations, CPU fallbacks, or precision mismatches on HPU.
  3. Because `ReduceLROnPlateau` and `EarlyStopping` monitor `val_mean_auroc`, corrupted HPU metric values cause premature learning rate drops and wrong `best_model.pt` selection.
- **How it works on CUDA:** NVIDIA GPUs natively support 64-bit integer indexing (`int64`), dynamic boolean masking (`target != -1`), and Thrust radix sort on GPU. In `src/evaluate.py`, `_resolve_metric_device(device)` routes `"hpu"` to `"cpu"` while leaving `"cuda"` on GPU (note: running `torchmetrics` on `"cpu"` or `sklearn.metrics.roc_auc_score` on CPU also works identically on CUDA machines).

```python
# In src/evaluate.py:
def _resolve_metric_device(device: Union[str, torch.device]) -> Union[str, torch.device]:
    """Return CPU when device is HPU so torchmetrics int64/sorting/ignore_index ops do not degrade."""
    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    if dev_type == "hpu":
        return "cpu"
    return device
```

---

### Fix 5: PyTorch Lightning on Gaudi 2 (`src/train_v2_lightning.py`)
- **Scope:** **Gaudi 2 Only** (Standard `L.Trainer(accelerator="auto", strategy="ddp")` works natively on CUDA).
- **Why on Gaudi 2:** Standard PyTorch Lightning requires the **`lightning-habana`** plugin to support HPU:
  1. Without `lightning-habana` (or explicit `hpu_mark_step()` hooks in `on_before_optimizer_step` and `optimizer_step`), Lightning does **not** flush Lazy Mode graphs during training.
  2. `strategy="ddp"` attempts to use `nccl`/`gloo` instead of Intel's **`hccl`** (`HPUParallelStrategy` / `SingleHPUStrategy`).
- **How it works on CUDA:** PyTorch Lightning has built-in support for NVIDIA GPUs (`CUDAPlugin`, `DDPStrategy` with `nccl`, and `TorchSyncBatchNorm`). In `src/train_v2_lightning.py`, when `device_type == "cuda"`, it uses Lightning's native `accelerator="auto"` and `strategy="ddp"`. When `device_type == "hpu"`, it automatically loads `lightning_habana` (`HPUAccelerator`, `SingleHPUStrategy` / `HPUParallelStrategy`, `HPUPrecisionPlugin`) if installed, and also calls `hpu_mark_step()` inside `on_before_optimizer_step` and `optimizer_step`.

---

### Fix 6: Optimizer (`FusedAdamW`), DataLoader `pin_memory`, and Hyperparameter Scaling
- **Scope:**
  - **`FusedAdamW` (`habana_frameworks.torch.hpex.optimizers.FusedAdamW`):** **Gaudi 2 Only.** On NVIDIA CUDA, the equivalent fast optimizer is `torch.optim.AdamW(..., fused=True)`. Standard `torch.optim.AdamW` works on both.
  - **`pin_memory=(device_type == "cuda")`:** **Gaudi 2 Fix / CUDA Enabled.** Page-locked (pinned) host memory in `DataLoader(pin_memory=True)` uses the CUDA driver allocator to speed up host-to-GPU DMA transfers on NVIDIA GPUs. Setting `pin_memory=(device_type == "cuda")` keeps `pin_memory=True` on CUDA while disabling it on HPU/CPU so PyTorch does not invoke the CUDA allocator on Gaudi nodes.
  - **Batch Size & Learning Rate Scaling:** **Applies to BOTH Gaudi 2 and CUDA.** The `run-*-best.sh` scripts are tuned for `batch_size=64`, `lr=2e-4`, and `eval_every_iter=1000`. Whether you increase `train.batch_size=256` on an Intel Gaudi 2 (96 GB HBM2e) or an NVIDIA A100 (80 GB HBM2e), the model performs **$4\times$ fewer gradient updates per epoch**. On either platform, either:
    - Keep `train.batch_size=64` or `128` to match the baseline optimization trajectory directly, **or**
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
