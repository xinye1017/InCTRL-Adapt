# Google Drive Checkpoint Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic resume-from-latest-checkpoint behavior in `InCTRL_TextualAdapter_Ablation.ipynb` so each experiment label continues training from its newest Google Drive checkpoint instead of always restarting from the base pretrained weights.

**Architecture:** Keep the notebook structure intact and add a lightweight checkpoint-resume layer inside the existing training notebook. Extend the global config with Drive checkpoint paths and resume toggles, add helper functions that discover the newest checkpoint for the current experiment label, restore model/optimizer/scheduler/history state when present, and save complete training state back to both local and Drive checkpoint directories after each epoch and at the end.

**Tech Stack:** Python, PyTorch, Jupyter Notebook, Google Drive mounted in Colab, `pathlib`, `shutil`, existing InCTRL/open_clip training code.

---

## File Structure

- Modify: `InCTRL_TextualAdapter_Ablation.ipynb`
  - Cell 3: add resume-related config values and Drive checkpoint paths
  - Cell 4: ensure local/Drive checkpoint directories exist after sync
  - Cell 6: add checkpoint helper functions, resume logic, per-epoch saving, end-of-training final save, and completed-run short-circuiting
  - Cell 8: keep the experiment call unchanged unless adding optional logging output is necessary
- Verify manually in Colab by running notebook cells in order

### Task 1: Add global checkpoint-resume configuration

**Files:**
- Modify: `InCTRL_TextualAdapter_Ablation.ipynb` (Cell 3 — 全局路径与实验配置)
- Test: `InCTRL_TextualAdapter_Ablation.ipynb` (run Cell 3 in Colab)

- [ ] **Step 1: Update Cell 3 to define local and Drive checkpoint directories plus resume switches**

Replace the checkpoint-related section in Cell 3 with this exact block so the notebook has explicit resume configuration:

```python
DRIVE_ROOT = Path('/content/drive/MyDrive')
WORK_ROOT = Path('/content')
PROJECT_NAME = 'InCTRL'
FORCE_CODE_SYNC = True    # 重新同步代码（InCTRL 目录）
FORCE_DATA_EXTRACT = False # 重新解压数据集（耗时较长，按需开启）
RESUME_TRAINING = True
SAVE_EVERY_EPOCH = True

PROJECT_SRC = DRIVE_ROOT / PROJECT_NAME
PROJECT_DST = WORK_ROOT / PROJECT_NAME

DATA_TAR_SRC = DRIVE_ROOT / 'InCTRL_data_industrial5.tar'
DATASETS_ROOT = WORK_ROOT / 'datasets'
DATA_TAR_LOCAL = WORK_ROOT / DATA_TAR_SRC.name

FEW_SHOT_ARCHIVE_DIR = DRIVE_ROOT / 'few-shot samples'
FEW_SHOT_ROOT = WORK_ROOT / 'fs_samples'
FEW_SHOT_ARCHIVES = ['visa.zip', 'AITEX.zip', 'elpv.zip']
FEW_SHOT_LOCAL_ARCHIVES = [WORK_ROOT / archive_name for archive_name in FEW_SHOT_ARCHIVES]

CKPT_NAME = 'vit_b_16_plus_240-laion400m_e32-699c4b84.pt'
DRIVE_CKPT = PROJECT_SRC / CKPT_NAME
LOCAL_CKPT = PROJECT_DST / CKPT_NAME

LOCAL_RESULTS_DIR = PROJECT_DST / 'results'
LOCAL_CHECKPOINT_DIR = LOCAL_RESULTS_DIR / 'checkpoints'
DRIVE_RESULTS_DIR = PROJECT_SRC / 'results'
DRIVE_CHECKPOINT_DIR = DRIVE_RESULTS_DIR / 'checkpoints'

TRAIN_DATASET_NAME = 'mvtec'
TEST_DATASETS = ['aitex', 'elpv', 'visa']
SHOT_LIST = [2, 4, 8]
SHOT = SHOT_LIST[0]

print('正在挂载 Google Drive ...')
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except:
    print("不在 Colab 环境，跳过 Drive 挂载。")

REQUIRED_SOURCES = [
    PROJECT_SRC,
    DATA_TAR_SRC,
    *[FEW_SHOT_ARCHIVE_DIR / name for name in FEW_SHOT_ARCHIVES],
]

# 兼容旧变量名
FORCE_RESYNC = FORCE_CODE_SYNC
```

- [ ] **Step 2: Run Cell 3 to verify the new config values are defined**

Run Cell 3.

Expected output includes:

```python
print(DRIVE_CHECKPOINT_DIR)
print(LOCAL_CHECKPOINT_DIR)
print(RESUME_TRAINING, SAVE_EVERY_EPOCH)
```

Expected result: paths print successfully and booleans show `True True`.

- [ ] **Step 3: Commit the config-only change**

```bash
git add InCTRL_TextualAdapter_Ablation.ipynb
git commit -m "feat: add checkpoint resume configuration"
```

### Task 2: Ensure checkpoint directories exist after sync

**Files:**
- Modify: `InCTRL_TextualAdapter_Ablation.ipynb` (Cell 4 — 同步代码、数据与 few-shot 资源)
- Test: `InCTRL_TextualAdapter_Ablation.ipynb` (run Cell 4 in Colab)

- [ ] **Step 1: Add local and Drive checkpoint directory creation inside `sync_data()`**

Inside `sync_data()`, immediately after the existing directory creation block:

```python
WORK_ROOT.mkdir(parents=True, exist_ok=True)
DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
FEW_SHOT_ROOT.mkdir(parents=True, exist_ok=True)
```

insert this exact code:

```python
LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Add a post-sync sanity print so the notebook shows checkpoint paths are ready**

After the final `print('资源同步完成')` line inside `sync_data()`, add:

```python
print(f'本地 checkpoints: {LOCAL_CHECKPOINT_DIR}')
print(f'Drive checkpoints: {DRIVE_CHECKPOINT_DIR}')
```

- [ ] **Step 3: Run Cell 4 to verify directories are created without changing existing dataset sync behavior**

Run Cell 4.

Expected output should still include the existing sync progress plus:

```text
本地 checkpoints: /content/InCTRL/results/checkpoints
Drive checkpoints: /content/drive/MyDrive/InCTRL/results/checkpoints
```

- [ ] **Step 4: Commit the directory-setup change**

```bash
git add InCTRL_TextualAdapter_Ablation.ipynb
git commit -m "feat: prepare local and drive checkpoint directories"
```

### Task 3: Add checkpoint discovery, load, and save helpers

**Files:**
- Modify: `InCTRL_TextualAdapter_Ablation.ipynb` (Cell 6 — 核心辅助函数定义)
- Test: `InCTRL_TextualAdapter_Ablation.ipynb` (run Cell 6 in Colab)

- [ ] **Step 1: Add helper functions directly above `run_experiment(...)`**

Insert these exact helper functions after `find_fs_pt(...)` and before `run_experiment(...)`:

```python
def checkpoint_prefix(label):
    return f'{label}_multishot'


def checkpoint_epoch_path(base_dir, label, epoch):
    return base_dir / f'{checkpoint_prefix(label)}_epoch{epoch:03d}.pth'


def checkpoint_final_path(base_dir, label):
    return base_dir / f'{checkpoint_prefix(label)}_latest.pth'


def list_experiment_checkpoints(base_dir, label):
    if not base_dir.exists():
        return []
    pattern = f'{checkpoint_prefix(label)}*.pth'
    return sorted(
        base_dir.glob(pattern),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )


def find_latest_drive_checkpoint(label):
    candidates = list_experiment_checkpoints(DRIVE_CHECKPOINT_DIR, label)
    return candidates[0] if candidates else None


def copy_checkpoint_to_targets(src_path, label, epoch=None):
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if epoch is not None:
        local_epoch_path = checkpoint_epoch_path(LOCAL_CHECKPOINT_DIR, label, epoch)
        drive_epoch_path = checkpoint_epoch_path(DRIVE_CHECKPOINT_DIR, label, epoch)
        shutil.copy2(src_path, local_epoch_path)
        shutil.copy2(src_path, drive_epoch_path)

    local_latest_path = checkpoint_final_path(LOCAL_CHECKPOINT_DIR, label)
    drive_latest_path = checkpoint_final_path(DRIVE_CHECKPOINT_DIR, label)
    shutil.copy2(src_path, local_latest_path)
    shutil.copy2(src_path, drive_latest_path)

    return local_latest_path, drive_latest_path


def build_checkpoint_payload(label, use_adapter, model, optimizer, scheduler, history_loss, epoch, lr, batch_size, steps_per_epoch):
    return {
        'label': label,
        'use_adapter': use_adapter,
        'shot_list': list(SHOT_LIST),
        'history_loss': [float(x) for x in history_loss],
        'epoch': int(epoch),
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'lr': float(lr),
        'batch_size': int(batch_size),
        'steps_per_epoch': int(steps_per_epoch),
    }


def try_resume_training(label, model, optimizer, scheduler, device):
    if not RESUME_TRAINING:
        print('RESUME_TRAINING=False，跳过 checkpoint 恢复')
        return 0, []

    latest_ckpt = find_latest_drive_checkpoint(label)
    if latest_ckpt is None:
        print(f'未找到 {label} 的 Drive checkpoint，从头开始训练')
        return 0, []

    print(f'发现最新 Drive checkpoint: {latest_ckpt}')

    try:
        checkpoint = torch.load(latest_ckpt, map_location=device)
        if checkpoint.get('label') != label:
            print(f'checkpoint label 不匹配: {checkpoint.get("label")} != {label}，从头开始训练')
            return 0, []

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

        history_loss = [float(x) for x in checkpoint.get('history_loss', [])]
        start_epoch = int(checkpoint.get('epoch', -1)) + 1

        print(f'✅ 已恢复训练状态：start_epoch={start_epoch}, 历史 loss 条数={len(history_loss)}')
        return start_epoch, history_loss
    except Exception as e:
        print(f'⚠️ checkpoint 恢复失败，将从头开始训练: {e}')
        return 0, []


def save_training_checkpoint(label, use_adapter, model, optimizer, scheduler, history_loss, epoch, lr, batch_size, steps_per_epoch, save_epoch_copy=True):
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_checkpoint_path = LOCAL_CHECKPOINT_DIR / f'{checkpoint_prefix(label)}_tmp.pth'

    payload = build_checkpoint_payload(
        label=label,
        use_adapter=use_adapter,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history_loss=history_loss,
        epoch=epoch,
        lr=lr,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )
    torch.save(payload, tmp_checkpoint_path)

    local_latest_path, drive_latest_path = copy_checkpoint_to_targets(
        tmp_checkpoint_path,
        label,
        epoch=epoch if save_epoch_copy else None,
    )

    if tmp_checkpoint_path.exists():
        tmp_checkpoint_path.unlink()

    print(f'✅ checkpoint 已保存 | local: {local_latest_path} | drive: {drive_latest_path}')
    return local_latest_path, drive_latest_path
```

- [ ] **Step 2: Run Cell 6 once to verify helper functions define successfully**

Run Cell 6.

Expected result: no syntax errors, existing imports still work, and the final helper definitions are available.

Optional verification commands in a temporary follow-up cell:

```python
print(checkpoint_prefix('B_with_adapter'))
print(checkpoint_final_path(DRIVE_CHECKPOINT_DIR, 'B_with_adapter'))
print(find_latest_drive_checkpoint('B_with_adapter'))
```

Expected result: the first two lines print valid names/paths; the third prints either a checkpoint path or `None`.

- [ ] **Step 3: Commit the helper-function change**

```bash
git add InCTRL_TextualAdapter_Ablation.ipynb
git commit -m "feat: add drive checkpoint helper functions"
```

### Task 4: Integrate resume logic into the training loop

**Files:**
- Modify: `InCTRL_TextualAdapter_Ablation.ipynb` (Cell 6 — `run_experiment(...)`)
- Test: `InCTRL_TextualAdapter_Ablation.ipynb` (run Cell 6 and Cell 8 in Colab)

- [ ] **Step 1: Replace the checkpoint setup block in `run_experiment(...)` with resume-aware state initialization**

Inside `run_experiment(...)`, keep the existing model/cfg/train_loader/tokenizer/optimizer/scheduler/loss function setup, but replace this block:

```python
history_loss = []
print(f'开始训练 {n_epochs} Epochs...')

# 外层进度条：Epoch 进度
epoch_pbar = tqdm(total=n_epochs, desc=f'[TRAIN] Epoch 0/{n_epochs}', unit='epoch')

for epoch in range(n_epochs):
```

with this exact block:

```python
start_epoch, history_loss = try_resume_training(
    label=label,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
)

if start_epoch >= n_epochs:
    print(f'✅ {label} 已训练完成（start_epoch={start_epoch}, n_epochs={n_epochs}），跳过训练阶段')
else:
    print(f'开始训练 {n_epochs} Epochs... 当前从 epoch {start_epoch + 1} 开始')

# 外层进度条：Epoch 进度
epoch_pbar = tqdm(total=n_epochs, desc=f'[TRAIN] Epoch {start_epoch}/{n_epochs}', unit='epoch')
if start_epoch > 0:
    epoch_pbar.update(start_epoch)
    if history_loss:
        epoch_pbar.set_postfix({'avg_loss': f'{history_loss[-1]:.4f}'})

for epoch in range(start_epoch, n_epochs):
```

- [ ] **Step 2: Save a checkpoint after every completed epoch**

Inside the epoch loop, immediately after these existing lines:

```python
scheduler.step()
avg_loss = float(epoch_loss / steps_per_epoch)
history_loss.append(avg_loss)
```

insert this exact block:

```python
if SAVE_EVERY_EPOCH:
    save_training_checkpoint(
        label=label,
        use_adapter=use_adapter,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history_loss=history_loss,
        epoch=epoch,
        lr=lr,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        save_epoch_copy=True,
    )
```

- [ ] **Step 3: Replace the old final model-save block with a full-state final checkpoint save**

Delete this existing block near the end of `run_experiment(...)`:

```python
checkpoints_dir = PROJECT_DST / 'results' / 'checkpoints'
checkpoints_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoints_dir / f'{label}_multishot_model.pth'
torch.save({
    'label': label,
    'use_adapter': use_adapter,
    'shot_list': SHOT_LIST,
    'history_loss': [float(x) for x in history_loss],
    'state_dict': model.state_dict(),
}, checkpoint_path)
run_experiment.last_checkpoint_path = checkpoint_path
print(f'✅ 模型已保存: {checkpoint_path}')
```

and replace it with this exact block:

```python
final_epoch = max(len(history_loss) - 1, start_epoch - 1)
local_ckpt_path, drive_ckpt_path = save_training_checkpoint(
    label=label,
    use_adapter=use_adapter,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    history_loss=history_loss,
    epoch=final_epoch,
    lr=lr,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    save_epoch_copy=False,
)
run_experiment.last_checkpoint_path = local_ckpt_path
run_experiment.last_drive_checkpoint_path = drive_ckpt_path
print(f'✅ 最终 checkpoint 已保存: {local_ckpt_path}')
print(f'✅ Drive checkpoint 已同步: {drive_ckpt_path}')
```

- [ ] **Step 4: Run the training cell once from scratch and verify the first save happens in both local and Drive directories**

Run Cell 6, then Cell 8.

Expected training output includes lines like:

```text
未找到 B_with_adapter 的 Drive checkpoint，从头开始训练
✅ checkpoint 已保存 | local: /content/InCTRL/results/checkpoints/B_with_adapter_multishot_latest.pth | drive: /content/drive/MyDrive/InCTRL/results/checkpoints/B_with_adapter_multishot_latest.pth
```

Expected verification in a new temporary cell:

```python
print(MODEL_PATH_B)
print(getattr(run_experiment, 'last_drive_checkpoint_path', None))
print((DRIVE_CHECKPOINT_DIR / 'B_with_adapter_multishot_latest.pth').exists())
```

Expected result: local path prints, Drive path prints, and the existence check returns `True`.

- [ ] **Step 5: Re-run Cell 8 and verify it resumes instead of restarting**

Without deleting the Drive checkpoint, run Cell 8 again.

Expected output includes one of these resume signals:

```text
发现最新 Drive checkpoint: /content/drive/MyDrive/InCTRL/results/checkpoints/B_with_adapter_multishot_latest.pth
✅ 已恢复训练状态：start_epoch=..., 历史 loss 条数=...
```

If the previous run already completed all epochs, expected output includes:

```text
✅ B_with_adapter 已训练完成（start_epoch=10, n_epochs=10），跳过训练阶段
```

- [ ] **Step 6: Commit the resume-enabled training loop**

```bash
git add InCTRL_TextualAdapter_Ablation.ipynb
git commit -m "feat: resume notebook training from drive checkpoints"
```

### Task 5: Preserve evaluation and reporting behavior after resume

**Files:**
- Modify: `InCTRL_TextualAdapter_Ablation.ipynb` (Cell 8 and Cell 9 only if needed for checkpoint path reporting)
- Test: `InCTRL_TextualAdapter_Ablation.ipynb` (run Cells 8-9 in Colab)

- [ ] **Step 1: Keep the existing experiment invocation unchanged unless you need to expose the Drive checkpoint path in output**

The cell should remain:

```python
loss_b, results_b = run_experiment('B_with_adapter', use_adapter=True, **SHARED_KWARGS)
MODEL_PATH_B = getattr(run_experiment, 'last_checkpoint_path', None)
print(f'已保存模型: {MODEL_PATH_B}')
```

If you want Drive visibility too, add exactly one extra line:

```python
print(f'Drive checkpoint: {getattr(run_experiment, "last_drive_checkpoint_path", None)}')
```

- [ ] **Step 2: Run Cells 8 and 9 together to verify resumed training still produces full evaluation outputs**

Run Cell 8, then Cell 9.

Expected result:
- `loss_b` remains a valid list of epoch losses
- `results_b` retains per-shot/per-dataset AUROC and AUPR values
- JSON/CSV export still succeeds
- plot generation still succeeds

Quick validation snippet in a temporary cell:

```python
print(type(loss_b), len(loss_b))
print(results_b.keys())
print((RESULTS_DIR / 'generalization_summary_multishot.json').exists())
print((RESULTS_DIR / 'generalization_summary_multishot.csv').exists())
```

Expected result: a list with length `<= N_EPOCHS`, dict keys containing `2, 4, 8`, and both file checks return `True`.

- [ ] **Step 3: Commit the verification-friendly output tweak if you added one**

```bash
git add InCTRL_TextualAdapter_Ablation.ipynb
git commit -m "chore: expose drive checkpoint path in notebook output"
```

If you did not change Cell 8 or Cell 9, skip this commit.

## Self-Review

- **Spec coverage:** The plan covers the selected approach A completely: Drive checkpoint directory configuration, same-label latest-checkpoint lookup, full-state restore (`model`, `optimizer`, `scheduler`, `history_loss`, `epoch`), fallback to fresh training, per-epoch and final synchronization back to Drive, and no disruption to downstream evaluation/reporting.
- **Placeholder scan:** No `TODO`, `TBD`, or vague “handle appropriately” language remains. All code-edit steps contain concrete code blocks and all validation steps contain concrete commands/snippets and expected outcomes.
- **Type consistency:** The helper names (`try_resume_training`, `save_training_checkpoint`, `find_latest_drive_checkpoint`, `last_drive_checkpoint_path`) are used consistently across tasks. The checkpoint payload fields are referenced consistently by both save and load paths.

Plan complete and saved to `docs/superpowers/plans/2026-04-07-google-drive-checkpoint-resume.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
