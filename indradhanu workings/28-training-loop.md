# 28 · The training loop

> **The one thing this part teaches:** what actually happens when you run the
> training script — and one configuration field that looks important and does
> nothing at all.

**Sources:**
[`app/abstract_classes/foundation_trainer.py`](../app/abstract_classes/foundation_trainer.py)
(the generic loop, 628 lines) and
[`app/foundation_models/trainers/hyperspectral_segformer_mae_trainer.py`](../app/foundation_models/trainers/hyperspectral_segformer_mae_trainer.py)
(Indradhanu's overrides, 357 lines).

---

## Starting a run

```bash
python scripts/train_foundation_model.py configs/hyperspectral_segformer_exp_2.json
```

The script is 35 lines. All of it:

```python
config  = TrainingConfig(**raw)      # validate the JSON
trainer = get_trainer(config)        # look up the right trainer class
trainer.train()                      # go
```

`TrainingConfig` is a pydantic model, so a typo in the JSON produces a clear
validation error immediately, rather than an `AttributeError` forty minutes into
the run.

`get_trainer` is the factory
([`trainer_factory.py`](../app/foundation_models/trainers/trainer_factory.py)):

```python
FoundationModelName.HYPERSPECTRAL_SEGFORMER_MAE: HyperspectralSegFormerMAETrainer,
```

One dictionary entry per model. (Remember part 05: there are three other
registries a new model must also appear in.)

---

## The class structure

`FoundationTrainer` is an abstract base class that owns everything generic:

- device selection (GPU if available)
- the optimiser and the learning-rate scheduler
- data loaders and hot-storage syncing
- the epoch loop
- checkpoint saving and pruning
- Weights and Biases logging

A subclass must implement exactly **three** methods:

```python
build_model()                  -> nn.Module
compute_loss(batch, model)     -> (loss_tensor, num_valid_samples)
validation_step(batch, model)  -> (loss_float,  num_valid_samples)
```

Indradhanu's trainer implements those three and **overrides** two more:

- `_run_epoch` — to track the epoch number (needed for the SAM ramp) and to log
  the loss components separately;
- `_run_train_pass` — to add gradient accumulation.

> **Why this design matters to you.** If you ever add a model, you write three
> small methods and get the entire training infrastructure for free. And when you
> are reading the code, you know exactly which three places contain
> model-specific logic.

---

## What one epoch does

```python
def _run_epoch(self, epoch, train_loaders, test_loaders):
    self.model.train()
    for size in patch_sizes:
        size_loss, size_samples = self._run_train_pass(train_loaders[size], cap)
    avg_train_loss = epoch_train_loss / max(epoch_train_samples, 1)

    self.model.eval()
    for size in patch_sizes:
        val_losses[size] = self._run_val_pass(test_loaders[size], cap)
    avg_val_loss = sum(val_losses.values()) / len(val_losses)

    self.scheduler.step()
    ...log...
    if (epoch + 1) % ckpt.save_every_n_epochs == 0:
        self._save_checkpoint(epoch + 1, avg_train_loss, val_losses)
        self._cleanup_checkpoints()
```

Five phases: **train, validate, step the scheduler, log, maybe checkpoint.**

### `model.train()` and `model.eval()`

These do not train or evaluate anything. They flip a flag that changes how
certain layers behave:

| Layer | `train()` | `eval()` |
|---|---|---|
| Dropout | randomly zeroes values | does nothing |
| BatchNorm | uses this batch's statistics, and updates its running averages | uses the stored running averages |

Both matter here: Indradhanu has `drop_rate = 0.4` and a BatchNorm in the
compressor. Forgetting `model.eval()` before validation is a classic bug — your
validation numbers become noisy and slightly optimistic, and nothing crashes to
tell you.

### `max(epoch_train_samples, 1)`

A small defensive detail. If every patch in an epoch were filtered out, this
would be a division by zero. The `max(..., 1)` makes it a division by 1
instead, giving 0.0 rather than a crash.

---

## Sample-weighted averaging

```python
total_loss += loss.item() * num_kept
valid_samples += num_kept
...
return total_loss / max(valid_samples, 1)
```

`compute_loss` returns a **per-sample average**, so multiplying by `num_kept`
recovers that batch's total. Then dividing the grand total by the grand count
gives a correctly weighted mean.

Without the weighting, a batch where only 3 patches survived the validity filter
would count for as much as one where all 128 did. The code even carries a
comment explaining it, because it is the sort of thing a reader would otherwise
"simplify" away.

---

## Gradient accumulation

### The problem

You want a large batch — large batches give smoother gradient estimates and
train more stably. But a 165-band model on 128x128 patches uses a lot of GPU
memory, and at some point `batch_size = 128` simply will not fit.

### The trick

Process several small batches, but only update the weights once, after all of
them.

```python
scaled_loss = loss / accum_steps
scaled_loss.backward()
...
if accum_count % accum_steps == 0:
    self.optimizer.step()
    self.optimizer.zero_grad()
```

### Why it works

`backward()` **adds** to each parameter's `.grad` field; it does not replace it.
PyTorch only clears gradients when you call `zero_grad()`.

So calling `backward()` four times without stepping leaves the *sum* of four
batches' gradients sitting in `.grad`. Stepping then applies all four at once.

### Why the division

Summing four batches' gradients gives four times the magnitude of one batch's.
Since the optimiser moves in proportion to the gradient, your effective learning
rate would silently quadruple.

Dividing each loss by `accum_steps` makes the accumulated gradient an **average**
rather than a sum — exactly what a single batch of size `128 x 4` would have
produced. Your configured learning rate keeps its meaning.

### The flush

```python
if accum_count % accum_steps != 0:
    self.optimizer.step()
    self.optimizer.zero_grad()
```

If the loader ran out mid-group, the leftover gradients are applied rather than
discarded.

### Is it on?

The current config sets `gradient_accumulation_steps: 1`, so the override
short-circuits straight back to the base implementation:

```python
if accum_steps <= 1:
    return super()._run_train_pass(loader, sample_cap)
```

Reach for accumulation when you cannot fit the batch size you want on your GPU.

---

## Optimiser and learning rate

```python
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
```

Plain **Adam**. No weight decay, no AdamW, no gradient clipping. Adam adapts a
per-parameter step size from recent gradient history, which makes it forgiving
about learning-rate choice — a reasonable default when you are not running
extensive hyperparameter searches.

### The schedule

v0.2.0 uses cosine annealing: start at `1e-3`, decay smoothly to `1e-6` over 200
epochs.

```
lr(t) = min_lr + 0.5 * (lr_0 - min_lr) * (1 + cos(pi * t / T_max))
```

The shape: high and flat at the start, steepest in the middle, gently flattening
to almost nothing at the end.

**Worked**, with `lr_0 = 1e-3`, `min_lr = 1e-6`, `T_max = 200`:

| epoch `t` | `pi * t / 200` | `cos(...)` | `lr` |
|---|---|---|---|
| 0 | 0 | 1.000 | 1.00e-3 |
| 50 | pi/4 | 0.707 | 8.54e-4 |
| 100 | pi/2 | 0.000 | 5.00e-4 |
| 150 | 3pi/4 | -0.707 | 1.47e-4 |
| 200 | pi | -1.000 | 1.00e-6 |

Check epoch 100 by hand:

```
lr = 0.000001 + 0.5 * (0.001 - 0.000001) * (1 + 0)
   = 0.000001 + 0.5 * 0.000999
   = 0.000001 + 0.0004995
   = 0.0005005      ~ 5.00e-4
```

Why decay at all? Large steps early explore the space quickly; small steps late
settle into a good minimum instead of bouncing around it.

---

## The gotcha: `warmup_epochs` does nothing

`LRScheduleConfig` has this field:

```python
warmup_epochs: int = Field(
    default=5,
    ge=0,
    description="Linear warmup from min_lr to learning_rate",
)
```

Both shipped configurations set it to 5. It is documented. It has a sensible
default. It appears in the JSON.

**And it is never read.**

Check for yourself:

```bash
grep -rn "warmup" --include="*.py" app/ scripts/
```

The only hits are the field definition itself and a docstring example.
`_build_scheduler` handles `cosine`, `step` and `plateau`, and none of the three
consults it.

### Why this matters concretely

Warmup exists to avoid enormous, destabilising updates in the first few steps,
while Adam's internal statistics are still being estimated. Without it, the
v0.2.0 run began at the full `1e-3` on step one.

And look at what the research notes say happened:

> *"v0.2.0 epoch 2 hits 0.035 (lr=1e-3 was too aggressive; later epochs regress
> to ~0.04)."*

An unusually good epoch 2, then a regression that never fully recovers. That is
consistent with early instability — precisely the thing warmup would have
prevented.

> **The lesson, which the repo states as policy:** *verify config fields against
> source before assuming they do anything.* A field can be documented, defaulted,
> validated and set in every config file, and still be dead code.

---

## Checkpoints

Each saved checkpoint is a dictionary containing:

- `model_state_dict` — every parameter and buffer,
- `optimizer_state_dict` — Adam's internal statistics, needed to resume cleanly,
- `epoch`, `train_loss`, per-size `val_losses`,
- the entire training config as JSON, for reproducibility.

Naming: `{model_name}_v{version}_epoch{epoch}.pt`, which is why the file is

```
hyperspectral_segformer_mae_v0.2.0_epoch200.pt
```

`keep_top_k` (20 in the current config) prunes all but the best by validation
loss after each save. Without it, saving every epoch of a 200-epoch run at 22 MB
each would be over 4 GB.

### Resuming: two modes

| `resume_mode` | What is restored | When to use it |
|---|---|---|
| `"resume"` | weights, optimiser state, epoch counter, scheduler position | a run crashed or was interrupted |
| `"finetune"` | **weights only** — fresh optimiser, scheduler and epoch | adapting a trained model to new data |

The distinction matters. Resuming a 200-epoch cosine schedule at epoch 150 with
a *fresh* scheduler would restart the learning rate at `1e-3`, undoing much of
the careful decay. `"resume"` restores the scheduler's position too.

---

## What a training log line looks like

```
Epoch 101/200 | train_loss: 0.041203 | val_loss: [128px: 0.043490] |
avg_val: 0.043490 | lr: 4.94e-04
  L1: 0.009812 | SAM: 0.062776 rad (3.60 deg) | lambda: 0.500
```

Reading it:

- `train_loss` and `val_loss` are the **combined** L1 + lambda x SAM.
- `lr` is 4.94e-4 — consistent with epoch 101 on the cosine schedule above.
- The second line gives the components separately.

**Watch the second line**, for the reasons in part 25.

---

## Common confusions

**"Does `train()` do the training?"**
No. `model.train()` only sets a mode flag. `trainer.train()` (a different
method, on a different object) runs the loop. Unfortunate naming collision, and
a real source of confusion.

**"Why is the validation loss averaged across patch sizes?"**
Because the thermal models train on several sizes. Indradhanu has one size, so
the average is over a single value.

**"Can I change the learning rate mid-run?"**
Edit the config and resume with `"finetune"` — but you will lose the optimiser
state and the scheduler position. Consider whether that is what you want.

**"Should I add warmup, since it is missing?"**
It would probably help. But it is a real behavioural change to a shared base
class that seven models use, so it belongs in a proper design discussion — and
this repo's workflow (the `iterative-nano-chunking` skill) requires agreeing the
design before writing code.

---

## Check yourself

1. Name the three methods a trainer subclass must implement.
2. What do `model.train()` and `model.eval()` actually change?
3. Explain gradient accumulation, and why the loss is divided by `accum_steps`.
4. Compute the cosine learning rate at epoch 200 for `lr_0 = 1e-3`,
   `min_lr = 1e-6`, `T_max = 200`.
5. What does `warmup_epochs` do, and how would you have found that out?

<details>
<summary>Answers</summary>

1. `build_model`, `compute_loss` and `validation_step`.
2. They flip a mode flag. Dropout is active in train mode and inert in eval;
   BatchNorm uses batch statistics and updates its running averages in train
   mode, and uses the stored averages in eval.
3. Call `backward()` on several small batches without stepping, so gradients
   accumulate, then step once. Divide each loss by `accum_steps` so the
   accumulated gradient is an average rather than a sum — otherwise the effective
   learning rate scales with the number of accumulation steps.
4. `cos(pi) = -1`, so `lr = 1e-6 + 0.5 * (1e-3 - 1e-6) * (1 - 1) = 1e-6`.
5. Nothing — it is never read anywhere in the codebase.
   `grep -rn "warmup" --include="*.py" app/ scripts/` returns only the field
   definition and a docstring.

</details>

---

**Next:** running the trained model on a real scene, in
[29-inference.md](29-inference.md)
