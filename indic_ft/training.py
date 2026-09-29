"""Used in during training (stage 2)  -- the Trainer.

"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler

from .config import Config
from .metrics import transcribe_rows, wer_cer



class FrozenEncoderTrainer(Trainer):
    def __init__(self, *args, label_smoothing: float = 0.1, lengths=None,
                 group_by_length: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self._lengths = lengths
        # transformers 5 dropped TrainingArguments.group_by_length, so the flag
        # is ours to carry
        self._group_by_length = group_by_length

    def _get_train_sampler(self, train_dataset=None):
        """Trainer's own group_by_length path only reads lengths off a
        datasets.Dataset; ours is a plain torch Dataset, so hand them over."""
        if self._group_by_length and self._lengths is not None:
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=self._lengths,
            )
        return super()._get_train_sampler(train_dataset)

    def evaluate(self, *args, metric_key_prefix: str = "val", **kwargs):
        """Prefix the validation metrics val_* instead of transformers' eval_*.

        The training loop calls this with the default prefix, so every metric it
        produces -- val_loss, val_runtime -- is named the way this project names
        things, all the way into metric_for_best_model and trainer_state.json.
        """
        return super().evaluate(*args, metric_key_prefix=metric_key_prefix, **kwargs)

    def _determine_best_metric(self, metrics, trial) -> bool:
        """Pick the best checkpoint by metric_for_best_model, taken literally.

        transformers force-prepends 'eval_', which would turn 'val_loss' into the
        non-existent 'eval_val_loss'. This looks the name up as given (accepting a
        val_/eval_ prefixed fallback) so metric_for_best_model='val_loss' works.
        """
        import numpy as np
        from transformers.trainer_utils import SaveStrategy

        name = self.args.metric_for_best_model
        if name is None:
            return False
        for key in (name, f"val_{name}", f"eval_{name}"):
            if key in metrics:
                value = metrics[key]
                break
        else:
            raise KeyError(
                f"metric_for_best_model '{name}' not in validation metrics "
                f"{list(metrics)}")

        better = np.greater if self.args.greater_is_better else np.less
        if self.state.best_metric is None:
            self.state.best_metric = float("-inf") if self.args.greater_is_better else float("inf")
        if better(value, self.state.best_metric):
            self.state.best_metric = value
            if self.args.save_strategy in (SaveStrategy.STEPS, SaveStrategy.EPOCH,
                                           SaveStrategy.BEST):
                self.state.best_global_step = self.state.global_step
            return True
        return False

    #: Dropped before the console, TensorBoard or log_history see them.
    #: Together with 02_train.py not calling log_metrics/save_metrics, this
    #: keeps the end-of-run summary out of the run entirely: it reports the
    #: cost of the run, not the quality of the model. The *_per_second keys are
    #: throughput -- transformers computes them in speed_metrics() after the
    #: work is done and reads them back nowhere, so they measure GPU contention
    #: rather than anything about the model.
    DROP = {"epoch", "total_flos", "train_loss", "train_runtime",
            "train_samples_per_second", "train_steps_per_second",
            "val_samples_per_second", "val_steps_per_second"}

    #: transformers names every validation metric eval_*; this project says val_*
    @staticmethod
    def rename(key: str) -> str:
        return "val_" + key[len("eval_"):] if key.startswith("eval_") else key

    def log(self, logs: dict, start_time=None) -> None:
        """Report val_* metrics against optimiser steps, not fractional epochs.

        Deliberately does NOT call super().log(). Trainer.log re-inserts
        logs["epoch"] = state.epoch on every call, so stripping epoch before
        delegating has no effect -- it comes straight back. The body below is
        Trainer.log with that injection left out and the keys renamed; nothing
        reads state.log_history for control flow, it is only serialised into
        trainer_state.json.

        The incoming dict is never mutated: for an evaluation the very same
        object is handed to _determine_best_metric afterwards, which looks up
        metric_for_best_model ("val_loss") by its original name.
        """
        logs = {k: v for k, v in ((self.rename(k), v) for k, v in logs.items())
                if k not in self.DROP}
        if not logs:
            return          # nothing left worth a line (the end-of-run summary)
        logs = {"step": self.state.global_step, **logs}
        self.state.log_history.append(dict(logs))
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        base = model.module if hasattr(model, "module") else model

        # frozen encoder: no graph, and called outside the DDP wrapper so there
        # are no gradients to all-reduce
        with torch.no_grad():
            encoder_outputs = base.get_encoder()(
                inputs["input_features"], attention_mask=inputs["attention_mask"]
            )

        outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=inputs["decoder_input_ids"],
            use_cache=False,
        )
        # Label smoothing is a TRAINING regulariser, not a metric. Scoring with
        # it puts a floor under val_loss (0.1 smoothing over a 7152 vocab costs
        # 1.2124 nats even for a perfect model) and rewards under-confidence,
        # which is backwards when the number is also picking the best
        # checkpoint. Trainer calls model.eval() before every prediction step,
        # so model.training separates the two cleanly.
        smoothing = self.label_smoothing if model.training else 0.0
        loss_fct = CrossEntropyLoss(
            ignore_index=-100, label_smoothing=smoothing, reduction="sum"
        )
        loss = loss_fct(
            outputs.logits.reshape(-1, outputs.logits.size(-1)).float(), labels.reshape(-1)
        )
        # this forward takes **kwargs, so Trainer treats it as loss-kwarg aware and
        # skips the grad-accum division -- normalise by the global token count it
        # supplies, falling back to the local count when it does not
        denom = num_items_in_batch if num_items_in_batch is not None else (labels != -100).sum()
        loss = loss / torch.clamp(torch.as_tensor(denom, device=loss.device).float(), min=1.0)
        return (loss, outputs) if return_outputs else loss


class WerCallback(TrainerCallback):
    """Greedy WER/CER at each evaluation. Rank 0 only; decoding is sequential."""

    def __init__(self, rows, model, feature_extractor, tokenizer, history=None, trainer=None):
        self.rows = rows
        self.model = model
        self.fe = feature_extractor
        self.tok = tokenizer
        self.history = history if history is not None else []
        self.trainer = trainer          # set after construction; needed to emit metrics

    def on_evaluate(self, args, state, control, **kwargs):
        if args.process_index != 0 or not self.rows:
            return
        hyps = transcribe_rows(self.model, self.fe, self.tok, self.rows, args.device)
        scores = wer_cer([r["text"] for r in self.rows], hyps)
        scores["step"] = state.global_step
        self.history.append(scores)
        # tqdm.write, not print: a bare print lands in the middle of the live
        # progress bar and smears both. write() puts the line above it.
        from tqdm.auto import tqdm
        tqdm.write(f"[val @ step {state.global_step}] "
                   f"WER {scores['wer']:.4f}  CER {scores['cer']:.4f}  (n={scores['n']})")
        # route through Trainer.log so the scalars reach the logger; ScalarLogger
        # files anything named eval_*/val_* under val/ -- a bare print() reaches
        # no logger at all
        if self.trainer is not None:
            self.trainer.log({"val_wer": scores["wer"], "val_cer": scores["cer"]})


class ScalarLogger(TrainerCallback):
    """TensorBoard writer with our own tag names.

    The stock TensorBoardCallback routes every key through rewrite_logs(), which
    understands only the prefixes "eval_" and "test_" and prepends "train/" to
    everything else. That is why "eval/wer" came out as "train/eval/wer": the
    slash meant the prefix never matched. Owning the SummaryWriter lets every
    validation series be called val/* and keeps the axis-only keys out.
    """

    #: Never plotted. These two are the only keys that reach here and do not
    #: belong on a chart: "step" is the x-axis itself, so plotting it draws a
    #: diagonal line, and val_runtime is a timing. Everything else worth
    #: excluding is already gone -- FrozenEncoderTrainer.DROP strips it in
    #: log(), upstream of this callback.
    SKIP = {"step", "val_runtime"}

    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self.writer = None

    @staticmethod
    def tag(key: str) -> str:
        """eval_loss/val_wer -> val/loss, val/wer;  loss -> train/loss."""
        for prefix in ("eval_", "val_"):
            if key.startswith(prefix):
                return "val/" + key[len(prefix):]
        return "train/" + key

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.writer is None or not state.is_world_process_zero:
            return
        for k, v in (logs or {}).items():
            if k not in self.SKIP and isinstance(v, (int, float)) and not isinstance(v, bool):
                self.writer.add_scalar(self.tag(k), v, state.global_step)
        self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class TextLogger(TrainerCallback):
    """Plain-text twin of the TensorBoard event file.

    The event files are TFRecord-framed protobuf: compact, checksummed and
    tail-able by TensorBoard, but unreadable in an editor. This writes the same
    numbers as one line per log event, appending so successive runs accumulate
    behind a header rather than overwriting each other. Line-buffered, so
    `tail -f` works while training.
    """

    def __init__(self, path, header: dict | None = None):
        self.path = Path(path)
        self.header = header or {}
        self.fh = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", buffering=1, encoding="utf-8")
        self.fh.write(f"\n{'=' * 78}\n"
                      f"run started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        for k, v in self.header.items():
            self.fh.write(f"  {k:24} {v}\n")
        self.fh.write(f"{'=' * 78}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.fh is None or not state.is_world_process_zero:
            return
        # byte-for-byte the dict the terminal shows: transformers' own
        # ProgressCallback formats floats to 4 significant digits as strings and
        # prints str(dict), so the file and the console never disagree. The
        # step/max_steps and percentage carry what the progress bar conveys,
        # which cannot itself be written to a file -- it is carriage returns.
        shown = {k: (f"{v:.4g}" if isinstance(v, float) else v)
                 for k, v in (logs or {}).items()}
        if shown:
            pct = 100 * state.global_step / state.max_steps if state.max_steps else 0
            self.fh.write(f"{datetime.now():%H:%M:%S}  "
                          f"{state.global_step:>6}/{state.max_steps} {pct:3.0f}%  "
                          f"{shown}\n")

    def on_train_end(self, args, state, control, **kwargs):
        if self.fh is not None:
            self.fh.write(f"run finished {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            self.fh.close()
            self.fh = None


def build_training_args(cfg: Config) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(cfg.ckpt_dir),
        max_steps=cfg.max_steps,              # the only thing that sets run length
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        # transformers 5 removed warmup_ratio; max_steps is fixed, so the
        # equivalent step count is exact
        warmup_steps=round(cfg.warmup_ratio * cfg.max_steps),
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_batch,
        per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        dataloader_num_workers=cfg.num_workers,
        # step-based so the run is validated and checkpointed at max_steps itself;
        # "epoch" would leave the tail after the last whole epoch unvalidated
        # (eval_strategy/eval_steps are transformers' own argument names)
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.val_steps,
        # load_best_model_at_end requires save_steps % val_steps == 0; the
        # config is checked for that before the model is ever downloaded
        save_steps=cfg.save_checkpoint_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="val_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        label_names=["labels"],
        remove_unused_columns=False,      # the collator consumes raw manifest dicts
        ddp_find_unused_parameters=False, # frozen encoder params are simply absent
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        # "tensorboard" is handled by ScalarLogger, which owns the tag names;
        # leaving it here too would write every scalar a second time under the
        # stock train/eval/... names
        report_to=[r for r in cfg.report_to if r != "tensorboard"],
        seed=cfg.seed,
    )
