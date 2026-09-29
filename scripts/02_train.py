#!/usr/bin/env python3
"""Stage 3 -- training.

    torchrun --nproc_per_node=2 scripts/02_train.py
    python scripts/02_train.py                          # single GPU

Reads only the artifacts 01_prepare_data.py wrote.

Before the first weight update this runs every check: settings, CUDA, split integrity, target round-trip, one real
batch through the collator, the encoder freeze, and a forward/backward proving
gradients reach the decoder and nowhere else. They cost a few seconds against a
500-step run, so they are never skipped.

The final model lands in final_ckpt_for_inference: weights, config, and both
SentencePiece models. The architecture code is NOT copied in -- it lives once in
indic_ft/IndicCanary/ -- so load it through this package (import indic_ft, then
from_pretrained), not standalone with trust_remote_code.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from indic_ft import manifest as mf
from indic_ft.config import DEFAULT_CONFIG, Config
from indic_ft.data import ManifestDataset, SpeechCollator, load_feature_extractor
from indic_ft.hub import code_dir
from indic_ft.runlog import append
from indic_ft.modeling import (freeze_encoder, keep_code_out_of_checkpoints,
                               load_model, summarize)
from indic_ft.text import MultiLangEncoder, load_tokenizer
from indic_ft.training import (FrozenEncoderTrainer, ScalarLogger, TextLogger,
                               WerCallback, build_training_args)

# these must sit beside the weights or the fine-tuned checkpoint will not load
# The model code is NOT copied here -- it lives once in indic_ft/IndicCanary/.
# final/ holds weights + config + tokenizer/feature-extractor assets, and loads
# through the vendored package (import indic_ft, then from_pretrained). Only the
# non-code generation config is carried over, in case save_model omits it.
BUNDLE = ("generation_config.json",)

OK, BAD = "  ok  ", " FAIL "
#: torchrun sets these; plain `python` leaves them unset and both default to 0
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


def parse_args() -> Config:
    """Every setting comes from the config file; --config only chooses which one."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="settings file (hand-edited; never written back)")
    return Config.load(p.parse_args().config)


def _run(label, fn, failures: list[str]) -> None:
    """Run one check. Report the failure and keep going, so a single launch
    surfaces every problem instead of one per attempt."""
    try:
        detail = fn()
        if RANK == 0:
            print(f"[{OK}] {label}" + (f" -- {detail}" if detail else ""))
    except Exception as e:
        if RANK == 0:
            print(f"[{BAD}] {label} -- {type(e).__name__}: {e}", file=sys.stderr)
        failures.append(label)


def check_config(cfg: Config) -> None:
    """Reject an unrunnable config before a single GPU byte is allocated.

    Pure settings arithmetic and stat() calls: no model download, no CUDA
    context. A bad value caught here costs nothing; caught after the 4.9 GB
    pull it costs minutes, and mid-run it costs the whole job.
    """
    bad: list[str] = []
    if cfg.max_steps <= 0:
        bad.append("max_steps must be > 0: training length is step-based")
    if cfg.val_steps <= 0:
        bad.append("val_steps must be > 0: it also gates the checkpoint interval")
    if cfg.val_steps > cfg.max_steps:
        bad.append(f"val_steps ({cfg.val_steps}) > max_steps ({cfg.max_steps}): "
                   "the run would end without ever validating or saving")
    if cfg.save_checkpoint_steps <= 0:
        bad.append("save_checkpoint_steps must be > 0")
    elif cfg.save_checkpoint_steps % cfg.val_steps != 0:
        # load_best_model_at_end picks the best checkpoint by validation loss, so
        # every save has to land on a step that was also validated
        bad.append(f"save_checkpoint_steps ({cfg.save_checkpoint_steps}) must be a "
                   f"multiple of val_steps ({cfg.val_steps}), because "
                   "load_best_model_at_end selects among saved checkpoints")
    if cfg.save_checkpoint_steps > cfg.max_steps:
        bad.append(f"save_checkpoint_steps ({cfg.save_checkpoint_steps}) > max_steps "
                   f"({cfg.max_steps}): no checkpoint would ever be written")
    if cfg.save_total_limit <= 0:
        # 0 or negative means "never delete" to the Trainer, which is a disk
        # hazard at ~6 GB a checkpoint; say so rather than silently allowing it
        bad.append("save_total_limit must be >= 1 (it caps how many checkpoint "
                   "directories are kept; the best and the newest are never deleted)")
    if cfg.per_device_batch <= 0:
        bad.append("per_device_batch must be > 0")
    if cfg.grad_accum_steps <= 0:
        bad.append("grad_accum_steps must be > 0")
    if cfg.lr <= 0:
        bad.append("lr must be > 0")
    if not 0.0 <= cfg.warmup_ratio < 1.0:
        bad.append(f"warmup_ratio must be in [0, 1), got {cfg.warmup_ratio}")
    if not 0.0 <= cfg.label_smoothing < 1.0:
        bad.append(f"label_smoothing must be in [0, 1), got {cfg.label_smoothing}")
    if cfg.wer_subset < 0:
        bad.append("wer_subset must be >= 0 (0 disables the WER callback)")

    # Stage 1 must have run: training reads only what it wrote.
    for p in (cfg.train_split, cfg.val_split):
        if not p.exists():
            bad.append(f"missing {p} -- run scripts/01_prepare_data.py first")

    # Fail now if the output paths are unwritable, not after the first epoch.
    for label, p in (("ckpt_dir", cfg.ckpt_dir), ("log_dir", cfg.log_dir),
                     ("final_ckpt_for_inference", cfg.final_ckpt_for_inference)):
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / f".write_probe.{RANK}"
            probe.touch()
            probe.unlink()
        except OSError as e:
            bad.append(f"{label} {p} is not writable: {e}")

    # More than one visible GPU without torchrun makes the Trainer fall back to
    # nn.DataParallel, which cannot scatter the encoder_outputs compute_loss
    # passes and dies inside NCCL. Refuse it here rather than at step 0.
    visible = torch.cuda.device_count()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if visible > 1 and not distributed:
        bad.append(
            f"{visible} GPUs visible but not launched with torchrun -- the Trainer "
            f"would use DataParallel and fail in NCCL. Launch with:\n"
            f"         CUDA_VISIBLE_DEVICES={','.join(str(g) for g in cfg.gpus)} "
            f"torchrun --nproc_per_node={len(cfg.gpus)} scripts/02_train.py\n"
            f"         or set \"gpus\" to a single device in the config.")
    if distributed and visible != len(cfg.gpus):
        bad.append(f'{visible} GPUs visible but "gpus" lists {len(cfg.gpus)} '
                   f"-- CUDA_VISIBLE_DEVICES does not match the config")

    if bad:
        for b in bad:
            print(f"[config] {b}", file=sys.stderr)
        raise SystemExit(1)


def check_runtime(cfg: Config, model, collator, encoders, train_rows, val_rows,
                  stats: dict) -> None:
    """Everything that needs real tensors: CUDA, splits, a live batch, the
    freeze, and one forward/backward. Reuses the objects training is about to
    use, so this validates the actual path, not a replica of it.

    Missing packages are not checked for -- the import that needs them raises
    on its own, with a clearer message than any probe here would give."""
    failures: list[str] = []

    def gpus():
        assert torch.cuda.is_available(), "no CUDA device"
        return ", ".join(torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count()))
    _run("cuda available", gpus, failures)

    def split_ok():
        assert train_rows and val_rows, "empty split -- run 01_prepare_data.py first"
        overlap = {r["path"] for r in train_rows} & {r["path"] for r in val_rows}
        assert not overlap, f"{len(overlap)} utterances in BOTH train and val"
        return (f"train {len(train_rows)} / val {len(val_rows)}, "
                f"{sum(r['duration'] for r in train_rows)/3600:.2f}h train")
    _run("split is disjoint and non-empty", split_ok, failures)

    def round_trip():
        bad = [r for r in train_rows[:2000]
               if not encoders[r["lang"]].round_trips(r["text"])]
        assert not bad, f"{len(bad)} training targets do not round-trip"
        return "2000 sampled targets round-trip exactly"
    _run("targets are representable", round_trip, failures)

    ds = ManifestDataset(train_rows)
    batch: dict = {}

    def build_batch():
        batch.update(collator([ds[i] for i in range(4)]))
        f, m = batch["input_features"], batch["attention_mask"]
        assert f.shape[1] == 128, f"expected 128 mel bins, got {f.shape[1]}"
        assert m.shape == (f.shape[0], f.shape[2])
        plen = encoders[train_rows[0]["lang"]].prompt_len
        assert (batch["labels"][:, :plen - 1] == -100).all(), "prompt is being scored"
        return (f"features {tuple(f.shape)}, "
                f"decoder_input {tuple(batch['decoder_input_ids'].shape)}, "
                f"scored tokens {(batch['labels'] != -100).sum().item()}")
    _run("collator produces a valid batch", build_batch, failures)

    def frozen():
        assert not any(p.requires_grad for p in model.get_encoder().parameters())
        model.train()   # Trainer does this every step
        assert not model.get_encoder().training, "encoder came back out of eval mode"
        return (f"trainable {stats['trainable_params']/1e6:.1f}M / frozen "
                f"{stats['frozen_params']/1e6:.1f}M ({stats['trainable_pct']:.1f}%)")
    _run("encoder frozen and pinned to eval", frozen, failures)

    device = (torch.device(f"cuda:{LOCAL_RANK}") if torch.cuda.is_available()
              else torch.device("cpu"))

    def fwd_bwd():
        assert batch, "no batch to run -- the collator check failed"
        model.to(device)
        b = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            enc = model.get_encoder()(b["input_features"],
                                      attention_mask=b["attention_mask"])
        out = model(encoder_outputs=enc, decoder_input_ids=b["decoder_input_ids"],
                    use_cache=False)
        loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
            out.logits.reshape(-1, out.logits.size(-1)).float(), b["labels"].reshape(-1))
        assert torch.isfinite(loss), "loss is not finite"
        loss.backward()
        enc_grads = [n for n, p in model.named_parameters()
                     if p.grad is not None and n.startswith("model.encoder")]
        dec_grads = [n for n, p in model.named_parameters()
                     if p.grad is not None and n.startswith("model.decoder")]
        assert not enc_grads, f"{len(enc_grads)} encoder params received gradients"
        assert dec_grads, "no decoder gradients -- nothing would train"
        return (f"loss {loss.item():.4f} "
                f"(ln(vocab)={torch.log(torch.tensor(7152.0)):.2f}), "
                f"{len(dec_grads)} decoder tensors with grads, 0 encoder")
    _run("forward/backward, gradients only on the decoder", fwd_bwd, failures)

    # the check above left real gradients on the decoder; step 1 must not see them
    model.zero_grad(set_to_none=True)

    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed -- " + "; ".join(failures))
    if RANK == 0:
        print("[checks] all passed\n")


def main() -> int:
    cfg = parse_args()
    check_config(cfg)
    cfg.ensure_dirs()

    train_rows = mf.read_jsonl(cfg.train_split)
    val_rows = mf.read_jsonl(cfg.val_split)
    if not train_rows:
        print("empty train split -- run scripts/01_prepare_data.py first", file=sys.stderr)
        return 1
    print(f"[data] train {len(train_rows)} | val {len(val_rows)} | "
          f"{sum(r['duration'] for r in train_rows)/3600:.2f}h")

    tokenizer = load_tokenizer(cfg.repo_id)
    feature_extractor = load_feature_extractor(cfg.repo_id)
    encoders = MultiLangEncoder(tokenizer, [r["lang"] for r in train_rows + val_rows])

    model = load_model(cfg.repo_id)
    stats = freeze_encoder(model)
    keep_code_out_of_checkpoints(model)   # checkpoints hold weights + config only
    print(f"[model] {summarize(model)}")
    print(f"[model] trainable {stats['trainable_params']/1e6:.1f}M | "
          f"frozen {stats['frozen_params']/1e6:.1f}M ({stats['trainable_pct']:.1f}%)")

    collator = SpeechCollator(feature_extractor, encoders)
    check_runtime(cfg, model, collator, encoders, train_rows, val_rows, stats)

    train_ds = ManifestDataset(train_rows)
    val_ds = ManifestDataset(val_rows)

    # A SECOND front-end, deliberately: scoring runs it on the GPU, while the
    # collator's copy must stay on CPU inside the dataloader workers. Sharing one
    # object would drag CUDA buffers into the workers. It is only a window and a
    # filterbank, so the duplicate costs almost nothing.
    wer_fe = load_feature_extractor(cfg.repo_id)
    # WER/CER go to the console and to TensorBoard as val/wer and val/cer;
    # nothing is written to disk
    wer_cb = WerCallback(val_rows[:cfg.wer_subset], model, wer_fe, tokenizer)

    # the event files are protobuf; this is the same numbers in an editable file
    callbacks = [wer_cb, TextLogger(cfg.log_dir / "train.log", header={
        "repo_id": cfg.repo_id,
        "gpus": cfg.gpus,
        "max_steps": cfg.max_steps,
        "val_steps": cfg.val_steps,
        "lr": cfg.lr,
        "warmup_ratio": cfg.warmup_ratio,
        "per_device_batch": cfg.per_device_batch,
        "grad_accum_steps": cfg.grad_accum_steps,
        "label_smoothing": cfg.label_smoothing,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "bf16": cfg.bf16,
        "seed": cfg.seed,
        "train / val rows": f"{len(train_rows)} / {len(val_rows)}",
    })]
    if "tensorboard" in cfg.report_to:
        callbacks.append(ScalarLogger(cfg.log_dir))

    trainer = FrozenEncoderTrainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        label_smoothing=cfg.label_smoothing,
        lengths=train_ds.lengths,
        group_by_length=cfg.group_by_length,
        callbacks=callbacks,
    )
    wer_cb.trainer = trainer        # so the callback can emit val/wer to TensorBoard

    # if trainer.args.process_index == 0:
    #     print(f"[tensorboard] tensorboard --logdir {cfg.log_dir} --port 6006")

    trainer.train()

    final = cfg.final_ckpt_for_inference
    trainer.save_model(str(final))
    # No log_metrics/save_metrics: the end-of-run summary they report
    # (train_loss, train_runtime, total_flos, throughput) describes the run
    # rather than the model, and every one of those numbers is either
    # meaningless here or better read from the per-step curves.

    if trainer.args.process_index == 0:
        tokenizer.save_pretrained(str(final))
        feature_extractor.save_pretrained(str(final))
        src_dir = Path(code_dir())          # vendored code, no download
        for name in BUNDLE:
            if (src_dir / name).exists():
                shutil.copy2(src_dir / name, final / name)
        print(f"[done] {final}")
        # record the final location in the run log too, not just the console
        append(cfg.log_dir / "train.log", f"[done] final checkpoint saved to {final}")
        #print(f"[next] python scripts/04_evaluate.py --model {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
