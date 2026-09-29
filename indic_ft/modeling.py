"""Stage 4 -- model loading and the encoder freeze."""
from __future__ import annotations

import types

import torch
import torch.nn as nn

from .hub import code_dir, weights_dir


def load_model(repo_id: str = "bodhan-ai/indic-transcribe-core"):
    code_dir()                              # vendored classes onto sys.path first
    from modeling_indic_canary import IndicCanaryForConditionalGeneration

    # code from the committed package, weights from the local cache -- the
    # concrete class means no trust_remote_code and no auto_map dependency
    model = IndicCanaryForConditionalGeneration.from_pretrained(weights_dir(repo_id))
    model.config.use_cache = False          # training is teacher-forced

    # lm_head is tied to the decoder token embedding, but the port declares
    # _tied_weights_keys as a dict (transformers-v5 style) -- verify, don't assume.
    if model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(), \
        "lm_head is not tied to the decoder embedding; gradients would diverge from pretraining"
    return model


def keep_code_out_of_checkpoints(model) -> None:
    """Stop the Trainer copying the model's remote-code .py into every checkpoint.

    save_pretrained copies configuration_*.py / modeling_*.py only when
    is_remote_code() is true, which is gated on the class attribute _auto_class.
    Clearing it suppresses the copy while leaving config.auto_map in place, so
    the intermediate checkpoints hold weights + config only. Nothing loads a
    checkpoint via from_pretrained -- load_best_model_at_end reads the state
    dict directly -- so the absent code costs nothing. The final export is
    unaffected: 02_train.py copies the full code bundle into it explicitly.
    """
    type(model)._auto_class = None
    model._auto_class = None
    model.config._auto_class = None


def freeze_encoder(model) -> dict:
    """Freeze the FastConformer encoder in three layers.

    1. requires_grad_(False)      -- no gradients
    2. pin .train() to eval       -- Trainer calls model.train() every step and
                                     would otherwise walk back into the encoder
    3. (see training.compute_loss) -- run it under no_grad, outside the DDP
                                     wrapper, so no graph is kept and there is
                                     nothing to all-reduce
    """
    encoder = model.get_encoder()
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.train = types.MethodType(lambda self, mode=True: nn.Module.train(self, False), encoder)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    assert not any(p.requires_grad for p in encoder.parameters()), "encoder is not frozen"
    return {
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 100 * trainable / (trainable + frozen),
    }


def summarize(model) -> str:
    c = model.config
    return (f"d_model={c.d_model} encoder_layers={c.encoder_layers} "
            f"decoder_layers={c.decoder_layers} vocab={c.vocab_size}")
