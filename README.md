# Decoder fine-tuning for `bodhan-ai/indic-transcribe-core`

Fine-tunes the transformer decoder on IndicTTS_Marathi dataset while the FastConformer encoder
stays frozen.

## Pipeline

| Stage | Command | Reads | Writes |
|---|---|---|---|
| 1. tokenization audit, manifest, split | `scripts/01_prepare_data.py` | raw corpus | `reports/text_audit.json`, `manifest.jsonl`, `splits/*.jsonl` |
| 2. checks + training | `scripts/02_train.py` | splits | `checkpoints/`, `final/`, `logs/` |

Stage 2 runs its own checks first -- settings, dependencies, CUDA, split
integrity, target round-trip, a live batch through the collator, the encoder
freeze, and a forward/backward proving gradients reach only the decoder.

Library stages, in dependency order: `config` -> `hub` -> `text` (tokenization)
-> `manifest` -> `data` (dataloading) -> `modeling` (load + freeze) ->
`training` (loss, sampler, callbacks) -> `metrics`.

