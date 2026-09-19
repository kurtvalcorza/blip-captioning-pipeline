# BLIP image-captioning-base image captioning pipeline

DIMER inference and fine-tuning wrapper for **BLIP fine-tuned on COCO Captions** (`Salesforce/blip-image-captioning-base`), Salesforce's 224M-parameter vision-language model (ViT-B/16 image encoder at 384×384 and a BERT-style text decoder cross-attending to the image features) that describes an image in one generated English sentence, optionally continuing a prefix you supply — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts one image and an optional prefix, decodes greedily under a caller-owned token budget, and returns the caption string with a `truncated` flag; it returns no score and no location. It also carries a bounded adaptation contract: `adapt` fine-tunes the caption decoder's last blocks and head on a validated `{id, image, captions}` dataset, `evaluate` scores a held-out split with BLEU-4, ROUGE-L, CIDEr-D and unigram F1 beside two non-neural baselines, and `save_artifact` / `from_artifact` export and reload the trained tensors as a safetensors adapter bound to the pinned base.

**Weight format.** Upstream hosts no SafeTensors at the pinned revision. This package executes the digest-pinned `pytorch_model.bin` (a pickle, deserialised with `weights_only=True` only after its SHA-256 matched the manifest); the `tf_model.h5` upstream also hosts is the DIMER upload artifact (DIMER does not accept `.bin`) and is never loaded here. Both digests are recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## Upstream alignment

- Model: `Salesforce/blip-image-captioning-base`
- Revision: `82a37760796d32b1411fe092ab5d4e227313294b`
- Upstream weight license: BSD-3-Clause
- Upstream task: image captioning (COCO Captions)
- Repository adaptation: bounded supervised fine-tuning of the caption decoder's last *k* blocks plus head transform and bias (`adapt`; the vision encoder, embeddings and tied output projection stay frozen); the tutorial's default corpus is VizWiz-Captions (`mm-eval/VizWiz-Captions` @ `c4a6d897836e7885d0095134f92d392e4e770539`, CC BY 4.0), read column-only plus one row group of photographs with per-file digests at run time

## Quick start

```python
from PIL import Image
from blip_captioning_pipeline import BlipCaptioningPipeline, keyword_hits

pipe = BlipCaptioningPipeline.from_pretrained()        # stages + verifies weights/blip-image-captioning-base first
result = pipe.caption(Image.open("photo.jpg"))          # unconditional
print(result["caption"], result["new_tokens"], result["truncated"])   # generated text; no score exists

result = pipe.caption(Image.open("photo.jpg"), prefix="a photography of")   # conditional (upstream README example)
print(keyword_hits(result["caption"], ["house", "tree"]))   # an observation, not a metric

# Adaptation: records are {id, image, captions}; every image stays in one split
from blip_captioning_pipeline import fetch_sample_dataset, constant_caption_baseline

splits = fetch_sample_dataset()                          # pinned VizWiz-Captions sample, 208 / 40 / 70 by image
print(constant_caption_baseline(splits["train"], splits["test"])["cider_d"])
print(pipe.evaluate(splits["test"])["cider_d"])          # frozen model
pipe.adapt(splits["train"], splits["validation"], epochs=4, lr=1e-5)   # last 2 decoder blocks + head
print(pipe.evaluate(splits["test"])["cider_d"])          # adapted model, same held-out photographs
pipe.save_artifact("outputs/adapter")                    # adapter.safetensors + manifest.json
again = BlipCaptioningPipeline.from_artifact("outputs/adapter")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (36 tests, no weights needed; `tests/test_model_backed.py` adds 6 model-backed tests, one on CUDA, when the snapshot is staged). On a fresh clone the manifest is committed but the weights are not: `BlipCaptioningPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/blip-image-captioning-base/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (8 files)
  config.json                # BlipForConditionalGeneration: ViT-B/16 @ 384 + 12-layer text decoder (vocab 30524)
  preprocessor_config.json   # BlipImageProcessor: resize 384x384, CLIP mean/std
  special_tokens_map.json  tokenizer.json  tokenizer_config.json  vocab.txt
  pytorch_model.bin          # git-ignored, 989,820,849 bytes — the executed artifact (pickle, digest-checked)
  README.md
```

`tf_model.h5` (990,275,136 bytes) exists upstream, is the DIMER-hosted blob, and is deliberately not listed or loaded; there is no `model.safetensors`.

## Input ceilings and request parameters

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`; `MAX_PREFIX_CHARS = 128` (optional prefix, whitespace collapsed); `MAX_NEW_TOKENS = 64`, `DEFAULT_MAX_NEW_TOKENS = 30`; `DECODING = "greedy"`; `IMAGE_SIZE = 384` (the processor's fixed resize, aspect ratio not preserved; documentation only). See `MODEL_CARD.md` for who owns the budget, why the report is `not-measurable` without references, and the measured CPU timings.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/blip-captioning-pipeline/blob/main/tutorials/blip_captioning_colab.ipynb)

`tutorials/blip_captioning_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules (`pipeline.py`, `metrics.py`, `samples.py`), model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path resolves the pinned model through the carried staging and verification path (the 990 MB pickle re-hashed before `torch` is imported), reads the text columns of one pinned VizWiz-Captions shard column-only and the 336 photographs of its first row group in one digest-checked range read, validates and splits them by image (208 / 40 / 70), captions three cartoon scenes drawn in code through `validate_inputs` and `caption` with keyword observations, scores the frozen model on the test photographs beside the constant-caption and colour-nearest-neighbour baselines (BLEU-4, ROUGE-L, CIDEr-D, unigram F1, per `text` / `no-text` category), fine-tunes the caption decoder's last two blocks and head for four epochs with validation-CIDEr-D epoch selection, scores the held-out split again, re-captions the scenes, and exports a safetensors adapter that it reloads with verified parity — one seeded split of one corpus, no benchmark claim. BYOD (one zip of images plus `records.jsonl`) is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. One local fresh-kernel execution is recorded in `docs/release-verification.md` as pre-flight; the supported-runtime run is pending. Complete that record against the exact release revision before calling the notebook release-grade.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests (both weight files), input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, the executed-vs-hosted artifact note and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are BSD-3-Clause; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
