# Release verification

`tutorials/blip_captioning_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned
  VizWiz-Captions dataset revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BlipCaptioningPipeline.from_pretrained(weights_dir=...)`, `fetch_annotations` and `fetch_images` from the pinned
  cache path, `build_sample_dataset(seed=SPLIT_SEED, image_paths=...)` / `load_byod_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `write_dataset_jsonl`, the ceiling print, `validate_inputs` with the tiny-image
  refusal probe, `pipe.caption` with the sanity checks, `keyword_hits` and the per-image `evaluation_report` on the
  drawn scenes, `constant_caption_baseline`, `colour_neighbour_baseline`, `pipe.evaluate` on the frozen model and on
  the validation and test splits after adaptation with the CIDEr-D assertions, the per-category breakdown through
  `cider_d` and `reference_captions`, `pipe.adapt` with its explicit hyperparameters, `evaluation_report` on the
  scenes after adaptation, `pipe.save_artifact`, `BlipCaptioningPipeline.from_artifact` and the reload-parity
  assertion, and the provenance fields `weight_format`, `weight_sha256`, `hosted_tf_weight_file_not_loaded` and the
  `corpus` block), the six expected `outputs/` paths, the learner-facing statements (BSD-3-Clause weights, no score
  exists, not evidence that it describes the image, adaptation with reference captions, the CC BY 4.0 corpus, the
  first row group, the two non-neural baselines, CIDEr-D, no dispersion estimate, the OCR exclusion, the
  weight-format note) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `BlipForConditionalGeneration` / `BlipProcessor` / `.generate(` /
  `from huggingface_hub import` / `get_hf_file_metadata` / `urllib.request` / `pyarrow` / `safetensors` /
  `torch.optim` / `.backward(` / `pipe._model` / `extractall(` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy`, `pillow`,
`huggingface-hub` and `pyarrow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner, annotation and image fetchers,
tiny PIL drawings, temporary manifests, no weights — `tests/test_model_backed.py` is skipped without the snapshot).
These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/blip-image-captioning-base/` or the corpus cache `weights/vizwiz-captions/` (the standalone
   path writes the manifest itself, stages the missing files from the Hub, reads the pinned VizWiz-Captions text
   columns and the pinned row group of photographs from the Hub, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `CAPTION_MAX_TOKENS = 30`, `EPOCHS = 4`, `LEARNING_RATE = 1e-5`,
   `BATCH_SIZE = 8`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2`, `pyarrow==25.0.1` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `BlipCaptioningPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `keyword_hits`, `unigram_f1`, `bleu4`,
     `rouge_l`, `cider_d`, `caption_metrics`, `medoid_caption`, `constant_caption_baseline`,
     `colour_neighbour_baseline`, `fetch_annotations`, `fetch_images`, `build_sample_dataset`, `validate_dataset`,
     `check_split_disjoint`, `split_dataset`, `load_byod_dataset`, `write_dataset_jsonl`, `reference_captions` and
     the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `Salesforce/blip-image-captioning-base` at
     the immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files, the 990 MB pickle
     re-hashed before `torch` is imported), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the
     verified directory with `source` `local-snapshot`;
   - Section 4: `fetch_annotations` checking the shard's declared size and SHA-256 against the pins, reading only
     its four text columns (1,550 rows) and matching the pinned text digest `9799ebb1…`; `fetch_images` reading the
     336 photographs of row group 0 with every size and SHA-256 matching; the seeded split of the 318 captioned
     photographs into 208 / 40 / 70 with `check_split_disjoint` reporting no shared image, the category mix printed
     (`text` a little over half) and the three dataset digests; `outputs/…_train.jsonl` written; the four dataset
     refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `IMAGE_SIZE` 384, `MAX_PREFIX_CHARS` 128,
     `MAX_NEW_TOKENS` 64, `DEFAULT_MAX_NEW_TOKENS` 30, `MIN_RECORDS` 8, `MAX_RECORDS` 5000, `MIN_CAPTIONS` 1,
     `MAX_CAPTION_CHARS` 500) surfaced; the three scenes drawn; `validate_inputs` writing
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the tiny-image probe);
     `pipe.caption` on the three scenes with every sanity check `True`, the keyword hits printed and the per-image
     `evaluation_report` verdict `not-measurable` (the card-pass smoke captioned the orange as a second apple; a
     different caption on another runtime is a finding to record, not a failure);
   - Section 6: the constant-caption baseline (CIDEr-D ≈ 0.05), the colour-neighbour baseline (≈ 0.04) and the
     frozen model's test score (CIDEr-D ≈ 0.59, BLEU-4 ≈ 0.21, ROUGE-L ≈ 0.45 in the build record on the RTX
     5070 Ti) with the per-category breakdown, and the cell's assertion that the frozen CIDEr-D is above the constant
     caption's;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 19,526,204 trainable of 223,971,644 parameters,
     208 training photographs and about 900 (image, caption) pairs, and a four-epoch history with validation CIDEr-D
     moving by hundredths (0.675 frozen → 0.699 → 0.656 → 0.709 → 0.716 in the build record; `best_epoch` 4);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on four metrics, the
     per-category breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test CIDEr-D
     exceeds the frozen one — on the CPU pre-flight 0.629 versus 0.590 (`no-text` 0.525 → 0.615, `text` 0.638 → 0.639),
     with BLEU-4 0.213 → 0.190 and mean caption length 8.9 → 11.2 words; 0.627 on the RTX 5070 Ti);
   - Section 9: the three scenes captioned by the adapted model with the `not-measurable` report and the keyword
     hits, `outputs/…_captions.csv` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (57 tensors, 78,112,280 bytes) and
     `BlipCaptioningPipeline.from_artifact` reloading it with 8/8 identical captions (the cell asserts it);
     `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the snapshot block
     (`weight_format`, `weight_sha256`, `hosted_tf_weight_file_not_loaded`), the `corpus` block, the
     inference-contract items, the comparison, the artifact digest, the reload parity, the runtime versions and
     device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `blip_captioning_colab.ipynb` (`E2E`) | `5cf3e88` / `bbced801` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-blip-captioning` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 355 files, 1075 MB staged from the Hub into a clean cache; comparison {bleu4: {constant: 0.041, neighbour: 0, frozen: 0.213, adapted: 0.19}, rouge_l: {constant: 0.336, neighbour: 0.227, frozen: 0.453, adapted: 0.439}, cider_d: {constant: 0.052, neighbour: 0.042, frozen: 0.59, adapted: 0.629}, unigram_f1: {constant: 0.385, neighbour: 0.251, frozen: 0.492, adapted: 0.49}, mean_words: {constant: 11, neighbour: 12.2, frozen: 8.9, adapted: 11.2}, delta_vs_frozen: {bleu4: -0.023, rouge_l: -0.014, cider_d: 0.039, unigram_f1: -0.002}, by_category: {no-text: {n: 30, constant: 0.053, frozen: 0.525, adapted: 0.615}, text: {n: 40, constant: 0.051, frozen: 0.638, adapted: 0.639}}}; reload parity {identical_captions: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-blip-captioning/v2/evidence/` in the workspace |
| `blip_captioning_colab.ipynb` (`TASK-INFERENCE`, superseded) | `504cd92` / `88c86a876f02` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-blip-captioning` v1) | PASSED — 8/8 code cells, 242.6 s, 18 files, 991 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/blip_captioning_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/blip_captioning_colab.ipynb`). Wall times, when recorded, are the sum of per-cell
times reported by the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `5cf3e88` / `bbced801` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-blip-captioning` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 439.3 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 355 files, 1075 MB staged from the Hub into a clean cache; comparison {bleu4: {constant: 0.041, neighbour: 0, frozen: 0.213, adapted: 0.19}, rouge_l: {constant: 0.336, neighbour: 0.227, frozen: 0.453, adapted: 0.439}, cider_d: {constant: 0.052, neighbour: 0.042, frozen: 0.59, adapted: 0.629}, unigram_f1: {constant: 0.385, neighbour: 0.251, frozen: 0.492, adapted: 0.49}, mean_words: {constant: 11, neighbour: 12.2, frozen: 8.9, adapted: 11.2}, delta_vs_frozen: {bleu4: -0.023, rouge_l: -0.014, cider_d: 0.039, unigram_f1: -0.002}, by_category: {no-text: {n: 30, constant: 0.053, frozen: 0.525, adapted: 0.615}, text: {n: 40, constant: 0.051, frozen: 0.638, adapted: 0.639}}}; reload parity {identical_captions: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-blip-captioning/v2/evidence/` in the workspace |
| 2026-09-19 | generated at `fbdbb67` / blob `765b1875cf2d` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, snapshot, annotation cache and the 336 photographs pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (8 files), annotations read from the pre-staged cache and 336 photographs re-hashed, 318 captioned split 208 / 40 / 70 by image (`text` 110 / 25 / 40), three scene captions with the recorded orange miss, baselines CIDEr-D 0.052 / 0.042, frozen test CIDEr-D 0.590 (31.9 s; `no-text` 0.525, `text` 0.638), four epochs 486.7 s over 889 pairs (validation CIDEr-D 0.675 → 0.699 → 0.656 → 0.709 → 0.716, epoch 4 kept), adapted test CIDEr-D 0.629 / BLEU-4 0.190 / ROUGE-L 0.439 (`no-text` 0.615, `text` 0.639; mean words 8.9 → 11.2), fruit scene re-captioned `an apple and an orange on a white background`, adapter 78,112,280 B / 57 tensors, reload parity 8/8, 6 outputs written; the committed blob differs from the executed one in markdown prose only (CPU figures filled in after this run) | 878.8 s | PASS — pre-flight only; not promotion evidence |

## Current status

**Release-grade.** The `E2E` notebook blob `bbced801` (committed at `5cf3e88`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 439.3 s, 355 files, 1075 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should still weigh: the frozen model is already a competent captioner on VizWiz photographs (CIDEr-D 0.590 against 0.052 for the best constant caption on the T4 run), so the adaptation gain is small — 0.590 → 0.629 CIDEr-D, carried by the `no-text` photographs (0.525 → 0.615) while the `text` ones stay at 0.64, with BLEU-4 and ROUGE-L moving slightly the other way as the captions lengthen toward the corpus's style — and the 70-photograph test split carries no dispersion estimate; the T4 run reproduced the CPU pre-flight's numbers to three decimals (same seed, deterministic path); a higher learning rate (5e-5) or twice the training data at 2e-5 overfit within two epochs in the build record and the selector kept the frozen weights or lost held-out CIDEr-D; the 40-photograph validation split makes epoch selection noisy; and the drawn scenes re-captioned after adaptation (the orange now named) are three images of evidence about behaviour outside the corpus, not a measurement.
