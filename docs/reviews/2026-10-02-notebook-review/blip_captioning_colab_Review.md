# BLIP Image-Captioning E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/blip-captioning-pipeline`  
**Notebook:** `tutorials/blip_captioning_colab.ipynb`  
**Reviewed commit:** `61d1be27f00046decc349cb4c7be102d535d4e68` (`main`, confirmed with `gh api repos/kurtvalcorza/blip-captioning-pipeline/commits/main`)  
**Notebook Git blob:** `bbced8017b73a9ba4a5c979459bd99926fd644e0`. This is the blob committed at `5cf3e88` and executed in the recorded Kaggle run of 2026-09-19. Later commits changed docs and tests only; generator `--check` exits 0 at the reviewed commit.  
**Finding prefix:** `CAP`

## Executive assessment

The default path is careful engineering and it reproduces. The notebook carries its three modules byte for byte (generator `--check` and the release validator both exit 0), digest-verifies the pinned pickle before `torch.load(weights_only=True)`, reads a digest-pinned real captioning corpus column-only, splits it by image, frames the model with two non-neural baselines and four metrics, states plainly that a caption has no score, and reloads its safetensors adapter with asserted parity. A direct CPU execution of all 11 code cells, verbatim, with the real weights and the exact pins, reproduced the recorded numbers to three decimals: baselines CIDEr-D 0.052 / 0.042, frozen 0.590, validation history 0.675 → 0.699 → 0.656 → 0.709 → 0.716, adapted 0.629, `no-text` 0.525 → 0.615, parity 8/8.

Five problems stand in the way of `Ready for intended use`:

1. `Run all` in a fresh hosted runtime does not finish in one pass. The recorded qualification run's first pass stopped in cell 3 with `RuntimeError: … numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime`, yet `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` record a `Run all` PASS (CAP-M1).
2. The held-out test split is not independent. The default learning rate is described as "the configuration that gained on the held-out split", and Section 8 **asserts** that the adapted test CIDEr-D beats the frozen one. When it does not, the notebook stops before export. The notebook's own first optional experiment (`LEARNING_RATE = 5e-5`) therefore ends in a bare `AssertionError` (verified: selector keeps epoch 0, adapted = frozen = 0.5895) (CAP-M2).
3. Rerunning a section after a field change reuses the already adapted model. Section 7 then labels the VizWiz-adapted model "frozen model" at epoch 0 (0.716 instead of 0.675), and the learner's experiment silently reproduces the previous result. The documented BYOD route ("re-run from that cell") has the same flaw (CAP-M3).
4. The BYOD contract's stated minimum (8 records) is not the enforced one (50 for the default fractions). A second upload silently trains on the first upload's records. Images in a zip subfolder cannot be referenced, and a cancelled upload or a zip without a records file raises a bare `StopIteration` (CAP-M4).
5. The notebook is declared `GUIDED`, but most of the guided layer is absent, and 1,883 lines of carried code are not labelled or collapsed as infrastructure (CAP-M5).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, and the opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` `b1cfe13` |
| Intended audience | Not stated. The Prerequisites ask for basic Python and PIL, encoder–decoder generation, pickle-checkpoint risk, and what BLEU-4, ROUGE-L and CIDEr-D measure |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32, CUDA when present; ~991 MB model + ~84 MB photographs |
| Promised outcomes | One-pass `Run all` with no configuration edit; pinned install; three carried modules; digest-verified pickle snapshot; digest-pinned VizWiz-Captions text and 336 photographs; validation, 208/40/70 image-disjoint split, four refusals; inference contract on three drawn scenes; two baselines and the frozen model on four metrics, per category; bounded fine-tuning of the last two decoder blocks with validation-CIDEr-D epoch selection; held-out evaluation; scenes re-captioned; safetensors adapter with reload parity; six `outputs/` files; BYOD "through the same validation, … fine-tuning, held-out evaluation, artifact export and reload-parity cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/blip_captioning_pipeline/` @ `fbdbb67` |

### Evidence actually obtained

- **Source inspection:** all 25 cells (11 code), the carried `pipeline.py` (`stage_missing_files`, `evaluate`, `adapt`, `save_artifact`), `samples.py` (`_check_record`, `validate_dataset`, `split_dataset`, `load_byod_dataset`, `fetch_*`), the generator and template, `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md`, `docs/release-verification.md`.
- **Documented execution evidence:** `docs/release-verification.md` and the archived executor summary `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-blip-captioning/v2/evidence/run_summary.json` (workspace). Kaggle Tesla T4, 2026-09-19, **blob `bbced801`, the reviewed blob**: attempt 1 failed in cell 3 with the restart `RuntimeError` after 162.2 s; attempt 2, after an interpreter restart, ran 11/11 cells (`restarted_after_install_cell: true`, 439.3 s total). No Colab run of this blob and no BYOD run are recorded.
- **Direct execution (this review):** `run_probes.py` on Windows, CPython 3.12.10, in an existing workspace venv that holds the notebook's exact pins with CPU torch (`torch 2.14.0+cpu`, `transformers 4.57.6`, `safetensors 0.8.0`, `numpy 2.5.3`, `pillow 11.3.0`, `huggingface-hub 0.36.2`, `pyarrow 25.0.1`); CPU only, `HF_HUB_OFFLINE=1`. The **real weights** and the pinned VizWiz cache were hard-linked into a scratch working directory: the notebook wrote its own manifest, `stage_missing_files` fetched nothing and every file was re-hashed, so the runtime was not clean and nothing was downloaded. Cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` (install skipped). Every code cell ran **verbatim from the notebook JSON** in one namespace; only form-field literals were substituted, and `google.colab.files.upload` was replaced by a fake. BYOD images are **stand-ins** (flat coloured shapes drawn with Pillow, two captions each). The default path took 527 s of cell time. Static checks: JSON parse, compile of all 11 code cells, blob id, generator `--check` (exit 0), `tools/validate_release_assets.py` (exit 0).
- **Not verified:** a Colab run of any kind; a one-pass hosted `Run all`; the CUDA path in this review; BYOD through the real upload widget or with real photographs; learner understanding.

## 2. Separate judgments

- **Technical correctness:** sound on the default path (identity, digests, image-disjoint split, train-only baselines, export, reload parity; direct execution matched the record). Defects: the restart-dependent install (CAP-M1); result assertions that stop the notebook on a legitimate outcome (CAP-M2); state that survives a rerun and is mislabelled (CAP-M3); the BYOD loader and contract (CAP-M4).
- **Scientific validity:** the metrics, baselines, per-category breakdown and the "few hundredths of CIDEr-D, BLEU-4 the other way" reading are honest. But the default hyperparameters were chosen by their held-out test result, and the test assertion encodes that choice, so the 0.590 → 0.629 test gain is not independent evidence (CAP-M2). The 0.039 gain has no dispersion estimate (stated by the notebook).
- **Promise fulfilment:** the default capability list is delivered. One-pass `Run all` is not (CAP-M1). The optional experiments are not deliverable as written (CAP-M2, CAP-M3). BYOD is delivered for one zip layout of 50+ records uploaded once into a fresh base (P5 stand-in run reached export with parity 8/8), not for the stated contract (CAP-M4).
- **Learner experience:** dense, careful stage prose with two "Look for" notes and explicit limits; no prediction, checkpoint, worked answer, troubleshooting or conclusion template, and 1,883 lines of unlabelled infrastructure (CAP-M5).
- **Spec conformance (2.2):** applicable `MUST`s fail: RUN1, RUN10, ENV6, REL2 (CAP-M1); SPL7, RUN9/UX7 (CAP-M2); UX7 and OUT8 (CAP-M3); DAT12, DAT19, REL12 (CAP-M4). GDL1–GDL15 largely unmet (SHOULD). Declared spec 2.0 (CAP-S1).

## 3. Promise and objective tracing

| Claim (where) | Implementation | Observable result (this review) | Learner interpretation |
|---|---|---|---|
| One-pass `Run all`, no intervention (cell 0) | cell 3 in-kernel `pip install` + stale-import guard | Kaggle record: attempt 1 `RuntimeError` (numpy 2.0.2 → 2.5.3), restart, attempt 2 ok; local: install skipped | **Not delivered** (CAP-M1) |
| Pinned, digest-verified pickle snapshot (cells 10–11) | `stage_missing_files`, `verify_snapshot`, `from_pretrained` | 8 files verified, `fetched: []` (pre-staged), `local-snapshot`, cpu | Delivered |
| Pinned VizWiz text + 336 photographs, 208/40/70 by image, four refusals (cells 12–13) | `fetch_annotations`, `fetch_images`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint` | 208/40/70, `text` 110/25/40, four probes rejected with the rule named | Delivered |
| Inference contract on three drawn scenes (cells 14–15) | `validate_inputs`, `caption`, `keyword_hits`, `evaluation_report` | captions equal the card's; orange missed; 4×4 probe rejected; verdict `not-measurable` | Delivered, well explained |
| Baselines + frozen model, per category (cells 16–17) | `constant_caption_baseline`, `colour_neighbour_baseline`, `evaluate` | 0.052 / 0.042 / 0.590; `no-text` 0.525, `text` 0.638 | Delivered |
| Bounded fine-tuning with validation epoch selection (cells 18–19) | `adapt(lr=1e-5, epochs=4)` | 19,526,204 trainable, 889 pairs, val 0.675 → 0.716, best epoch 4, 307 s | Delivered |
| Held-out evaluation, "never used for … selection" (cells 20–21) | `evaluate` + `assert adapted > frozen` | 0.629 vs 0.590; but the default config was chosen on this split and the assert gates on it | **Not independent** (CAP-M2) |
| Re-caption scenes, export, reload parity (cells 22–23) | `save_artifact`, `from_artifact` | fruit now "an apple and an orange"; 57 tensors, 78,112,280 B; parity 8/8; 6 outputs | Delivered |
| "set `LEARNING_RATE = 5e-5` … the selector keeps epoch 0" (cell 24) | cells 19→21 | after a fresh base: best epoch 0, adapted = frozen = 0.5895, `AssertionError` in cell 21, Section 9 never runs | **Crashes** (CAP-M2); without a fresh base, epoch 0 is the old adapted model (CAP-M3) |
| BYOD "8..5,000 records", "re-run from that cell" (cells 0, 1, 13) | cell 13 zip flatten + `load_byod_dataset` + `split_dataset` + per-split `validate_dataset` | 8–49 records fail in Section 4; 50 and 60 pass; 60 stand-ins on a fresh base reach export with parity 8/8 | Partly delivered (CAP-M3, CAP-M4) |

| Objective (cell 0) | Learner activity | Evidence it was exercised |
|---|---|---|
| install; read what the carried modules guarantee; stage and digest-verify | run cells | Procedural only |
| read captions/photographs of a pinned corpus, validate, split by image | read printed digests and refusals | Shown, not practised |
| caption and read `caption` / `new_tokens` / `truncated` (no score) | read the scene output | Shown; no check of the reading |
| score frozen model vs baselines, read per category | read tables | Shown; no prediction or question |
| bounded fine-tuning with validation selection; evaluate on a disjoint test split | run cells | Runs; test independence compromised (CAP-M2) |
| export and reload with parity | run cell | Shown |
| (optional) learning rate, decoder layers, epochs, BYOD | edit a field, no rerun instructions | First experiment crashes (CAP-M2); reruns are stale (CAP-M3) |

## 4. Prioritized findings

### CAP-M1 — Major: fresh-runtime `Run all` needs a manual restart after the install cell, yet the release records call it a `Run all` PASS

- **Cell/section:** Section 1 (cell 3); generator `tools/build_notebook.py:47-70` (`_INSTALL_GUARD`) and `:453-471`.
- **Observed issue:** cell 3 pip-installs 9 pins into the running kernel and raises `RuntimeError(… 'Restart the runtime, then rerun from the top.')` when a pin replaces an imported distribution. The recorded qualification run of this exact blob did exactly that: `run_summary.json` attempt 1 failed with `numpy: loaded=2.0.2, installed=2.5.3` (plus `cuda-bindings`), and attempt 2 ran after a restart. `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` record this as "PASSED — 11/11 code cells ok (1 restart after install cell)" and "Release-grade", and the release procedure (step 4) declares the restart "expected". The opening cell promises "no configuration edit" and a single `Run all`.
- **Consequence:** a learner's first `Run all` stops in cell 3. The release status rests on a two-pass run the spec defines as non-conformant (§1, RUN10, §25.7).
- **Evidence:** documented execution (archived executor summary, release record); source inspection. Whether Colab's preloaded packages trigger it identically is **not verified**, but the Kaggle image's NumPy 2.0.2 is typical of hosted images and the pin is 2.5.3.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator (`tools/build_notebook.py` / `tools/notebook_template.py`), regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Colab (or Kaggle) runtime runs **Run all** once, with no restart and no intervention, through cell 23; the executor summary shows one pass. README, STATUS, `tutorials/README.md` and `docs/release-verification.md` no longer call the 2026-09-19 two-pass run a Run-all PASS, and the procedure no longer calls a restart "expected".
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11, §25.7 (MUST).

### CAP-M2 — Major: the test split chose the default and gates the run; a non-improving adaptation stops the notebook before export

- **Cell/section:** Section 7 prose (cell 18), Section 8 (cells 20–21), Section 6 (cell 17), optional experiments (cell 24); template `tools/notebook_template.py:348`, `:366-368`, `:397`, `:433`, `:533-535`; MODEL_CARD §Variability.
- **Observed issue:** (a) Cell 18 says "the default is the configuration that gained on the held-out split", and MODEL_CARD records the sweep in held-out terms ("2e-5 over twice the photographs lowered held-out CIDEr-D to 0.496"). Cell 20 then says the test photographs "were never used for training or epoch selection" and presents the 0.590 → 0.629 test delta as the evidence. The hyperparameters were selected on that split. (b) Cell 21 ends with `assert adapted_test['cider_d'] > frozen_test['cider_d']`, and cell 17 with `assert frozen_test['cider_d'] > baseline_constant['cider_d']`. A correct, honest outcome (the selector keeps epoch 0; adaptation does not help; the frozen model loses to the constant caption on the user's corpus) becomes a crash. (c) The first optional experiment in cell 24 is exactly such an outcome.
- **Consequence:** the central held-out number is not independent evidence, and the learner is told it is. Any learner who tries the notebook's own `LEARNING_RATE = 5e-5` experiment, any BYOD corpus where adaptation does not beat the frozen model on 20 % of the data, and any seed where the noisy 40-photograph validation picks a worse epoch, end in a bare `AssertionError` with no explanation. `Run all` stops there: no captions CSV, no adapter, no reload, no `result.json`. The spec's reference practice is that "a negative held-out result is preserved rather than replaced" (§25.13).
- **Evidence:** direct execution P3: fresh base (cell 11 rerun), `LEARNING_RATE = 5e-5`, 4 epochs: validation CIDEr-D 0.675 → 0.628 → 0.594 → 0.587 → 0.528, best epoch 0, adapted test = frozen test = 0.5895, cell 21 `AssertionError`; Section 9 not reached. Source inspection of cells 17, 18, 20, 21 and MODEL_CARD.
- **Recommended correction:** replace both result assertions with reporting: print the delta and, when the selector keeps epoch 0 or the delta is ≤ 0, say so and explain it (export still proceeds; the artifact manifest records `best_epoch`). Freeze the hyperparameters on validation only, or state truthfully that the default was chosen with the test split and therefore the test delta is optimistic; preferably carve a development split from training for the sweep and re-run the sweep without the test split. Rewrite cells 18, 20 and 24 accordingly. Consider a bootstrap interval over the 70 test images (CAP-S2).
- **Acceptance check:** (a) with `LEARNING_RATE = 5e-5` on a fresh base, cells 19→23 complete, cell 21 prints that epoch 0 was kept and the delta is 0.000, and `result.json` records it. (b) No learner-facing text claims the test split was unused for selection unless the documented sweep used only training/validation data. (c) `grep "^assert .*cider" tutorials/*.ipynb` returns nothing.
- **Spec:** SPL6, SPL7, EVAL14 (MUST); RUN9, UX7 (MUST); §25.13.

### CAP-M3 — Major: rerunning a section after changing a field reuses the adapted model and labels it "frozen"

- **Cell/section:** Section 7 (cell 19) and Section 4 / BYOD instruction (cells 0, 13); carried `pipeline.adapt` (starts from the current weights; `history[0]` is always noted "frozen model"); template `tools/notebook_template.py:41-44`, `:533-535`.
- **Observed issue:** `adapt` trains from whatever weights `pipe` currently holds, and Section 3 is the only place a base model is loaded. The optional experiments give no rerun instruction, so a learner changes `LEARNING_RATE` and reruns Section 7. Epoch 0 is then the previously adapted model printed as `'note': 'frozen model'`. In P2 it scored 0.716 (the old best) instead of the true frozen 0.675. The new epoch fell to 0.541, the selector kept "epoch 0", and Section 8 printed the old comparison (adapted 0.629) unchanged, so the learner would conclude that 5e-5 behaves like 1e-5. Rerunning Section 9 would then write an artifact whose manifest records `lr = 5e-5, epochs = 1, best_epoch = 0` around tensors trained at 1e-5. The BYOD instruction ("set `USE_BYOD = True` in Section 4 and re-run from that cell") has the same mechanism: Section 6's "frozen model" is the VizWiz-adapted model, and Section 7 adapts on top of it (source inspection; the P5 BYOD run deliberately reloaded the base first).
- **Consequence:** the documented experiments and the BYOD comparison report a baseline that is not the base model, under the base model's name, and an artifact whose provenance misdescribes its tensors.
- **Evidence:** direct execution P2 (rerun of cells 19 and 21 after the default run, `LEARNING_RATE = 5e-5`, `EPOCHS = 1`): epoch-0 val CIDEr-D 0.7161 (= P1 adapted val) vs true frozen 0.6749; best epoch 0; comparison unchanged. Source inspection of `adapt` and `save_artifact`.
- **Recommended correction:** make Section 7 start from the verified base every time (reload in the cell, or have `adapt` refuse to start when `self.adapter is not None` unless told to continue, with the message naming the cell to rerun), and record in `history[0]` whether epoch 0 is the base. State the rerun range for each optional experiment and for BYOD ("rerun from Section 3" or a reset helper), following GDL10's Predict → Change → Run → Observe → Explain.
- **Acceptance check:** after a completed default run, changing `LEARNING_RATE` and rerunning Section 7 either reloads the base (epoch-0 val CIDEr-D equals 0.675 on the sample) or stops with an actionable message; the BYOD path's Section 6 "frozen" score equals the base model's score on the BYOD test split.
- **Spec:** UX7 (MUST), OUT8 / ART8 (MUST), GDL10, UX10.

### CAP-M4 — Major: the BYOD contract's stated limits and layout are not the enforced ones; bad or repeated uploads fail late, silently or opaquely

- **Cell/section:** Prerequisites (cell 1), BYOD paragraph (cell 0), Section 4 (cell 13); carried `samples.py` `split_dataset`, `validate_dataset`; template `tools/notebook_template.py:41-44`, `:130`, `:161-180`.
- **Observed issue** (direct execution P4 on stand-in zips through cell 13, unless noted):
  1. **Wrong minimum.** The contract says "a dataset needs 8..5,000 records". `split_dataset` takes 20 % test and 15 % validation, and cell 13 then validates **each split** with the default `min_records = 8`. 8 records fail with "split leaves 5 training records"; 12, 20, 40 and 49 fail with "2 / 4 / 6 / 7 records; 8..5000 are required", a message that names neither the split nor the real rule. The smallest passing dataset is **50** records (direct API scan, 8..80).
  2. **Silent stale data.** `work/byod` is never cleared. Uploading `b.zip` (with `records.json`) after `a.zip` (with `records.jsonl`) printed `data_source: 'BYOD (b.zip)'` while all 60 records came from upload A, because `records.jsonl` is searched first.
  3. **Subfolders.** Members are flattened to their base names, but `image` is resolved as written: a zip with `photos/img_000.jpg` referenced as `photos/img_000.jpg` fails with `image file not found: work\byod\photos\img_000.jpg`. Two images with the same name in different folders overwrite each other silently (source inspection).
  4. **Opaque failures.** A cancelled upload and a zip without `records.jsonl` / `records.json` both raise a bare `StopIteration`.
  5. **Provenance under BYOD.** Cell 13 prints the VizWiz `text_sha256` and `pinned_photographs: 336`, and cell 23 writes the VizWiz `corpus` block into `result.json`, even when the data is the user's.
- **Consequence:** a user with a modest labelled set, the likely BYOD user, meets the documented contract and is rejected with a message about "6 records" when they uploaded 40. A second attempt in the same session can train and evaluate on the wrong data under the new file name.
- **Evidence:** direct execution P4 (cell 13 verbatim, fake upload; stand-in images); positive control P5: 60 stand-in records on a fresh base ran cells 13→23, adapted CIDEr-D above frozen, artifact 57 tensors, parity 8/8. Empty-caption rejection is clear (`records[0]: every reference caption must be a non-empty string`).
- **Recommended correction:** state the true minimum (derived from the fractions and `MIN_RECORDS`) before upload, or validate splits with a smaller per-split minimum and name the split in the message. Clear `work/byod` before extracting. Preserve member paths under a guarded root (or reject subfolders with a message). Turn an empty upload or a missing records file into "no records.jsonl/records.json found in <zip>; …". Make the provenance print and `result.json` describe the BYOD source. Add a location field (EXE2). Implement in `samples.py` and the template, regenerate, and record a BYOD run (positive and negative) in `docs/release-verification.md`.
- **Acceptance check:** (a) the documented minimum equals the smallest set that runs cells 13→23, and a smaller set is rejected in Section 4 with a message naming the split and the count required; (b) a second upload's `data_source` and records are both upload B's; (c) a zip whose records reference `photos/x.jpg` loads or is rejected with a layout message; (d) a cancelled upload prints an actionable message; (e) under BYOD, `result.json` carries no VizWiz corpus block.
- **Spec:** DAT12, DAT19, REL12 (MUST); VAL1, UX10, EXE2, OUT7.

### CAP-M5 — Major: declared `GUIDED`, but most of the guided layer and any structured learner activity are absent; infrastructure is not labelled

- **Cell/section:** whole notebook; generator `tools/build_notebook.py` (`render`) and `tools/notebook_template.py`.
- **Observed issue:** no intended-learner statement, **How to use this notebook**, roadmap, Input → Model → Output contract, glossary, prediction prompt, interpretation checkpoint, worked answer, troubleshooting section or conclusion template (P0 marker counts all 0). The objectives are mostly procedural ("install the pinned runtime; read what the carried … modules guarantee; stage and digest-verify"). The three carried module cells (818 + 258 + 807 lines) sit between Section 1 and Section 3, untitled as **Infrastructure**, not collapsed (`cellView: form` count 0), and not marked safe to skip. Present: two "Look for" notes (Sections 1 and 4), clear stage prose with consequences, a strong limits section, and three optional experiments without rerun instructions (CAP-M3).
- **Consequence:** a self-paced learner new to captioning metrics has to work out what to notice, why CIDEr-D and BLEU-4 move in opposite directions, and how to read the per-category split, and nothing checks understanding. The first screens after the install are 1,883 lines of code that look like prerequisite reading.
- **Evidence:** source inspection; P0 marker counts.
- **Recommended correction:** add the guided layer in the template, following the spec's 2.2 reference notebook (§25.13): audience and prerequisites, how-to-use, roadmap, Input → Model → Output contract, a short glossary (greedy decoding, reference caption, BLEU-4, ROUGE-L, CIDEr-D, document frequency, epoch selection), a prediction before Section 6 and Section 8, "What to notice" after each principal stage, collapsible "Check your reasoning" answers (for example: why does BLEU-4 fall while CIDEr-D rises?), one **Predict → Change one thing → Run → Observe → Explain** activity with its rerun range, troubleshooting (install, download, memory, BYOD layout) and an evidence-based conclusion template. Title the carried cells `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL14 maps to a named cell; the three carried cells are titled Infrastructure and collapsed; at least one activity asks for a prediction before a result and gives a worked answer after it.
- **Spec:** GDL1–GDL15, UX5, UX8, UX9 (SHOULD).

### CAP-m1 — Minor: doubled braces in the data contract, including the id pattern

- **Cell/section:** cell 0 (BYOD paragraph) and cell 1 (Data contract); template `tools/notebook_template.py:42`, `:130` (strings not passed through `.format`).
- **Observed issue:** `{{id, image, captions}}` and the id pattern `[A-Za-z0-9_.:-]{{1,64}}` render with doubled braces; the regex as shown is not the enforced `{1,64}`.
- **Consequence:** a user copying the contract or the pattern gets a wrong schema string and a wrong regex.
- **Evidence:** source inspection; P0 `doubled_braces`.
- **Recommended correction:** use single braces in strings that are not formatted.
- **Acceptance check:** no `{{` or `}}` in any rendered markdown cell.
- **Spec:** DAT12.

### CAP-m2 — Minor: measured runtime figures do not name their environment

- **Cell/section:** cell 0 ("On CPU the whole path takes about fifteen minutes"), cell 1 ("the build record measured about 32 s … 487 s … 879 s"), cells 16, 20.
- **Observed issue:** the figures come from a local Windows CPU pre-flight (release record) but the notebook names no machine; "about fifteen minutes" is not labelled as an estimate for a stated runtime. This review's CPU run took 527 s of cell time, the Kaggle T4 run 277 s after the restart.
- **Consequence:** a Colab CPU learner has no basis for the expectation.
- **Evidence:** source inspection; release record; P1 timings.
- **Recommended correction:** name the environment for each measured figure and label estimates.
- **Acceptance check:** every runtime figure in the notebook names an environment or says "estimate".
- **Spec:** UX12 (MUST).

### CAP-m3 — Minor: the BYOD zip handler has no expanded-size or member-count limit

- **Cell/section:** Section 4 (cell 13); template `:164-176`.
- **Observed issue:** each member is read fully into memory and written out with no total-size or count ceiling. Flattening to base names does keep writes inside `work/byod` (no traversal), which is good.
- **Consequence:** a large or hostile archive can exhaust the runtime's disk or memory before validation names a limit.
- **Evidence:** source inspection.
- **Recommended correction:** enforce an expanded-size and member-count limit before writing, with the limit stated in the Prerequisites.
- **Acceptance check:** a zip whose declared uncompressed size exceeds the stated limit is rejected before any member is written.
- **Spec:** §20 (SHOULD), VAL6.

### Suggestions

- **CAP-S1:** update the declared notebook spec from 2.0 to 2.2 (metadata, opening cell, `NOTEBOOK_SOURCE`, References).
- **CAP-S2:** report a bootstrap interval over the 70 test images for the CIDEr-D delta, so "a few hundredths" carries its own uncertainty.
- **CAP-S3:** reuse the predictions from `pipe.evaluate` for the per-category breakdown instead of captioning the test split a second time in Sections 6 and 8 (about 25 s per pass on CPU).
- **CAP-S4:** extend reload parity from 8 photographs to the whole test split (VER4).

## 5. Readiness

**Needs revision.** Remaining gates:

1. CAP-M1: a one-pass hosted `Run all` of a revised blob (uv isolated environment), and corrected release records.
2. CAP-M2 and CAP-M3: no result assertions on held-out metrics, a test split untouched by selection (or an honest statement), and experiments/BYOD that start from the base model with stated rerun ranges.
3. CAP-M4: a BYOD contract that matches enforcement, recorded with a positive and a negative BYOD run (REL12).
4. CAP-M5: the guided layer.

Then a fresh clean-runtime record of the new blob in `docs/release-verification.md`.

## 6. Verified versus inferred

- **Verified by direct execution (CPU, real weights, exact pins, not a clean runtime):** the default numbers (P1); the stale-rerun mislabel (P2); the `LEARNING_RATE = 5e-5` crash (P3); the BYOD minimum, stale second upload, subfolder failure and `StopIteration` cases (P4); a 60-record stand-in BYOD reaching export with parity (P5).
- **Verified from documented evidence:** the restart in the Kaggle qualification run of this blob.
- **Inferred from source:** the BYOD "re-run from that cell" stale baseline; basename collisions; the zip size risk; Colab behaving like Kaggle at the install.
- **Not verified:** Colab, CUDA in this review, real-photograph BYOD, learner understanding.
- **Most likely to be wrong:** CAP-M2's part (a) as a defect rather than disclosed practice: the notebook and card are open that the default "gained on the held-out split", and a reader could treat that as acceptable for a tutorial. It stays Major because cell 20 tells the learner the test split was used for nothing but the final evaluation, and the assertion turns that choice into a gate that crashes the notebook's own first experiment.

Probes: `blip_captioning_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
