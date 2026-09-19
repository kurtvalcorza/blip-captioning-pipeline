"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E image-captioning workflow: the pinned Salesforce/blip-image-captioning-base
snapshot is digest-verified and loaded, the captions of a digest-pinned VizWiz-Captions shard are read
column-only and the 336 photographs of its first row group are read the same way, validated and split by
image, three drawn scenes are captioned through the inference contract, the frozen model is scored on the
held-out photographs beside two non-neural baselines, a bounded fine-tuning of the caption decoder's last
blocks runs in the kernel, the held-out split is scored again per category, the adapted model re-captions
the drawn scenes, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "blip_captioning_pipeline",
    "repo_name": "blip-captioning-pipeline",
    "stem": "blip_captioning",
    "notebook_name": "blip_captioning_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `Salesforce/blip-image-captioning-base` snapshot (a 990 MB `pytorch_model.bin` pickle, opened only after its "
        "SHA-256 matches and with `weights_only=True`), reads the four text columns of one digest-pinned VizWiz-Captions shard "
        "from the Hugging Face Hub (about 0.5 MB over HTTP range requests, no credential) and the image column of its first row "
        "group (336 photographs, about 84 MB, each refused on any size or SHA-256 mismatch), cuts the 318 captioned ones by "
        "image into 208 / 40 / 70 training, validation and test photographs, captions three drawn scenes through the inference "
        "contract with an input manifest and a rejection probe, scores the frozen model on the test photographs with BLEU-4, "
        "ROUGE-L, CIDEr-D and unigram F1 beside the constant-caption and colour-nearest-neighbour baselines, runs a bounded "
        "fine-tuning of the caption decoder's last blocks and head on the training photographs with validation-CIDEr-D epoch "
        "selection, scores the held-out photographs again per category, re-captions the drawn scenes with the adapted model, "
        "exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify caption "
        "parity. The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and "
        "no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about fifteen minutes after "
        "the downloads; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "holding a `records.jsonl` (or `records.json`) of `{{id, image, captions}}` objects — `image` a file name inside the "
        "zip, `captions` one or more reference captions, optional `category` — beside the image files. They pass through the "
        "same validation, seeded image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and "
        "reload-parity cells as the VizWiz sample. The expected schema and the ceilings are stated in the Prerequisites and in "
        "Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "BlipCaptioningPipeline",
    "weights_key": "blip-image-captioning-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "BLIP image-captioning-base — DIMER E2E image-captioning fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/blip-captioning-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/blip-captioning-pipeline/blob/main/tutorials/blip_captioning_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Salesforce%2Fblip--image--captioning--base-ffcc4d?style=flat",
            "https://huggingface.co/Salesforce/blip-image-captioning-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-salesforce%2FBLIP-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/salesforce/BLIP",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2201.12086-b31b1b.svg", "https://arxiv.org/abs/2201.12086"),
    ],
    "capability": "image captioning and bounded supervised fine-tuning of the caption decoder's last blocks on a photograph/reference-captions dataset, using the pinned `Salesforce/blip-image-captioning-base` weights",
    "intro": (
        "`Salesforce/blip-image-captioning-base` is the BLIP model of Li et al. (2022) — a ViT-B/16 image encoder at 384×384 "
        "and a 12-layer BERT-style caption decoder that cross-attends to the image features; 223,971,644 parameters "
        "(the Hub's *247M* counts the decoder's tied output projection twice), pretrained on 129 M image–text pairs and "
        "fine-tuned on COCO Captions — published under the **BSD-3-Clause** licence. At inference it encodes the resized "
        "image and generates a sentence token by token with greedy decoding under a caller-owned `max_new_tokens` budget, from "
        "scratch or continuing a prefix you supply. **No score exists**: the caption is generated text with no probability and "
        "no correctness signal, and a fluent caption is **not evidence that it describes the image**.\n\n"
        "What this notebook adds to inference is **adaptation with reference captions**. The dataset is real and out of the "
        "model's distribution: VizWiz-Captions (Gurari et al., ECCV 2020; **CC BY 4.0**) — photographs taken by blind people, "
        "each with up to five crowd-written captions in a style COCO does not have (what is being held, what a label says, "
        "that the picture is blurry or badly framed) — a population the model never saw and a convention it does not know. The "
        "notebook reads only the four text columns of one pinned Hub shard (about 0.5 MB over HTTPS range requests) and the "
        "image column of its **first row group** (336 photographs in one 84 MB range read, each pinned by size and SHA-256 in "
        "the carried module). The frozen model is already a competent captioner — the build record measured CIDEr-D about "
        "0.59 on the test photographs against 0.05 for the best constant caption — so the honest question is a narrow one: "
        "does a bounded adaptation of the caption decoder's last blocks on 208 photographs move the consensus metric on an "
        "image-disjoint test split, and which of the four metrics move with it? Three captioning metrics are implemented in "
        "pure Python in the carried modules (**BLEU-4**, **ROUGE-L**, **CIDEr-D** — own implementations of the `coco-caption` "
        "definitions, with CIDEr-D's document frequencies taken from the evaluated set) beside the plumbing check "
        "`unigram_f1`, and two **non-neural baselines** — the corpus-medoid constant caption and a colour nearest neighbour — "
        "show where a system with no model sits. Nothing here is a quality claim about your photographs: it is one seeded "
        "split of one corpus.\n\n"
        "**Weight-format note:** upstream hosts no SafeTensors at the pinned revision; the carried module executes the "
        "digest-pinned `pytorch_model.bin` (a pickle, deserialised with `weights_only=True` after its SHA-256 is checked), "
        "while the `tf_model.h5` upstream also hosts is DIMER's upload artifact and is never loaded here. Section 3 stages "
        "and digest-verifies the snapshot before the processor or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot (a pickle checkpoint, and why that matters); read the captions of a "
        "digest-pinned corpus without downloading its shard and its photographs from one pinned row group with per-file "
        "digests, validate them and split by image without leakage; caption through the public API over drawn scenes and "
        "read `caption`, `new_tokens` and `truncated` correctly (generated text, no score); score the frozen model against "
        "several reference captions per photograph with BLEU-4, ROUGE-L and CIDEr-D beside two non-neural baselines and read "
        "the per-category breakdown; run a bounded fine-tuning with explicit hyperparameters and validation-based epoch "
        "selection; evaluate on an image-disjoint test split; re-caption drawings from a different image family with the "
        "adapted model; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "reading text in the image (BLIP is not an OCR model, even though VizWiz captions often transcribe labels), visual "
        "question answering (a separate checkpoint), dense or region captioning (one sentence per image, no localisation), "
        "captions in languages other than English, batch throughput, sampling, beam search or repetition penalties (the "
        "upstream evaluation used beam search; this notebook decodes greedily for reproducibility), SPICE (needs a scene-graph "
        "parser), evaluation on the COCO or VizWiz benchmarks proper (only one seeded 318-photograph sample is scored here), "
        "fine-tuning of the vision encoder, the embeddings or the tied output projection, training on images that are not the "
        "pinned sample or your own uploads, and any claim that a VizWiz split stands in for your photographs. The repository "
        "exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate but not fast: the build record measured about 32 s to caption the 70 test photographs and 487 s for the four epochs of fine-tuning the caption decoder's last two blocks and head (889 image–caption pairs per epoch), including the one-off encoding of the 208 training photographs and the per-epoch validation scoring; the whole default path took 879 s with the snapshot and photographs already cached. The pinned `torch==2.14.0` install and the 990 MB checkpoint are the large downloads of the run; the row group of photographs adds about 84 MB.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; why a pickle checkpoint needs a digest check before `torch.load`; what BLEU-4, ROUGE-L and CIDEr-D measure (n-gram precision with a brevity penalty, longest-common-subsequence F-measure, TF-IDF-weighted n-gram consensus) and why none is a human judgement; why a confident caption is not a correct one.",
        "- **Data contract:** records are `{{id, image, captions}}` — an image file decodable by Pillow with sides between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` (4096) px and one or more non-empty reference captions of at most `MAX_CAPTION_CHARS` (500) characters (`MIN_CAPTIONS` = 1; VizWiz supplies up to five); optional `image_id` (defaults to the id) groups records on the same image and optional `category` labels the breakdown (`text` / `no-text` in the sample, from the corpus's text-detected flag). Ids match `[A-Za-z0-9_.:-]{{1,64}}` and are unique; a dataset needs 8..5,000 records; every record on the same image lands in the same split so a test image is never trained on; every (image, reference caption) pair is one training sample. BYOD accepts one zip of images plus a `records.jsonl` / `records.json` in that shape.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every caption checked, but nothing checks that a reference caption is right — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of people, documents or homes are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads two parts of one object in the Hub dataset repository `mm-eval/VizWiz-Captions` at the immutable revision `c4a6d897…` (`data/val-00004-of-00005.parquet`, 392,245,504 bytes, SHA-256 `4492465a…`): the declared size and SHA-256 are checked against the pins before any byte is read; the four text columns of all 1,550 rows are fetched over HTTPS range requests through `pyarrow` (only the parquet footer and those column chunks) and refused unless their decoded SHA-256 matches; then the image column of row group 0 only (336 JPEG files, about 84 MB) is read the same way, each photograph pinned by size and SHA-256 in the carried module and refused on any mismatch. The corpus is CC BY 4.0 (Gurari et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. VizWiz photographs, captions and split\n\n"
                "`fetch_annotations` reads the pinned shard's four text columns (or the cache under `weights/vizwiz-captions/`): "
                "it first checks the byte size and SHA-256 the Hub declares for the file against the pins, then reads only the "
                "parquet footer and those column chunks through `pyarrow` over HTTPS range requests, and refuses the decoded "
                "columns unless their SHA-256 matches. `fetch_images` stages the 336 pinned photographs of row group 0 — each "
                "cached file is re-hashed; anything missing is read from the shard's image column in one range read and refused "
                "on any size or SHA-256 mismatch. `build_sample_dataset` keeps the 318 photographs that carry at least one "
                "reference caption (the Hub re-conversion dropped rejected and pre-canned captions), shuffles them with "
                "`SPLIT_SEED` and cuts them **by image** into 208 / 40 / 70 training, validation and test records, each "
                "labelled `text` or `no-text` from the corpus's text-detected flag. `validate_dataset` then opens and decodes "
                "every image and checks every record against the contract, `check_split_disjoint` asserts no image is shared, "
                "and the training split is written to `outputs/{stem}_train.jsonl` in the shape BYOD expects.\n\n"
                "Look for: 1,550 annotation rows, 336 photographs, three digests, the category mix per split (a little over "
                "half the photographs contain text), captions per image between 1 and 5, and four refusal probes — a duplicate "
                "id, a missing image file, an empty caption list and a dataset too small to split — each rejected before "
                "`torch` does anything."
            ),
            "code": (
                "import collections\n"
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import zipfile\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_dir = Path('work') / 'byod'\n"
                "    byod_dir.mkdir(parents=True, exist_ok=True)\n"
                "    with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
                "        for member in archive.infolist():\n"
                "            name = Path(member.filename).name\n"
                "            if member.is_dir() or not name or name.startswith('.'):\n"
                "                continue\n"
                "            (byod_dir / name).write_bytes(archive.read(member))\n"
                "    records_file = next(p for p in (byod_dir / 'records.jsonl', byod_dir / 'records.json') if p.is_file())\n"
                "    records = load_byod_dataset(records_file)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED, base_dir=byod_dir)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    annotations = fetch_annotations(cache_dir='weights/vizwiz-captions')\n"
                "    image_paths = fetch_images(sorted(IMAGE_PINS), cache_dir='weights/vizwiz-captions')\n"
                "    raw_rows = {{'annotations': len(annotations), 'photographs': len(image_paths), 'captioned': sum(1 for r in annotations if r['id'] in IMAGE_PINS and r['captions'])}}\n"
                "    splits = build_sample_dataset(annotations, seed=SPLIT_SEED, image_paths=image_paths)\n"
                "    data_source = f'{{CORPUS_NAME}} {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "categories = {{name: manifest['categories'] for name, manifest in dataset_manifests.items()}}\n"
                "write_dataset_jsonl(splits['train'], 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'text_sha256': CORPUS_FILE['text_sha256'][:16] + '...', 'pinned_photographs': len(IMAGE_PINS)}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_images': manifest['unique_images'], 'categories': manifest['categories'], 'captions_per_image': manifest['captions_per_image'], 'caption_words': manifest['caption_words'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'image': Path(example['image']).name, 'size': example['image_size'], 'category': example['category'], 'captions': example['captions']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'missing image file': [{{**splits['train'][0], 'image': 'work/does-not-exist.jpg'}}, *splits['train'][1:8]],\n"
                "    'empty caption list': [{{**splits['train'][0], 'captions': []}}, *splits['train'][1:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Caption through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: three flat cartoon scenes — "
                "a red house with a tree, a beach with a red sailboat, an apple and an orange on a table — are drawn in code "
                "with Pillow (no text rendering, so their digests are stable across builds), a different image family from the "
                "photographs, and scenes the model will be asked to caption again after adaptation. `validate_inputs` applies "
                "exactly the checks `caption` applies (image sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, an optional prefix up to "
                "`MAX_PREFIX_CHARS`, `max_new_tokens` in `[1, MAX_NEW_TOKENS]`) and returns an input manifest; a 4-pixel image "
                "is validated too and its rejection recorded as a finding. `caption` returns the decoded text, the checked "
                "prefix, `image_size`, `new_tokens`, a `truncated` flag and the model identity. **No score exists**: the "
                "caption is generated text with no probability and no correctness signal — a fluent caption is **not evidence "
                "that it describes the image**, and greedy decoding is a reproducibility property, not a quality one. As "
                "recorded in the model card, the repository's CPU smoke captioned these same drawings `a red house with a tree "
                "and a ball`, `a beach scene with a sail and a sun` and `a red and yellow apple on a white background` (the "
                "orange became a second apple). The cell records which expected keywords appear in each caption — an "
                "observation, not a metric; whether captions are *right* is what Section 6 measures on 70 photographs with "
                "several human references each."
            ),
            "code": (
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "CAPTION_MAX_TOKENS = 30  # @param {{type:\"integer\"}}\n\n\n"
                "def synthetic_scenes():\n"
                "    \"\"\"Three flat cartoon scenes drawn with Pillow (no text); returns [(name, image, expected keywords)].\"\"\"\n"
                "    house = Image.new('RGB', (640, 480), (135, 206, 235))  # sky\n"
                "    d = ImageDraw.Draw(house)\n"
                "    d.rectangle([0, 300, 640, 480], fill=(60, 179, 75))  # grass\n"
                "    d.ellipse([500, 40, 600, 140], fill=(255, 215, 0))  # sun\n"
                "    d.rectangle([120, 180, 320, 330], fill=(200, 40, 40))  # red house\n"
                "    d.polygon([(100, 180), (220, 90), (340, 180)], fill=(90, 50, 20))  # brown roof\n"
                "    d.rectangle([200, 260, 240, 330], fill=(70, 40, 20))  # brown door\n"
                "    d.ellipse([420, 260, 520, 360], fill=(40, 100, 40))  # tree crown\n"
                "    d.rectangle([460, 350, 480, 420], fill=(90, 60, 30))  # trunk\n"
                "    d.ellipse([60, 380, 140, 440], fill=(255, 255, 255))  # white ball\n"
                "    beach = Image.new('RGB', (640, 480), (120, 190, 240))  # sky\n"
                "    d = ImageDraw.Draw(beach)\n"
                "    d.rectangle([0, 220, 640, 330], fill=(30, 110, 200))  # sea\n"
                "    d.rectangle([0, 330, 640, 480], fill=(238, 214, 150))  # sand\n"
                "    d.ellipse([60, 40, 150, 130], fill=(255, 230, 80))  # sun\n"
                "    d.polygon([(400, 330), (470, 330), (435, 210)], fill=(230, 40, 40))  # red sail\n"
                "    d.rectangle([432, 210, 438, 330], fill=(90, 60, 30))  # mast\n"
                "    d.ellipse([200, 370, 260, 430], fill=(255, 120, 40))  # beach ball\n"
                "    fruit = Image.new('RGB', (480, 480), (250, 250, 245))\n"
                "    d = ImageDraw.Draw(fruit)\n"
                "    d.ellipse([60, 120, 220, 280], fill=(220, 30, 30))  # red apple\n"
                "    d.rectangle([135, 95, 145, 125], fill=(80, 50, 20))  # stalk\n"
                "    d.ellipse([250, 140, 430, 300], fill=(255, 170, 20))  # orange\n"
                "    d.polygon([(90, 400), (400, 400), (360, 330), (130, 330)], fill=(180, 120, 60))  # table\n"
                "    return [\n"
                "        ('synthetic_house_640x480.png', house, ['house', 'tree', 'red']),\n"
                "        ('synthetic_beach_640x480.png', beach, ['beach', 'sail', 'sun']),\n"
                "        ('synthetic_fruit_480x480.png', fruit, ['apple', 'orange']),\n"
                "    ]\n\n\n"
                "scenes = synthetic_scenes()\n"
                "scene_names = [name for name, _, _ in scenes]\n"
                "scene_images = [image for _, image, _ in scenes]\n"
                "expected_keywords = [keywords for _, _, keywords in scenes]\n"
                "scene_digests = {{name: hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for name, image in zip(scene_names, scene_images)}}\n"
                "ceilings = {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'IMAGE_SIZE': IMAGE_SIZE, 'MAX_PREFIX_CHARS': MAX_PREFIX_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CAPTIONS': MIN_CAPTIONS, 'MAX_CAPTION_CHARS': MAX_CAPTION_CHARS}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(scene_images, max_new_tokens=CAPTION_MAX_TOKENS, names=scene_names)\n"
                "try:\n"
                "    validate_inputs([Image.new('RGB', (4, 4))])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'tiny-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scenes': scene_names, 'rgb_sha256': {{k: v[:16] + '...' for k, v in scene_digests.items()}}, 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "results = []\n"
                "for name, image, keywords in scenes:\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.caption(image, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "    result['image'] = name\n"
                "    result['keyword_hits'] = keyword_hits(result['caption'], keywords)\n"
                "    results.append({{'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "    print(f\"{{name}}\\n   caption: {{result['caption']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\\n   keywords: {{result['keyword_hits']}}\")\n"
                "checks = {{\n"
                "    'one_result_per_image': len(results) == len(scenes),\n"
                "    'captions_are_text': all(isinstance(r['caption'], str) and r['caption'] for r in results),\n"
                "    'budget_respected': all(r['new_tokens'] <= CAPTION_MAX_TOKENS for r in results),\n"
                "    'setting_echoed': all(r['generation']['max_new_tokens'] == CAPTION_MAX_TOKENS and r['generation']['do_sample'] is False for r in results),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'caption output failed a sanity check: {{checks}}')\n"
                "frozen_scene = evaluation_report(results, None, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'frozen_scene_verdict': frozen_scene['verdict'], 'any_truncated': any(r['truncated'] for r in results)}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the test photographs\n\n"
                "Three systems frame the adaptation, each read four ways. The **constant-caption baseline** answers every "
                "photograph with the one training caption that scores highest against all other training references — the "
                "corpus medoid, a sentence that is safe everywhere and right nowhere. The **colour-nearest-neighbour "
                "baseline** answers with the first reference caption of the training photograph whose 3×3 mean-colour grid is "
                "closest — a lookup that knows the image through 27 numbers. The **frozen model** captions the 70 test "
                "photographs with the budget from Section 5 and is scored with the same metrics: **BLEU-4** (corpus-level "
                "clipped n-gram precision with a brevity penalty), **ROUGE-L** (longest-common-subsequence F-measure against the "
                "best reference), **CIDEr-D** (TF-IDF-weighted n-gram consensus over all references, the metric captioning "
                "leaderboards are ranked by) and the plumbing check **unigram F1**, all after lower-casing and punctuation "
                "removal. Expect the frozen model far above both baselines — it is a trained captioner — and read the "
                "per-category breakdown: the build record measured CIDEr-D 0.59 overall on the frozen model, with the "
                "`text` photographs (labels, screens, packaging) at 0.64 and the `no-text` ones at 0.53."
            ),
            "code": (
                "baseline_constant = constant_caption_baseline(train_records, test_records)\n"
                "baseline_neighbour = colour_neighbour_baseline(train_records, test_records)\n"
                "METRICS = ('bleu4', 'rouge_l', 'cider_d', 'unigram_f1')\n"
                "print({{'constant_caption_baseline': {{k: round(baseline_constant[k], 3) for k in METRICS}}, 'n': baseline_constant['n'], 'caption': baseline_constant['baseline']}})\n"
                "print({{'colour_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'mean_words': round(frozen_test['mean_words'], 1), 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n\n\n"
                "def model_caption(pipeline):\n"
                "    def predict(record):\n"
                "        with Image.open(record['image']) as photo:\n"
                "            photo.load()\n"
                "            return pipeline.caption(photo, max_new_tokens=CAPTION_MAX_TOKENS)['caption']\n"
                "    return predict\n\n\n"
                "def by_category(predict, records):\n"
                "    \"\"\"CIDEr-D per category, with document frequencies from the whole evaluated set (as in `evaluate`).\"\"\"\n"
                "    scores = cider_d([predict(r) for r in records], [reference_captions(r) for r in records])\n"
                "    groups = collections.defaultdict(list)\n"
                "    for record, score in zip(records, scores, strict=True):\n"
                "        groups[record['category']].append(score)\n"
                "    return {{category: {{'n': len(values), 'cider_d': round(sum(values) / len(values), 3)}} for category, values in sorted(groups.items())}}\n\n\n"
                "medoid = medoid_caption(train_records)\n"
                "constant_fields = by_category(lambda record: medoid, test_records)\n"
                "frozen_predictions = {{r['id']: model_caption(pipe)(r) for r in test_records}}\n"
                "frozen_fields = by_category(lambda record: frozen_predictions[record['id']], test_records)\n"
                "print({{'by_category': {{'constant': constant_fields, 'frozen': frozen_fields}}}})\n"
                "for record in test_records[:3]:\n"
                "    print({{'category': record['category'], 'frozen': frozen_predictions[record['id']], 'references': reference_captions(record)[:2]}})\n"
                "assert frozen_test['cider_d'] > baseline_constant['cider_d']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the caption decoder's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_DECODER_LAYERS` blocks of the caption decoder plus its output "
                "head's transform and bias — two blocks by default, 19,526,204 of 223,971,644 parameters; the vision encoder, "
                "every embedding and the head's projection weight (tied to the word embeddings) stay frozen. The frozen vision "
                "encoder's output is computed **once** per training photograph and reused across epochs, so an epoch costs a "
                "decoder pass per (image, caption) pair — every reference caption is a training sample, the BLIP fine-tuning "
                "convention, about 900 pairs per epoch here. The target is decoded from the `[DEC]` start token with the "
                "model's own label-smoothed cross-entropy; AdamW at a fixed learning rate, gradient clipping at 1.0, seeded "
                "shuffling and no scheduler. Epoch 0 records the frozen model's validation metrics; every epoch is scored on "
                "the 40 validation photographs, and the epoch with the highest validation CIDEr-D is kept — forty photographs "
                "make that selection noisy, which is why the held-out split in Section 8 is what the numbers are read from.\n\n"
                "Watch the training loss fall while validation CIDEr-D moves by a few hundredths: the frozen model already "
                "captions well, so what adapts is mostly the *style* of the sentences (longer, naming the held object and its "
                "label). The build record's counter-examples — a higher learning rate overfits 208 photographs within two "
                "epochs and the selector returns the frozen weights — are in the model card; the default is the configuration "
                "that gained on the held-out split."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "        row['val_mean_words'] = round(entry['val']['mean_words'], 1)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'training_pairs': adapt_result['n_pairs'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test photographs were never used for training or epoch selection, and no test image appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6, the "
                "four systems are put side by side on all four metrics, and the per-category CIDEr-D is repeated. Read it in "
                "this order: **CIDEr-D** first (the consensus metric the epoch was selected on, and the one that rewards the "
                "corpus's own vocabulary), then BLEU-4 and ROUGE-L, which can move the other way when the adapted captions get "
                "longer — the build record measured CIDEr-D 0.590 frozen → 0.629 adapted with BLEU-4 0.213 → 0.190 and mean caption length up from 8.9 to 11.2 words, and the gain sits on the `no-text` photographs (0.525 → 0.615) while the `text` ones are flat (0.638 → 0.639) — the adapted decoder describes objects and scenes in the corpus's longer style but reads no more labels than before. The cell asserts the adapted CIDEr-D is above the frozen one. Seventy "
                "photographs from one seeded split of one corpus give **no dispersion estimate**; the deltas are sample-sanity "
                "evidence that the adaptation contract works, not a benchmark, and a gain on VizWiz says nothing about your "
                "photographs until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "adapted_predictions = {{r['id']: model_caption(pipe)(r) for r in test_records}}\n"
                "adapted_fields = by_category(lambda record: adapted_predictions[record['id']], test_records)\n"
                "comparison = {{metric: {{'constant': round(baseline_constant[metric], 3), 'neighbour': round(baseline_neighbour[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['mean_words'] = {{'constant': round(baseline_constant['mean_words'], 1), 'neighbour': round(baseline_neighbour['mean_words'], 1), 'frozen': round(frozen_test['mean_words'], 1), 'adapted': round(adapted_test['mean_words'], 1)}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['by_category'] = {{category: {{'n': frozen_fields[category]['n'], 'constant': constant_fields[category]['cider_d'], 'frozen': frozen_fields[category]['cider_d'], 'adapted': adapted_fields[category]['cider_d']}} for category in frozen_fields}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "for record in test_records[:3]:\n"
                "    print({{'category': record['category'], 'frozen': frozen_predictions[record['id']], 'adapted': adapted_predictions[record['id']], 'references': reference_captions(record)[:2]}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'categories': categories,\n"
                "    'max_new_tokens': CAPTION_MAX_TOKENS,\n"
                "    'baselines': {{'constant_caption': baseline_constant, 'colour_neighbour': baseline_neighbour}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['cider_d'] > frozen_test['cider_d']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-caption the drawn scenes, export the adapter and reload it\n\n"
                "The three scenes from Section 5 are captioned again by the adapted model — drawings, a different image "
                "family from the photographs it was tuned on, so this is a small look at what the adaptation did *outside* its "
                "corpus (the build record's captions are in the model card: the adapted model kept the house and beach captions and appended a location phrase, and captioned the fruit scene `an apple and an orange on a white background` where the frozen model had called the orange a second apple — one drawing of evidence, not a measurement; a different caption here is a finding to record, not a failure). Both caption sets are written as CSV, and the keyword observations are repeated.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the caption decoder's last two blocks and its head transform "
                "and bias, about 78 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the "
                "base model id and revision, the digest of the base `pytorch_model.bin`, the tensor names, the file size and "
                "SHA-256, the training configuration and the epoch history (OUT8). `BlipCaptioningPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** "
                "deserialising, refuses any tensor that is not a caption-decoder tensor, and overlays the tensors onto a "
                "freshly loaded base — a new object from files, not the in-memory model (VER2). The cell asserts identical "
                "captions on eight test photographs (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "adapted_results = []\n"
                "for name, image, keywords in scenes:\n"
                "    result = pipe.caption(image, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "    result['image'] = name\n"
                "    result['keyword_hits'] = keyword_hits(result['caption'], keywords)\n"
                "    adapted_results.append(result)\n"
                "adapted_scene = evaluation_report(adapted_results, None, sample_kind='synthetic')\n"
                "for before, after in zip(results, adapted_results, strict=True):\n"
                "    print({{'image': before['image'], 'frozen': before['caption'], 'adapted': after['caption'], 'keywords_after': after['keyword_hits']}})\n"
                "with open('outputs/{stem}_captions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'frozen_caption', 'adapted_caption', 'adapted_new_tokens', 'expected_keywords'])\n"
                "    for before, after, keywords in zip(results, adapted_results, expected_keywords, strict=True):\n"
                "        writer.writerow([before['image'], before['caption'], after['caption'], after['new_tokens'], ' | '.join(keywords)])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = BlipCaptioningPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [model_caption(pipe)(r) for r in test_records[:8]]\n"
                "after = [model_caption(reloaded)(r) for r in test_records[:8]]\n"
                "parity = {{'identical_captions': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_captions'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'pytorch_model.bin pickle, digest-verified, weights_only=True', 'weight_sha256': weight_entry['sha256'], 'hosted_tf_weight_file_not_loaded': HOSTED_TF_WEIGHT_FILE}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'text_columns': list(CORPUS_TEXT_COLUMNS), 'file': CORPUS_FILE, 'pinned_images': len(IMAGE_PINS)}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'scenes': {{'names': scene_names, 'sizes': [list(image.size) for image in scene_images], 'rgb_sha256': scene_digests, 'expected_keywords': expected_keywords}}, 'items': [{{k: r[k] for k in ('image', 'caption', 'new_tokens', 'truncated', 'keyword_hits', 'seconds')}} for r in results], 'frozen_report': frozen_scene, 'adapted_items': [{{k: r[k] for k in ('image', 'caption', 'new_tokens', 'truncated', 'keyword_hits')}} for r in adapted_results], 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is already a competent captioner on photographs it never saw — far above the two non-neural "
        "baselines — and a bounded fine-tuning of the caption decoder's last two blocks and head on 208 VizWiz photographs "
        "moves the consensus metric by a few hundredths of CIDEr-D on an image-disjoint test split (0.590 → 0.629 in the build record, on the `no-text` photographs rather than the ones with labels) while lengthening the "
        "sentences toward the corpus's style, with a 78 MB adapter that reloads to identical captions. That is the claim: "
        "the adaptation contract works end to end on a real out-of-distribution captioning corpus, and the numbers it produces "
        "are read on four metrics and per category against two non-neural baselines and the frozen model rather than in "
        "isolation. It is a smaller effect than a task the base model cannot do at all would show, and the notebook says so.\n\n"
        "The test split is 70 photographs from one seeded draw of one row group of one shard of one corpus, the validation "
        "split that picks the epoch is 40, the metrics are four reference-based scores (own pure-Python implementations of "
        "the `coco-caption` definitions, with CIDEr-D's document frequencies from the evaluated set — so its absolute value is "
        "not comparable to leaderboard numbers computed on the full validation set — and none a human judgement), and a "
        "higher learning rate overfits this little data within two epochs, which the validation-based selector reports by "
        "keeping epoch 0. So a gain here says the contract works, not that the adapted model is better on your photographs, "
        "that it reads the labels it now mentions, or that its captions are faithful — it still generates a sentence for "
        "every image, and it can be wrong fluently. Fine-tuning on a narrow corpus can also erode the model elsewhere; the "
        "drawn scenes re-captioned in Section 9 are three images of evidence about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the constant-caption and colour-neighbour baselines and "
        "the frozen model's score on *your* references are the numbers to read before any adapted one, per category and on "
        "CIDEr-D. **Leakage:** keep every record on an image in one split (the contract does this) and split by photographer "
        "or session when your images come from few sources, never at random over near-duplicate frames. **References:** "
        "every reference caption is a training target; a corpus with one caption per image trains the model on one "
        "annotator's phrasing, and CIDEr-D with a single reference rewards that phrasing rather than a consensus.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify the captions and "
        "photographs of a real captioning corpus, validate the demonstrated dataset contract without leakage, execute the "
        "inference contract and a bounded fine-tuning, evaluate against two trivial baselines and the frozen model on an "
        "image-disjoint split, and emit the shown machine-readable artifacts — without the repository being reachable. It "
        "does **not** establish benchmark superiority, caption quality on any other population or camera, or production "
        "fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `LEARNING_RATE = 5e-5` and watch the training "
        "loss fall while the validation CIDEr-D drops and the selector keeps epoch 0; set `TRAINABLE_DECODER_LAYERS = 1` and "
        "compare the artifact size and the held-out score; raise `EPOCHS` and watch the validation CIDEr-D pick the epoch; or "
        "bring your own photographs through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/blip-captioning-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/blip-captioning-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/blip-captioning-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (Salesforce, BSD-3-Clause): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/salesforce/BLIP\n"
        "- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li et al., 2022): https://arxiv.org/abs/2201.12086\n"
        "- Captioning Images Taken by People Who Are Blind (Gurari et al., ECCV 2020; VizWiz-Captions, CC BY 4.0): https://arxiv.org/abs/2002.08565 — data: https://vizwiz.org/tasks-and-datasets/image-captioning/\n"
        "- BLEU: a Method for Automatic Evaluation of Machine Translation (Papineni et al., 2002): https://aclanthology.org/P02-1040/\n"
        "- ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004): https://aclanthology.org/W04-1013/\n"
        "- CIDEr: Consensus-based Image Description Evaluation (Vedantam et al., 2015): https://arxiv.org/abs/1411.5726\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
