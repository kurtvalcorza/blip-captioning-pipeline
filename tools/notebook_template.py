"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "blip_captioning_pipeline",
    "repo_name": "blip-captioning-pipeline",
    "stem": "blip_captioning",
    "notebook_name": "blip_captioning_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "BlipCaptioningPipeline",
    "weights_key": "blip-image-captioning-base",
    "runtime_imports": ["torch", "transformers"],
    "title": "BLIP image-captioning-base — DIMER image captioning tutorial (standalone)",
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
    "capability": "Image captioning — one image (optionally with a caption prefix to continue) → one sentence of generated text — using the pinned `Salesforce/blip-image-captioning-base` weights",
    "intro": (
        "At inference the BLIP model (a ViT-B/16 image encoder at 384×384 and a 12-layer BERT-style text decoder that "
        "cross-attends to the image features; about 247M parameters, pretrained on 129M image–text pairs with "
        "captioning-and-filtering bootstrapping and fine-tuned on COCO Captions) encodes the resized image and generates a "
        "caption token by token, either from scratch (unconditional) or continuing a prefix you supply such as "
        "`a photography of` (conditional, the upstream README's example). Decoding is greedy (`do_sample=False`, one beam) "
        "under a caller-owned `max_new_tokens` budget. **No adaptation occurs:** no training, fine-tuning, in-context "
        "conditioning, or preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights, "
        "processor and tokenizer, and the carried module adds snapshot verification, the input contract (image side "
        "ceilings, an optional prefix up to 128 characters, the token budget), a fixed output contract, and the "
        "`keyword_hits`, `unigram_f1`, `validate_inputs` and `evaluation_report` helpers. **Weight-format note:** upstream "
        "hosts no SafeTensors at the pinned revision; the carried module executes the digest-pinned `pytorch_model.bin` "
        "(a pickle, deserialised with `weights_only=True` after its SHA-256 is checked), while the `tf_model.h5` upstream "
        "also hosts is DIMER's upload artifact and is never loaded here. The default sample is three flat cartoon scenes "
        "drawn in code with no reference captions, so the evaluation report is `not-measurable` by design and the "
        "printed keyword checks are observations, not a captioning benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision (a pickle checkpoint, and why that matters), draw three synthetic scenes (or "
        "upload your own photographs) and validate them into an input manifest, choose a token budget and an optional "
        "prefix, run the supported task, read the captions correctly (generated text, no score, a `truncated` flag), "
        "exercise an optional BYOD path, produce an evaluation report that is `not-measurable` without reference captions "
        "and `sample-sanity` with a bag-of-words `unigram_f1` when you supply some, and export the captions, an annotated "
        "contact sheet and provenance."
    ),
    "exclusions": (
        "Reading text in the image (BLIP is not an OCR model), visual question answering (a separate checkpoint), dense "
        "or region captioning (one sentence per image, no localisation), captions in languages other than English, "
        "batch throughput, sampling, beam search or repetition penalties (the upstream evaluation used beam search; this "
        "notebook decodes greedily for reproducibility), evaluation on COCO Captions (not bundled; CIDEr, BLEU-4 and SPICE "
        "need several references per image and are not computed here), and any training. The model was fine-tuned on "
        "photographs; drawings, diagrams, documents and expert imagery are outside what this notebook measures, and a "
        "fluent wrong caption carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 3.9 s to load and 0.2–0.5 s per caption on 640×480 drawn scenes in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 990 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; why a pickle checkpoint needs a digest check before `torch.load`; what reference-based caption metrics (CIDEr, BLEU) need; that a confident caption is not a correct one.",
        "- **Data:** the default sample is three deterministic cartoon scenes drawn in code with Pillow (a house with a tree and the sun; a beach with a sailboat; two fruits on a table — no text rendering, so their digests are stable across Pillow builds) with **no reference captions**, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one or more images decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scenes or optional BYOD\n\n"
                "The default sample is **synthetic**: three flat cartoon scenes — a red house with a brown roof and door, a "
                "tree, a white ball and the sun on grass under a blue sky; a beach with sea, sand, a red sailboat and the sun; "
                "a red apple and an orange on a wooden table — are drawn with Pillow (640×480, 640×480, 480×480), the same "
                "drawings the repository's smoke run used. **No reference captions are authored**, because a caption you write "
                "yourself is not an annotation standard: the evaluation report will therefore be `not-measurable`, and the "
                "notebook instead records which of a few expected keywords (`house`, `tree`, `beach`, `apple`, …) appear in "
                "each caption as an observation. The image digests are printed for the record. BYOD is optional and disabled "
                "by default; when enabled, upload one or more images.\n\n"
                "Two **caller-owned request parameters** are exposed: `max_new_tokens` bounds the caption "
                "(`DEFAULT_MAX_NEW_TOKENS = 30` fits any COCO-style sentence; `MAX_NEW_TOKENS = 64` is the ceiling), and "
                "`caption_prefix` (empty for unconditional captioning; `a photography of` is the upstream README's example — "
                "the model continues whatever you start, so the prefix shapes the caption). Nothing is validated in this cell "
                "— the next section hands the images to the pipeline's own validation stage, which is the only checker. Look "
                "for one dictionary per image naming the sample kind, size and digest, plus the budget and prefix."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "max_new_tokens = 30  # @param {{type:\"integer\"}}\n"
                "caption_prefix = ''  # @param {{type:\"string\"}}\n\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    samples = []\n"
                "    for name, data in uploaded.items():\n"
                "        image = Image.open(io.BytesIO(data))\n"
                "        image.load()\n"
                "        samples.append((name, image, []))\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawings: no randomness and no text rendering, so no seed is needed and the digests are stable.\n"
                "    samples = synthetic_scenes()\n"
                "    sample_kind = 'synthetic'\n\n"
                "names = [name for name, _, _ in samples]\n"
                "images = [image for _, image, _ in samples]\n"
                "expected_keywords = [keywords for _, _, keywords in samples]\n"
                "prefix = caption_prefix.strip() or None\n"
                "digests = {{name: hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for name, image in zip(names, images)}}\n"
                "for name, image in zip(names, images):\n"
                "    print({{'sample_kind': sample_kind, 'name': name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': digests[name]}})\n"
                "print({{'max_new_tokens': max_new_tokens, 'prefix': prefix, 'n_images': len(images)}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `caption` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, an optional prefix that is a non-empty string of "
                "at most `MAX_PREFIX_CHARS` characters (whitespace collapsed), and `max_new_tokens` in `[1, MAX_NEW_TOKENS]` — "
                "and returns an **input manifest** naming the schema (including the 384×384 resize that does not preserve aspect "
                "ratio, and the decoding rule), each input's observed mode and size, the checked prefix, the budget and the "
                "verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, "
                "the cell also validates a 4-pixel image and records the pipeline's own error message as a finding. Inside the "
                "pipeline each image is converted to RGB and resized to `IMAGE_SIZE`×`IMAGE_SIZE`; nothing else is dropped or "
                "altered. The pipeline cannot tell whether an image is a photograph or a drawing: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'IMAGE_SIZE': IMAGE_SIZE, 'MAX_PREFIX_CHARS': MAX_PREFIX_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING}}}})\n"
                "input_manifest = validate_inputs(images, prefix=prefix, max_new_tokens=max_new_tokens, names=names)\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs([Image.new('RGB', (4, 4))])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'tiny-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Caption the images and read the output correctly\n\n"
                "`caption` returns, per image, a dict with `caption` (the decoded text, stripped; the prefix is echoed at the "
                "start when one was given), the checked `prefix`, `image_size`, `new_tokens` (tokens generated after the "
                "prefix, end-of-sequence included), a `truncated` flag that is true when the budget was exhausted, the "
                "generation settings and the model identity. **No score exists**: the caption is generated text with no "
                "probability and no correctness signal, and a fluent caption is not evidence that it describes the image. "
                "Greedy decoding is deterministic on a fixed device and dtype; CUDA kernel selection can change a token and "
                "therefore the rest of the sentence, so GPU and CPU outputs need not match. Each call re-encodes the image, so "
                "cost is per image (about 0.2–0.5 s on the reference CPU). As recorded in the model card, the repository's CPU "
                "smoke captioned these same drawings `a red house with a tree and a ball`, `a beach scene with a sail and a sun` "
                "and `a red and yellow apple on a white background` (the orange became a second apple) — and captioned a blank "
                "white image `a white and black striped rug with a white border` and uniform noise `a very colorful and very "
                "colorful tv screen`: the model always produces a caption, whether or not there is anything to describe. The "
                "cell also records which expected keywords appear in each caption; that is an observation, not a metric."
            ),
            "code": (
                "import time\n\n"
                "results, seconds = [], []\n"
                "for name, image in zip(names, images):\n"
                "    t0 = time.time()\n"
                "    result = pipe.caption(image, prefix=prefix, max_new_tokens=max_new_tokens)\n"
                "    result['image'] = name\n"
                "    results.append(result)\n"
                "    seconds.append(round(time.time() - t0, 2))\n"
                "print({{'device': pipe.device, 'dtype': pipe.dtype, 'seconds_per_image': seconds, 'any_truncated': any(r['truncated'] for r in results)}})\n"
                "keyword_observations = []\n"
                "for result, keywords in zip(results, expected_keywords):\n"
                "    hits = keyword_hits(result['caption'], keywords) if keywords else {{}}\n"
                "    keyword_observations.append({{'image': result['image'], 'expected_keywords': keywords, 'hits': hits}})\n"
                "    print(f\"{{result['image']}}\\n   caption: {{result['caption']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\\n   keywords: {{hits}}\")\n"
                "if any(r['truncated'] for r in results):\n"
                "    print('A budget was exhausted: that caption is incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No quality is "
                "reported by default: caption metrics (CIDEr, BLEU-4, SPICE) need several human-written reference captions per "
                "image from the deployment domain and corpus-level statistics, and this repository ships none (COCO Captions is "
                "not bundled). The repository's helper `unigram_f1` — bag-of-words F1 after normalisation (lower-case, "
                "punctuation removed, whitespace collapsed) against the best-matching reference — exists so that a caller who "
                "does supply references gets a `sample-sanity` report with one entry per image; it is explicitly **not** a "
                "captioning metric. On the default path no references are supplied, the verdict is `not-measurable`, and the "
                "report states what would make the task measurable; the keyword observations from the previous section are "
                "attached to the report file under `observations` for the record. The report is written to "
                "`outputs/{stem}_evaluation_report.json`. To see the other branch, set `references` below to one list of "
                "reference captions per image."
            ),
            "code": (
                "references = None  # e.g. [['a red house with a tree'], ['a sailboat on the sea'], ['an apple and an orange on a table']]\n"
                "report = evaluation_report(results, references, sample_kind=sample_kind)\n"
                "report['observations'] = keyword_observations\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'per_image', 'observations')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:12}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for entry in report.get('per_image', []):\n"
                "    print(f\"  unigram_f1 {{entry['unigram_f1']:.2f}}  {{entry['image']}} -> {{entry['prediction']!r}} (references: {{entry['references']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference captions exist for these images, so nothing is scored; read the captions against the images yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every result (image, caption, prefix, `new_tokens`, `truncated`, the budget), "
                "the evaluation report with the keyword observations, the input manifest, the sample identities and digests, "
                "the notebook's source (repository, revision, embedded module digest, generator), the model identifier, the "
                "immutable model revision, the model licence, the executed weight file and the hosted TensorFlow file it is "
                "not, and the runtime identity (Python, `torch`, `transformers`, device). The captions are also written as CSV "
                "with explicit `image`, `prefix`, `caption`, `new_tokens`, `truncated` columns, and an annotated PNG contact "
                "sheet shows each image with its caption printed beneath it for visual inspection (the model returns no "
                "location, so nothing is drawn on the images themselves) — a supplement to, not a replacement for, the "
                "machine-readable files. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "thumb_w, thumb_h, panel_h = 320, 240, 44\n"
                "sheet = Image.new('RGB', (thumb_w * len(images), thumb_h + panel_h), 'white')\n"
                "draw = ImageDraw.Draw(sheet)\n"
                "panel_font = ImageFont.load_default(size=13)\n"
                "for index, (image, result) in enumerate(zip(images, results)):\n"
                "    thumb = image.convert('RGB').copy()\n"
                "    thumb.thumbnail((thumb_w, thumb_h))\n"
                "    sheet.paste(thumb, (index * thumb_w + (thumb_w - thumb.width) // 2, (thumb_h - thumb.height) // 2))\n"
                "    draw.text((index * thumb_w + 6, thumb_h + 6), result['caption'][:60], fill=(40, 90, 220), font=panel_font)\n"
                "    if len(result['caption']) > 60:\n"
                "        draw.text((index * thumb_w + 6, thumb_h + 24), result['caption'][60:120], fill=(40, 90, 220), font=panel_font)\n"
                "sheet.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'predictions': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'names': names, 'sizes': [list(image.size) for image in images], 'rgb_sha256': digests, 'expected_keywords': expected_keywords}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'executed_weight_file': WEIGHT_FILE,\n"
                "    'hosted_tf_weight_file_not_loaded': HOSTED_TF_WEIGHT_FILE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_captions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'prefix', 'caption', 'new_tokens', 'truncated'])\n"
                "    for result in results:\n"
                "        writer.writerow([result['image'], result['prefix'] or '', result['caption'], result['new_tokens'], result['truncated']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The captions are the text the model generates for an image; nothing in the output scores that text, the model "
        "returns no location or evidence, and it captions every image — including a blank one — with equal fluency. On the "
        "drawn scenes the evaluation report is `not-measurable` by design: no reference captions exist, and the keyword "
        "observations (the repository's smoke run found `house`, `tree`, `red`, `beach`, `sail`, `sun` and `apple`, and called "
        "the orange a second apple) are what you can check by eye, not a metric; they say nothing about photographs, cluttered "
        "scenes, counting, reading, attributes the model tends to hallucinate (colours, backgrounds) or captions longer than a "
        "sentence, and a BYOD result is a per-image observation with the same verdict. **The model captions any image** and "
        "stops only at end-of-sequence or the token budget: check `truncated`, and treat a plausible caption of an empty or "
        "meaningless image as the expected failure mode, not an exception. A prefix steers the caption — `a photography of` "
        "produced `a photography of a red house in a green field` for the house scene — so a conditional caption is partly "
        "your sentence. The pipeline provides no OCR, no VQA, no region captioning, no benchmark evaluation and no training "
        "capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model (a pickle checkpoint loaded with `weights_only=True` only after its digest "
        "matched), validate the demonstrated request, execute the public pipeline path, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen "
        "domain.\n\n"
        "**Next experiments:** set `caption_prefix` to `a drawing of` and compare (the smoke run got `a drawing of a house and "
        "tree`); lower `max_new_tokens` to 3 and watch `truncated` turn true on `a red house`; write one reference caption per "
        "image into `references` and see the verdict switch to `sample-sanity` with a `unigram_f1` you should not mistake for "
        "CIDEr; enable `USE_BYOD` with photographs you know.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/blip-captioning-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/blip-captioning-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/blip-captioning-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/salesforce/BLIP\n"
        "- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li et al., 2022): https://arxiv.org/abs/2201.12086\n"
        "- Microsoft COCO Captions: Data Collection and Evaluation Server (Chen et al., 2015): https://arxiv.org/abs/1504.00325\n"
        "- CIDEr: Consensus-based Image Description Evaluation (Vedantam, Zitnick, Parikh, 2015): https://arxiv.org/abs/1411.5726"
    ),
}
