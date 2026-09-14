"""Image captioning with the pinned ``Salesforce/blip-image-captioning-base`` checkpoint (BLIP, ViT-B).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the BLIP architecture comes from the pinned ``transformers`` release and no
model-repository code is executed. Upstream ships no SafeTensors at this revision: the PyTorch weights
are ``pytorch_model.bin`` (a pickle), so the trust boundary is the manifest SHA-256 checked before the
load plus ``weights_only=True`` deserialisation; the ``tf_model.h5`` upstream also hosts is the DIMER
upload artifact and is never loaded here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "Salesforce/blip-image-captioning-base"
MODEL_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"
MODEL_LICENSE = "bsd-3-clause"
MODEL_KEY = "blip-image-captioning-base"
WEIGHT_FILE = "pytorch_model.bin"  # the only PyTorch weight file upstream: a pickle, digest-pinned
HOSTED_TF_WEIGHT_FILE = "tf_model.h5"  # DIMER upload artifact (accepted format); never loaded by this package
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Generation ceilings. COCO-style captions are one sentence (the checkpoint's text_config max_length
# is 20); the default leaves room for a long sentence and the ceiling bounds runaway generation.
MAX_NEW_TOKENS = 64
DEFAULT_MAX_NEW_TOKENS = 30
DECODING = "greedy"
# Optional conditional-captioning prefix (the upstream README's "a photography of"); the model
# continues it. A prefix longer than a short phrase is not what the model was trained on.
MAX_PREFIX_CHARS = 128
# Input ceilings. The processor resizes every image to 384x384 (preprocessor_config.json, aspect
# ratio not preserved) into 24x24 = 576 ViT-B/16 patches, so image cost is bounded; the side ceiling
# only guards memory during decoding and resizing.
IMAGE_SIZE = 384
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
_PUNCT_RE = re.compile(r"[^\w\s]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def normalize_caption(text: str) -> str:
    """COCO-caption-style normalisation: lower-case, punctuation removed, whitespace collapsed."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def caption_tokens(text: str) -> list[str]:
    return normalize_caption(text).split()


def unigram_f1(prediction: str, references: Sequence[str]) -> float:
    """Bag-of-words F1 between the normalised prediction and the best-matching reference.

    A plumbing check, not a captioning metric: CIDEr, BLEU-4 and SPICE need several references per
    image and corpus-level statistics. Multiset overlap counts repeated words once per occurrence.
    """
    if not references:
        raise ValueError("references must contain at least one caption")
    pred = caption_tokens(prediction)
    best = 0.0
    for reference in references:
        ref = caption_tokens(reference)
        if not pred or not ref:
            continue
        ref_counts: dict[str, int] = {}
        for token in ref:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = 0
        for token in pred:
            if ref_counts.get(token, 0) > 0:
                overlap += 1
                ref_counts[token] -= 1
        if overlap:
            precision, recall = overlap / len(pred), overlap / len(ref)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def keyword_hits(caption: str, keywords: Sequence[str]) -> dict[str, bool]:
    """Which of the caller's keywords (normalised, whole-token match) appear in the caption."""
    tokens = set(caption_tokens(caption))
    return {keyword: all(part in tokens for part in caption_tokens(keyword)) for keyword in keywords}


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one image as PIL.Image.Image (any mode, converted to RGB) plus an optional caption prefix",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "prefix_chars": [0, MAX_PREFIX_CHARS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "decoding": f"{DECODING} (do_sample=False, num_beams=1), deterministic on a fixed device and dtype",
    "preprocessing": (
        "image resized to 384x384 (aspect ratio not preserved, CLIP mean/std) into 576 ViT-B/16 patches; "
        "the text decoder starts from the BOS token (unconditional) or from the tokenised prefix "
        "(conditional) and generates the caption"
    ),
    "output": "one caption string (the model's decoded text, prefix included when given), no score",
}


def _check_inputs(image: Any, prefix: Any, max_new_tokens: Any) -> tuple[Image.Image, str | None, int]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``caption`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    checked_prefix: str | None = None
    if prefix is not None:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str or None")
        checked_prefix = " ".join(prefix.split())
        if not checked_prefix:
            raise ValueError("prefix must contain at least one non-whitespace character or be None")
        if len(checked_prefix) > MAX_PREFIX_CHARS:
            raise ValueError(f"prefix has {len(checked_prefix)} chars > MAX_PREFIX_CHARS {MAX_PREFIX_CHARS}")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    return rgb, checked_prefix, max_new_tokens


def validate_inputs(
    images: Sequence[Image.Image],
    *,
    prefix: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Every image is checked exactly as ``caption`` would check it; rejection is reported by raising,
    and a caller that wants the finding recorded catches the exception and stores ``str(exc)`` under
    ``findings``.
    """
    if isinstance(images, Image.Image) or not isinstance(images, Sequence) or not images:
        raise TypeError("images must be a non-empty sequence of PIL.Image.Image")
    if names is not None and len(names) != len(images):
        raise ValueError(f"names has {len(names)} entries for {len(images)} images")
    checked_prefix = None
    observed = []
    for index, image in enumerate(images):
        _, checked_prefix, _ = _check_inputs(image, prefix, max_new_tokens)
        observed.append(
            {"id": names[index] if names else f"image-{index}", "mode": image.mode, "size": list(image.size)}
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": observed,
        "prefix": checked_prefix,
        "generation": {"max_new_tokens": int(max_new_tokens), "do_sample": False, "decoding": DECODING},
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Sequence[Mapping[str, Any]],
    references: Sequence[Sequence[str]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``references`` (one sequence of reference captions per result, in order) the report carries
    the mean ``unigram_f1`` over the images plus one per-image entry, verdict ``sample-sanity``;
    without references it is ``not-measurable`` and says what labelled data would make the task
    measurable. Neither is a captioning benchmark.
    """
    if not results:
        raise ValueError("results must contain at least one caption result")
    base = {
        "task": "image (+ optional prefix) -> caption text (image captioning)",
        "score_semantics": (
            "the caption is generated text and carries no score, probability or correctness signal; a "
            "fluent caption is not evidence that it describes the image. Greedy decoding makes the output "
            "reproducible on a fixed device and dtype, a reproducibility property, not a quality one"
        ),
        "sample_kind": sample_kind,
        "n_images": len(results),
        "truncated": [bool(result.get("truncated")) for result in results],
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if references is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference captions were supplied for the captioned images",
            "needs": (
                "several human-written reference captions per image from the deployment domain "
                "(COCO Captions-style annotations, five per image) scored with CIDEr / BLEU-4 / SPICE "
                "over a corpus; no such labelled set ships with this repository"
            ),
        }
    if len(references) != len(results):
        raise ValueError(f"references has {len(references)} entries for {len(results)} results")
    per_image = []
    for result, refs in zip(results, references, strict=True):
        if isinstance(refs, str) or not refs:
            raise ValueError("each references entry must be a non-empty sequence of captions")
        prediction = str(result["caption"])
        per_image.append(
            {
                "image": result.get("image"),
                "prediction": prediction,
                "references": list(refs),
                "unigram_f1": unigram_f1(prediction, refs),
            }
        )
    metrics = [
        {
            "id": "unigram_f1",
            "value": sum(entry["unigram_f1"] for entry in per_image) / len(per_image),
            "normalisation": "lower-cased, punctuation removed, whitespace collapsed; best reference",
            "relation_to_benchmarks": (
                "bag-of-words overlap with the best reference; not CIDEr, BLEU-4 or SPICE, which need "
                "several references per image and corpus-level statistics"
            ),
            "estimation": f"{len(per_image)} image(s), no dispersion estimate",
        }
    ]
    return {
        **base,
        "metrics": metrics,
        "per_image": per_image,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(per_image)} image(s) with caller-written reference captions; plumbing evidence, not a "
            "captioning benchmark"
        ),
        "needs": (
            "several human-written reference captions per image from the deployment domain scored with "
            "CIDEr / BLEU-4 / SPICE over a corpus for any quality claim; COCO Captions is not bundled"
        ),
    }


@dataclass
class BlipCaptioningPipeline:
    """``_runner(image, prefix, max_new_tokens)`` returns ``{"caption": str, "new_tokens": int}``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BlipCaptioningPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained(location, **common)
        # Trust boundary: the only PyTorch weight file upstream is a pickle (pytorch_model.bin). Its
        # SHA-256 was checked against the manifest above; use_safetensors=False names that fact, and
        # weights_only=True makes transformers deserialise with torch.load(weights_only=True), whose
        # restricted unpickler admits tensors, primitives and containers only.
        model = BlipForConditionalGeneration.from_pretrained(
            location, dtype=torch.float32, use_safetensors=False, weights_only=True, **common
        )
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, prefix: str | None, max_new_tokens: int) -> dict[str, Any]:
            if prefix is None:
                inputs = processor(images=image, return_tensors="pt").to(resolved_device)
                prompt_len = 0
            else:
                inputs = processor(images=image, text=prefix, return_tensors="pt").to(resolved_device)
                prompt_len = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
                )
            ids = generated[0]
            decoded = processor.decode(ids, skip_special_tokens=True)
            # Unconditional: bos + caption + sep. Conditional: the prompt ids minus their trailing
            # [SEP] are echoed first (BlipForConditionalGeneration.generate drops it), then the new ones.
            new_tokens = int(ids.shape[0]) - (prompt_len - 1 if prompt_len else 1)
            return {"caption": decoded, "new_tokens": max(new_tokens, 0)}

        return cls(runner, resolved_device, "float32", source)

    def caption(
        self,
        image: Image.Image,
        *,
        prefix: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Caption one image; ``caption`` is the decoded text, stripped (prefix included when given)."""
        rgb, checked_prefix, checked_tokens = _check_inputs(image, prefix, max_new_tokens)
        raw = self._runner(rgb, checked_prefix, checked_tokens)
        if not isinstance(raw, dict) or "caption" not in raw:
            raise RuntimeError("runner must return a dict with 'caption'")
        new_tokens = int(raw.get("new_tokens", 0))
        return {
            "caption": str(raw["caption"]).strip(),
            "prefix": checked_prefix,
            "image_size": list(rgb.size),
            "new_tokens": new_tokens,
            "truncated": new_tokens >= checked_tokens,
            "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
