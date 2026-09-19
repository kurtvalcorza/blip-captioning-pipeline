"""Offline tests for the captioning dataset contract, the pinned corpus readers, the corpus metrics and
baselines, BYOD loaders, JSONL export, artifact-manifest rejections and adapt() argument validation. Nothing
here imports torch, transformers or pyarrow; annotations and images come from injected fetchers and tiny PIL
drawings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from blip_captioning_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_FILE,
    DECODER_LAYERS,
    IMAGE_PINS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    BlipCaptioningPipeline,
    bleu4,
    build_sample_dataset,
    caption_metrics,
    check_split_disjoint,
    cider_d,
    colour_neighbour_baseline,
    colour_signature,
    constant_caption_baseline,
    dataset_digest,
    fetch_annotations,
    fetch_images,
    fetch_sample_dataset,
    load_byod_dataset,
    medoid_caption,
    rouge_l,
    split_dataset,
    text_digest,
    validate_dataset,
    write_dataset_jsonl,
)
from blip_captioning_pipeline import pipeline as pl
from blip_captioning_pipeline import samples as sm

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _picture(path, colour, size=(96, 64)):
    image = Image.new("RGB", size, "gray")
    ImageDraw.Draw(image).rectangle([16, 12, 80, 52], fill=colour)
    image.save(path, format="JPEG")
    return path


def _annotations(n, *, zero_caption_every=0):
    rows = []
    for i in range(n):
        colour = COLOURS[i % len(COLOURS)]
        captions = [
            f"A {colour} box on a gray background.",
            f"A {colour} rectangle in the middle of a grey picture",
            f"Someone is holding a {colour} object.",
        ]
        if zero_caption_every and i % zero_caption_every == 0:
            captions = []
        rows.append(
            {"id": str(29631 + i), "captions": captions, "question_type": "other", "text_detected": False}
        )
    return rows


def _records(tmp_path, n=12, prefix="r"):
    out = []
    for i, row in enumerate(_annotations(n)):
        path = _picture(tmp_path / f"{prefix}{i}.jpg", COLOURS[i % len(COLOURS)])
        out.append(
            {"id": f"{prefix}{i:03d}", "image_id": row["id"], "image": str(path), "captions": row["captions"]}
        )
    return out


def _fake_pipeline(text="a box on a gray background"):
    def runner(image, prefix, max_new_tokens):
        return {"caption": text, "new_tokens": 6}

    return BlipCaptioningPipeline(runner, "cpu", "float32", "injected")


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_REPO == "mm-eval/VizWiz-Captions" and len(sm.CORPUS_REVISION) == 40
    assert CORPUS_FILE["path"] == "data/val-00004-of-00005.parquet" and CORPUS_FILE["rows"] == 1_550
    assert len(CORPUS_FILE["sha256"]) == 64 and len(CORPUS_FILE["text_sha256"]) == 64
    assert CORPUS_FILE["row_group"] == 0 and CORPUS_FILE["row_group_rows"] == len(IMAGE_PINS) == 336
    assert sum(SAMPLE_SPLIT.values()) == 318  # the row-group photographs that carry a caption
    assert all(len(digest) == 64 and size > 0 for digest, size in IMAGE_PINS.values())
    assert all(image_id.isdigit() for image_id in IMAGE_PINS)


def test_fetch_annotations_verifies_digest_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    rows = _annotations(5)
    monkeypatch.setitem(CORPUS_FILE, "rows", 5)
    monkeypatch.setitem(CORPUS_FILE, "text_sha256", text_digest(rows))
    calls = []

    def fetcher():
        calls.append(1)
        return rows

    assert fetch_annotations(cache_dir=tmp_path, fetcher=fetcher) == rows
    assert fetch_annotations(cache_dir=tmp_path, fetcher=fetcher) == rows  # served from the cache
    assert len(calls) == 1
    with pytest.raises(ValueError, match="pinned"):
        fetch_annotations(
            cache_dir=tmp_path / "other",
            fetcher=lambda: rows[:4] + [{**rows[4], "captions": ["changed"]}],
        )
    monkeypatch.setitem(CORPUS_FILE, "rows", 99)
    with pytest.raises(ValueError, match="pinned 99"):
        fetch_annotations(cache_dir=tmp_path / "third", fetcher=fetcher)


def test_fetch_images_pins_every_file_and_reads_the_row_group_once(
    tmp_path, monkeypatch, forbid_model_imports
):
    payloads = {str(29631 + i): _picture(tmp_path / f"src{i}.jpg", COLOURS[i]).read_bytes() for i in range(3)}
    monkeypatch.setattr(
        sm, "IMAGE_PINS", {k: (hashlib.sha256(d).hexdigest(), len(d)) for k, d in payloads.items()}
    )
    calls = []

    def fetcher():
        calls.append(1)
        return payloads

    paths = fetch_images(sorted(payloads), cache_dir=tmp_path / "cache", fetcher=fetcher)
    assert sorted(paths) == sorted(payloads) and all(p.is_file() for p in paths.values())
    assert len(calls) == 1  # one row-group read serves every missing photograph
    again = fetch_images(sorted(payloads), cache_dir=tmp_path / "cache", fetcher=fetcher)
    assert again == paths and len(calls) == 1  # cached files are re-verified, not re-fetched
    (tmp_path / "cache" / "images" / "29631.jpg").write_bytes(b"drifted")
    assert fetch_images(["29631"], cache_dir=tmp_path / "cache", fetcher=fetcher) == {"29631": paths["29631"]}
    assert len(calls) == 2 and paths["29631"].read_bytes() == payloads["29631"]
    with pytest.raises(ValueError, match="pinned"):
        fetch_images(["29631"], cache_dir=tmp_path / "bad", fetcher=lambda: {"29631": b"tampered"})
    with pytest.raises(ValueError, match="pinned"):
        fetch_images(["29632"], cache_dir=tmp_path / "bad", fetcher=lambda: {})
    with pytest.raises(ValueError, match="not one of the"):
        fetch_images(["99999"], cache_dir=tmp_path / "bad", fetcher=fetcher)


def test_http_range_file_reads_ranges_and_refuses_full_responses(monkeypatch, forbid_model_imports):
    blob = bytes(range(256)) * 4
    requests = []

    class Response:
        def __init__(self, status, data):
            self.status, self._data = status, data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self._data

    def fake_urlopen(request, timeout=0):
        header = request.headers["Range"]
        requests.append(header)
        start, end = (int(x) for x in header.removeprefix("bytes=").split("-"))
        return Response(200 if start == 900 else 206, blob[start : end + 1])

    monkeypatch.setattr(sm.urllib.request, "urlopen", fake_urlopen)
    handle = sm._HttpRangeFile("https://example.invalid/shard.parquet", len(blob))
    assert handle.seek(-8, 2) == len(blob) - 8 and handle.read() == blob[-8:] and handle.tell() == len(blob)
    handle.seek(0)
    assert handle.read(4) == blob[:4] and handle.read(0) == b"" and handle.fetched == 12
    assert requests == ["bytes=1016-1023", "bytes=0-3"]
    handle.seek(900)
    with pytest.raises(ValueError, match="ignored the Range"):
        handle.read(4)


def test_build_sample_dataset_is_seeded_image_disjoint_and_skips_uncaptioned(
    tmp_path, monkeypatch, forbid_model_imports
):
    rows = _annotations(12, zero_caption_every=6)  # ids 29631 and 29637 carry no caption
    pins = {row["id"]: ("0" * 64, 1) for row in rows[:10]}
    monkeypatch.setattr(sm, "IMAGE_PINS", pins)
    paths = {k: tmp_path / f"{k}.jpg" for k in pins}
    sizes = {"train": 5, "validation": 2, "test": 1}
    splits = build_sample_dataset(rows, seed=1, sizes=sizes, image_paths=paths)
    assert {k: len(v) for k, v in splits.items()} == sizes and check_split_disjoint(splits)
    assert splits["train"][0]["id"] == "train-0000" and splits["train"][0]["image"].endswith(".jpg")
    chosen = {r["image_id"] for part in splits.values() for r in part}
    assert chosen == set(pins) - {"29631", "29637"}
    assert build_sample_dataset(rows, seed=1, sizes=sizes, image_paths=paths) == splits
    assert build_sample_dataset(rows, seed=2, sizes=sizes, image_paths=paths) != splits
    with pytest.raises(ValueError, match="split sizes need 9"):
        build_sample_dataset(rows, sizes={"train": 6, "validation": 2, "test": 1})
    with pytest.raises(ValueError, match="absent from the annotations"):
        build_sample_dataset(rows[:5], sizes=sizes)


def test_fetch_sample_dataset_end_to_end_with_injected_fetchers(tmp_path, monkeypatch, forbid_model_imports):
    rows = _annotations(10)
    payloads = {
        row["id"]: _picture(tmp_path / f"s{i}.jpg", COLOURS[i % 6]).read_bytes() for i, row in enumerate(rows)
    }
    monkeypatch.setitem(CORPUS_FILE, "rows", 10)
    monkeypatch.setitem(CORPUS_FILE, "text_sha256", text_digest(rows))
    monkeypatch.setattr(
        sm, "IMAGE_PINS", {k: (hashlib.sha256(d).hexdigest(), len(d)) for k, d in payloads.items()}
    )
    splits = fetch_sample_dataset(
        cache_dir=tmp_path / "cache",
        annotation_fetcher=lambda: rows,
        image_fetcher=lambda: payloads,
        sizes={"train": 8, "validation": 1, "test": 1},
    )
    report = validate_dataset(splits["train"])
    assert report["n_records"] == 8 and report["unique_images"] == 8
    assert report["captions_per_image"] == {"min": 3, "max": 3} and report["caption_words"]["min"] >= 5
    assert report["categories"] == {"no-text": 8}  # text_detected False in every fixture row


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path, forbid_model_imports):
    records = _records(tmp_path)
    report = validate_dataset(records)
    assert report["n_records"] == 12 and report["unique_images"] == 12
    assert report["categories"] == {"other": 12}  # BYOD records carry no category
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    assert report["records"][0]["image_size"] == [96, 64]
    good = records
    tiny = _picture(tmp_path / "tiny.jpg", "red", size=(8, 8))
    for bad, message in (
        (good[:7], "8..5000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "image": str(tmp_path / "missing.jpg")}, *good[1:]], "not found"),
        ([{**good[0], "image": str(tiny)}, *good[1:]], "MIN_IMAGE_SIDE"),
        ([{**good[0], "captions": []}, *good[1:]], "at least 1"),
        ([{**good[0], "captions": "one string"}, *good[1:]], "at least 1"),
        ([{**good[0], "captions": ["ok", "  "]}, *good[1:]], "non-empty string"),
        ([{**good[0], "captions": ["x" * 600]}, *good[1:]], "MAX_CAPTION_CHARS"),
        ([{**good[0], "prefix": "p" * 200}, *good[1:]], "MAX_PREFIX_CHARS"),
        ([{k: v for k, v in good[0].items() if k != "captions"}, *good[1:]], "missing 'captions'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)
    relative = [{**r, "image": Path(r["image"]).name} for r in records]
    assert validate_dataset(relative, base_dir=tmp_path)["n_records"] == 12


def test_split_dataset_keeps_images_together_and_is_seeded(tmp_path, forbid_model_imports):
    records = _records(tmp_path, n=16)
    records += [{**r, "id": r["id"] + "b", "captions": r["captions"][:1]} for r in records[:6]]
    splits = split_dataset(records, val_fraction=0.15, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == len(records) and check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.15, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:1], "test": [{**records[0], "id": "dup"}]})


# --- metrics and baselines ----------------------------------------------------------------------------


def test_caption_metrics_are_bounded_and_reward_the_references(tmp_path, forbid_model_imports):
    records = _records(tmp_path)
    refs = [r["captions"] for r in records]
    perfect = caption_metrics([r[0] for r in refs], refs)
    assert perfect["bleu4"] == pytest.approx(1.0) and perfect["rouge_l"] == pytest.approx(1.0)
    assert perfect["empty_rate"] == 0.0 and perfect["n"] == 12 and perfect["mean_words"] > 0
    empty = caption_metrics([""] * 12, refs)
    assert empty["bleu4"] == 0.0 and empty["rouge_l"] == 0.0 and empty["cider_d"] == 0.0
    assert empty["empty_rate"] == 1.0
    shuffled = [refs[(i + 3) % 12][0] for i in range(12)]  # a different image's caption
    assert caption_metrics(shuffled, refs)["cider_d"] < perfect["cider_d"]
    assert bleu4(["a b c d e"], [["a b c d e f"]]) < 1.0 < 1.0001  # brevity penalty applies
    assert bleu4(["a"], [["b"]]) == 0.0 and rouge_l("", ["a b"]) == 0.0
    assert rouge_l("a b c", ["a x c", "zzz"]) == pytest.approx(
        (1 + 1.44) * (2 / 3) * (2 / 3) / (2 / 3 + 1.44 * 2 / 3)
    )
    scores = cider_d(["a red box on a gray background"] * 2, [refs[0], refs[1]])
    assert scores[0] > scores[1] >= 0.0  # the red image's references match better than the green one's
    with pytest.raises(ValueError, match="reference lists"):
        caption_metrics(["a"], [["a"], ["b"]])
    with pytest.raises(ValueError, match="non-empty list"):
        caption_metrics(["a"], [[]])


def test_baselines_need_training_records_and_score_in_range(tmp_path, forbid_model_imports):
    records = _records(tmp_path)
    medoid = medoid_caption(records[:8])
    assert medoid in {c for r in records[:8] for c in r["captions"]}
    constant = constant_caption_baseline(records[:8], records[8:])
    assert constant["n"] == 4 and constant["cider_d"] >= 0.0 and "constant caption" in constant["baseline"]
    assert len(colour_signature(records[0]["image"])) == 27
    assert all(0.0 <= v <= 1.0 for v in colour_signature(Image.open(records[0]["image"])))
    neighbour = colour_neighbour_baseline(records[:6], records[6:])
    assert neighbour["n"] == 6 and "nearest neighbour" in neighbour["baseline"]
    # every test image's colour twin sits in the training half (same palette, six colours), so the
    # neighbour's first caption names the right colour and scores far above the constant caption
    assert (
        neighbour["bleu4"] > 0.5
        and neighbour["cider_d"] > constant_caption_baseline(records[:6], records[6:])["cider_d"]
    )
    with pytest.raises(ValueError, match="training records"):
        constant_caption_baseline([], records)
    with pytest.raises(ValueError, match="training records"):
        colour_neighbour_baseline([], records)


# --- BYOD loaders and JSONL ---------------------------------------------------------------------------


def test_byod_json_jsonl_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records(tmp_path, n=8)
    path = write_dataset_jsonl(records, tmp_path / "data.jsonl")
    assert load_byod_dataset(path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv").write_text("id,captions\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.json")


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    records = _records(tmp_path)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records, lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records, batch_size=0)
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.adapt(records, trainable_decoder_layers=DECODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(records)
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    # evaluate() only needs the caption path, so it works with an injected runner
    metrics = pipe.evaluate(records)
    assert (
        metrics["n"] == 12 and metrics["verdict"] == "measured-small-sample" and metrics["adapted"] is False
    )
    assert 0.0 < metrics["rouge_l"] < 1.0 and metrics["max_new_tokens"] == pl.DEFAULT_MAX_NEW_TOKENS


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["text_decoder.cls.predictions.bias"],
        "adapter": {"trainable_decoder_layers": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_decoder_layers": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {"trainable_decoder_layers": DECODER_LAYERS + 1}})
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
