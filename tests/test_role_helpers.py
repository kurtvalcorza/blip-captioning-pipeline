"""Role-helper contract: validate_inputs (validation stage) and evaluation_report (evaluation stage)."""

from __future__ import annotations

import pytest
from PIL import Image

from blip_captioning_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_PREFIX_CHARS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    caption_tokens,
    evaluation_report,
    keyword_hits,
    normalize_caption,
    unigram_f1,
    validate_inputs,
)


def _image(width: int = 640, height: int = 480) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(caption: str, image: str = "scene.png", truncated: bool = False) -> dict:
    return {"caption": caption, "image": image, "truncated": truncated}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([_image(), _image(480, 480)], names=["scene.png", "fruit.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["prefix_chars"] == [0, MAX_PREFIX_CHARS]
    assert manifest["inputs"] == [
        {"id": "scene.png", "mode": "RGB", "size": [640, 480]},
        {"id": "fruit.png", "mode": "RGB", "size": [480, 480]},
    ]
    assert manifest["prefix"] is None
    assert manifest["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_prefix_and_explicit_request() -> None:
    manifest = validate_inputs([_image()], prefix="  a photography   of ", max_new_tokens=16)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["prefix"] == "a photography of"
    assert manifest["generation"]["max_new_tokens"] == 16


def test_validate_inputs_rejects_like_caption() -> None:
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs(_image())
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs([])
    with pytest.raises(ValueError, match="MAX_PREFIX_CHARS"):
        validate_inputs([_image()], prefix="x" * (MAX_PREFIX_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs([_image()], max_new_tokens=0)
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs([_image(8, 8)])
    with pytest.raises(ValueError, match="names has"):
        validate_inputs([_image()], names=["a", "b"])


def test_normalize_caption_tokens_and_keyword_hits() -> None:
    assert normalize_caption("  A red House, with a tree! ") == "a red house with a tree"
    assert caption_tokens("Two apples.") == ["two", "apples"]
    hits = keyword_hits("a red house with a tree and a ball", ["house", "red house", "sun", "Tree"])
    assert hits == {"house": True, "red house": True, "sun": False, "Tree": True}


def test_unigram_f1_best_reference_and_multiset_overlap() -> None:
    assert unigram_f1("a red house", ["a red house"]) == 1.0
    assert unigram_f1("a red house with a tree", ["a red house"]) == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert unigram_f1("blue sky", ["a red house", "the blue sky"]) == pytest.approx(0.8)
    assert unigram_f1("a a a", ["a"]) == pytest.approx(2 * (1 / 3) * 1.0 / (4 / 3))  # repeats count once
    assert unigram_f1("", ["a red house"]) == 0.0
    with pytest.raises(ValueError, match="references"):
        unigram_f1("x", [])


def test_evaluation_report_not_measurable_without_references() -> None:
    report = evaluation_report([_result("a red house")], sample_kind="synthetic")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_images"] == 1 and report["truncated"] == [False]
    assert "CIDEr" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "no score" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_references() -> None:
    results = [_result("a red house"), _result("two apples", "fruit.png", truncated=True)]
    report = evaluation_report(results, [["a red house"], ["an apple and an orange", "fruit"]])
    assert report["verdict"] == "sample-sanity"
    assert report["truncated"] == [False, True]
    by_id = {metric["id"]: metric for metric in report["metrics"]}
    expected = (1.0 + unigram_f1("two apples", ["an apple and an orange", "fruit"])) / 2
    assert by_id["unigram_f1"]["value"] == pytest.approx(expected)
    assert "CIDEr" in by_id["unigram_f1"]["relation_to_benchmarks"]
    assert [entry["unigram_f1"] for entry in report["per_image"]] == [1.0, 0.0]
    assert report["per_image"][1]["references"] == ["an apple and an orange", "fruit"]


def test_evaluation_report_rejects_mismatched_or_empty_references() -> None:
    with pytest.raises(ValueError, match="references has"):
        evaluation_report([_result("a")], [["a"], ["b"]])
    with pytest.raises(ValueError, match="non-empty sequence"):
        evaluation_report([_result("a")], [[]])
    with pytest.raises(ValueError, match="results"):
        evaluation_report([], None)
