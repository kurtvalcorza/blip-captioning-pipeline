import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blip_captioning_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WEIGHTS_DIR,
    HOSTED_TF_WEIGHT_FILE,
    IMAGE_SIZE,
    MAX_IMAGE_SIDE,
    MAX_NEW_TOKENS,
    MAX_PREFIX_CHARS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    WEIGHT_FILE,
    BlipCaptioningPipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "Salesforce/blip-image-captioning-base"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert 1 <= DEFAULT_MAX_NEW_TOKENS <= MAX_NEW_TOKENS == 64
    assert IMAGE_SIZE == 384 and MAX_PREFIX_CHARS == 128
    assert WEIGHT_FILE == "pytorch_model.bin" and HOSTED_TF_WEIGHT_FILE == "tf_model.h5"
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION
        paths = [entry["path"] for entry in data["files"]]
        # The executed artifact is the pickle checkpoint; the hosted TensorFlow file is never listed.
        assert WEIGHT_FILE in paths and HOSTED_TF_WEIGHT_FILE not in paths
        assert "model.safetensors" not in paths


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "blip"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "blip"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def _fake_pipeline(calls: list | None = None) -> BlipCaptioningPipeline:
    def runner(image, prefix, max_new_tokens):
        if calls is not None:
            calls.append((image.mode, prefix, max_new_tokens))
        caption = f"{prefix} a red house" if prefix else " a red house with a tree "
        return {"caption": caption, "new_tokens": 3 if prefix else 8}

    return BlipCaptioningPipeline(runner, "cpu", "float32", "injected")


def test_caption_output_fields_and_defaults():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.caption(Image.new("L", (400, 300)))
    assert result["caption"] == "a red house with a tree"  # stripped
    assert result["prefix"] is None
    assert result["image_size"] == [400, 300]
    assert result["new_tokens"] == 8 and result["truncated"] is False
    assert result["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (result["device"], result["dtype"], result["source"]) == ("cpu", "float32", "injected")
    assert calls == [("RGB", None, DEFAULT_MAX_NEW_TOKENS)]


def test_caption_with_prefix_collapses_whitespace_and_echoes_it():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.caption(Image.new("RGB", (64, 64)), prefix="  a photography   of ")
    assert result["prefix"] == "a photography of"
    assert result["caption"] == "a photography of a red house"
    assert calls == [("RGB", "a photography of", DEFAULT_MAX_NEW_TOKENS)]


def test_caption_reports_truncation():
    pipe = _fake_pipeline()
    result = pipe.caption(Image.new("RGB", (64, 64)), max_new_tokens=8)
    assert result["truncated"] is True and result["caption"] == "a red house with a tree"
    result = pipe.caption(Image.new("RGB", (64, 64)), prefix="a drawing of", max_new_tokens=3)
    assert result["truncated"] is True


def test_caption_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.caption(np.zeros((30, 40, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.caption(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.caption(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)))
    with pytest.raises(TypeError, match="prefix must be a str"):
        pipe.caption(Image.new("RGB", (64, 64)), prefix=3)
    with pytest.raises(ValueError, match="non-whitespace"):
        pipe.caption(Image.new("RGB", (64, 64)), prefix="   ")
    with pytest.raises(ValueError, match="MAX_PREFIX_CHARS"):
        pipe.caption(Image.new("RGB", (64, 64)), prefix="x" * (MAX_PREFIX_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        pipe.caption(Image.new("RGB", (64, 64)), max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(TypeError, match="max_new_tokens"):
        pipe.caption(Image.new("RGB", (64, 64)), max_new_tokens=True)


def test_caption_rejects_malformed_runner_output():
    pipe = BlipCaptioningPipeline(lambda *args: {"tokens": 1}, "cpu")
    with pytest.raises(RuntimeError, match="caption"):
        pipe.caption(Image.new("RGB", (64, 64)))
