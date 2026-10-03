"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/blip_captioning_colab.ipynb`
(review PR #7: CAP-M1..M5, CAP-m1..m3).

The notebook's own cells are executed from the committed JSON in a namespace of the package's public API and
inert stand-ins (an injected caption runner, a fake `google.colab`, tiny PIL drawings); nothing here loads the
model, so these tests run in CI without weights.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import blip_captioning_pipeline as bcp
from blip_captioning_pipeline import BlipCaptioningPipeline, byod_minimum_records, split_dataset

# Windows conda trap (fleet note, bioclip2 row 6): import torch before any NumPy linear algebra in this process.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "blip_captioning_colab.ipynb"
COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _code_after(heading: str) -> str:
    """Source of the first code cell after the markdown cell containing `heading`."""
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in "".join(cell["source"]):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return "".join(nxt["source"])
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join("".join(c["source"]) for c in _cells() if c["cell_type"] == "markdown")


def _picture(path: Path, colour: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 48), "gray")
    ImageDraw.Draw(image).rectangle([8, 8, 56, 40], fill=colour)
    image.save(path, format="JPEG")
    return path


def _records(root: Path, n: int, *, prefix: str = "r", folder: str = "") -> list[dict]:
    out = []
    for i in range(n):
        colour = COLOURS[i % len(COLOURS)]
        rel = f"{folder}{prefix}{i:03d}.jpg"
        _picture(root / rel, colour)
        out.append({"id": f"{prefix}{i:03d}", "image": rel, "captions": [f"a {colour} box on a gray background"], "category": "drawing"})
    return out


def _zip(tmp_path: Path, name: str, n: int, *, prefix: str = "r", folder: str = "", records_name: str = "records.jsonl") -> bytes:
    src = tmp_path / f"src_{name}"
    records = _records(src, n, prefix=prefix, folder=folder)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if records_name:
            text = "\n".join(json.dumps(r) for r in records) if records_name.endswith(".jsonl") else json.dumps(records)
            archive.writestr(records_name, text)
        for r in records:
            archive.write(src / r["image"], r["image"])
    return buffer.getvalue()


def _fake_colab(monkeypatch, uploads: list[dict]):
    """A fake `google.colab.files.upload` returning the queued uploads in order."""
    queue = list(uploads)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: queue.pop(0)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _section4(monkeypatch, tmp_path, *, byod_path: str = "", use_byod: bool = True) -> dict:
    """Execute Section 4 verbatim (form literals substituted) in a namespace of the package API."""
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. VizWiz photographs, captions and split")
    source = source.replace("USE_BYOD = False", f"USE_BYOD = {use_byod}", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)
    ns: dict = {name: getattr(bcp, name) for name in bcp.__all__}
    ns.update({"os": __import__("os"), "Path": Path})
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


# --- CAP-M4: the BYOD minimum, refusals and layout ----------------------------------------------------


def test_byod_minimum_is_twelve_and_split_refusals_name_the_split(tmp_path):
    assert byod_minimum_records() == 12
    records = [{**r, "image": str(tmp_path / r["image"])} for r in _records(tmp_path, 60)]
    for n in range(8, 12):
        with pytest.raises(ValueError, match=r"the (train|validation) split has \d+ records .*at least 12 records"):
            split_dataset(records[:n], seed=42)
    for n in (12, 13, 20, 49, 50, 60):
        splits = split_dataset(records[:n], seed=42)
        assert len(splits["train"]) >= 8 and len(splits["validation"]) >= 2 and len(splits["test"]) >= 2


def test_twelve_records_pass_section4_and_eleven_are_refused_there(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{"small.zip": _zip(tmp_path, "small", 11)}, {"ok.zip": _zip(tmp_path, "ok", 12)}])
    with pytest.raises(ValueError, match="the train split has 7 records"):
        _section4(monkeypatch, tmp_path)
    ns = _section4(monkeypatch, tmp_path)
    assert ns["disjoint"] == {"test": 2, "validation": 2, "train": 8}
    assert ns["data_source"] == "BYOD (ok.zip)" and ns["byod"]["records"] == 12 and ns["byod"]["minimum_records"] == 12


def test_second_upload_replaces_the_first(monkeypatch, tmp_path):
    first = _zip(tmp_path, "a", 14, prefix="a", records_name="records.jsonl")
    second = _zip(tmp_path, "b", 13, prefix="b", records_name="records.json")
    _fake_colab(monkeypatch, [{"a.zip": first}, {"b.zip": second}])
    _section4(monkeypatch, tmp_path)
    ns = _section4(monkeypatch, tmp_path)
    ids = {r["id"] for part in (ns["train_records"], ns["val_records"], ns["test_records"]) for r in part}
    assert ns["data_source"] == "BYOD (b.zip)" and len(ids) == 13 and all(i.startswith("b") for i in ids)
    assert not (tmp_path / "work" / "byod" / "records.jsonl").exists()


def test_subfolder_image_paths_are_kept(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{"photos.zip": _zip(tmp_path, "p", 12, folder="photos/")}])
    ns = _section4(monkeypatch, tmp_path)
    assert all("photos" in Path(r["image"]).parts for r in ns["train_records"])


def test_cancelled_upload_and_missing_records_file_are_actionable(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{}, {"no_records.zip": _zip(tmp_path, "n", 12, records_name="")}])
    with pytest.raises(RuntimeError, match="Upload exactly one .zip file \\(received 0\\)"):
        _section4(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="No records.jsonl or records.json found in no_records.zip"):
        _section4(monkeypatch, tmp_path)


def test_byod_path_reads_a_zip_or_a_folder_without_colab(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    archive = tmp_path / "mine.zip"
    archive.write_bytes(_zip(tmp_path, "m", 12))
    ns = _section4(monkeypatch, tmp_path, byod_path=str(archive))
    assert ns["data_source"] == "BYOD (mine.zip)" and len(ns["byod"]["zip_sha256"]) == 64
    folder = tmp_path / "folder"
    records = _records(folder, 12)
    (folder / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, byod_path=str(folder))
    assert ns["data_source"] == "BYOD (folder)" and ns["byod"]["zip_sha256"] is None
    with pytest.raises(FileNotFoundError, match="does not exist in this runtime"):
        _section4(monkeypatch, tmp_path, byod_path=str(tmp_path / "missing.zip"))


def test_byod_provenance_carries_no_vizwiz_corpus_block():
    source = _code_after("## 9. Re-caption the drawn scenes")
    assert "'corpus': None if USE_BYOD else {" in source and "'byod': byod," in source
    section4 = _code_after("## 4. VizWiz photographs, captions and split")
    byod_print = section4.split("if USE_BYOD:\n    print(", 1)[1].split("\n", 1)[0]
    assert "text_sha256" not in byod_print and "pinned_photographs" not in byod_print


# --- CAP-m3: zip limits and unsafe members, checked before anything is written ------------------------


def test_zip_limits_and_unsafe_members_are_refused_before_writing(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{"ok.zip": _zip(tmp_path, "ok", 12)}])
    ns = _section4(monkeypatch, tmp_path)
    extract = ns["extract_zip"]
    payload = _zip(tmp_path, "big", 3)
    root = tmp_path / "dest"
    root.mkdir()
    ns["MAX_ZIP_BYTES"] = 10
    with pytest.raises(ValueError, match="the limits are"):
        extract(payload, root, "big.zip")
    ns["MAX_ZIP_BYTES"], ns["MAX_ZIP_MEMBERS"] = 2_000_000_000, 2
    with pytest.raises(ValueError, match="the limits are"):
        extract(payload, root, "big.zip")
    assert not any(root.iterdir())
    ns["MAX_ZIP_MEMBERS"] = 20_000
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("records.jsonl", "")
        archive.writestr("../escaped.txt", "x")
    with pytest.raises(ValueError, match="would land outside the upload folder"):
        extract(buffer.getvalue(), root, "evil.zip")
    assert not any(root.iterdir()) and not (tmp_path / "escaped.txt").exists()
    with pytest.raises(ValueError, match="not a readable zip file"):
        extract(b"not a zip", root, "broken.zip")


# --- CAP-M2 / CAP-M3: verdicts instead of assertions, and re-runs from the pretrained model ----------


def test_adapt_refuses_an_already_adapted_pipeline():
    pipe = BlipCaptioningPipeline(lambda image, prefix, n: {"caption": "x", "new_tokens": 1}, "cpu", "float32", "injected")
    pipe.adapter = {"best_epoch": 1}
    with pytest.raises(ValueError, match="already adapted"):
        pipe.adapt([], None)


def _evaluation_namespace(tmp_path, monkeypatch, *, best_epoch: int) -> dict:
    """Sections 6 and 8 executed verbatim with an injected caption runner and a stand-in adaptation result."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    records = [{**r, "image": str(tmp_path / r["image"])} for r in _records(tmp_path, 20)]
    splits = split_dataset(records, seed=42)
    pipe = BlipCaptioningPipeline(lambda image, prefix, n: {"caption": "a red box on a gray background", "new_tokens": 7}, "cpu", "float32", "injected")
    ns: dict = {name: getattr(bcp, name) for name in bcp.__all__}
    ns.update(
        {
            "os": __import__("os"), "Path": Path, "json": json, "time": __import__("time"), "collections": __import__("collections"), "Image": Image,
            "pipe": pipe, "CAPTION_MAX_TOKENS": 30, "USE_BYOD": True, "data_source": "BYOD (test.zip)",
            "train_records": splits["train"], "val_records": splits["validation"], "test_records": splits["test"],
            "dataset_manifests": {k: bcp.validate_dataset(v, min_records=1) for k, v in splits.items()},
            "disjoint": {k: len(v) for k, v in splits.items()}, "categories": {}, "reset_to_pretrained": lambda: None,
        }
    )
    exec(compile(_code_after("## 6. Baselines and the frozen model"), "<section 6>", "exec"), ns)
    ns.update(
        {
            "EPOCHS": 4, "LEARNING_RATE": 5e-5, "TRAINABLE_DECODER_LAYERS": 2, "adapt_seconds": 1.0,
            "adapt_result": {"best_epoch": best_epoch, "selection": "highest validation CIDEr-D", "history": [], "trainable_names": [], "n_trainable": 1},
        }
    )
    exec(compile(_code_after("## 8. Held-out evaluation"), "<section 8>", "exec"), ns)
    return ns


def test_kept_epoch_zero_is_reported_and_the_cell_carries_on(tmp_path, monkeypatch, capsys):
    ns = _evaluation_namespace(tmp_path, monkeypatch, best_epoch=0)
    assert ns["verdict"].startswith("the selector kept epoch 0") and "0.000" in ns["verdict"]
    report = json.loads((tmp_path / "outputs" / "blip_captioning_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["selection"]["best_epoch"] == 0 and report["selection"]["split"] == "validation"
    assert "optimistic" in report["selection"]["default_hyperparameters_chosen_on"]
    out = capsys.readouterr().out
    assert "frozen_above_constant_caption" in out and "run" in out


def test_not_improved_is_a_verdict_not_an_error(tmp_path, monkeypatch):
    ns = _evaluation_namespace(tmp_path, monkeypatch, best_epoch=3)
    assert ns["verdict"].startswith("not improved") and len(ns["run_history"]) == 1


def test_sections_5_to_7_reset_to_pretrained_and_section_6_refuses_an_adapted_model():
    for heading in ("## 5. Caption through the inference contract", "## 6. Baselines and the frozen model", "## 7. Bounded fine-tuning"):
        assert "reset_to_pretrained()" in _code_after(heading), heading
    assert "if frozen_test['adapted']:" in _code_after("## 6. Baselines and the frozen model")


def test_no_learner_cell_asserts_a_result():
    for cell in _cells():
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or "dimer" in cell.get("metadata", {}) or "# dimer: kernel cell" in source:
            continue
        assert not re.search(r"^\s*assert ", source, re.M), source[:120]


# --- CAP-M2: honest statement of how the defaults were chosen -----------------------------------------


def test_learner_text_states_the_defaults_were_chosen_on_the_test_split():
    markdown = _markdown()
    assert "The test photographs were never used for training or epoch selection" not in markdown
    assert "the configuration that gained on the held-out split" not in markdown
    assert "**selected, optimistic** number rather than independent evidence" in markdown
    assert "by their score on the test photographs" in markdown


# --- CAP-M1: isolated runtime ---------------------------------------------------------------------------


def test_exactly_two_kernel_cells_and_no_in_kernel_pip_install():
    kernel = [c for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in "".join(c["source"])]
    assert len(kernel) == 2
    install = "".join(kernel[0]["source"])
    assert "--require-hashes" in install and "--managed-python" in install and "LOCK_SHA256" in install
    assert "Restart the runtime, then rerun" not in _markdown()


# --- CAP-M5 / CAP-m1 / CAP-m2: guided layer, braces, timings -------------------------------------------


def test_guided_layer_and_infrastructure_labels():
    markdown = _markdown()
    for marker, least in (("**Predict before running:**", 6), ("**What to notice:**", 6), ("<summary>Check your reasoning</summary>", 7), ("> **Infrastructure.**", 3)):
        assert markdown.count(marker) >= least, marker
    for marker in ("**Who this is for.**", "**How to use this notebook.**", "**Roadmap:**", "## 10. Your turn — change one thing", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert marker in markdown, marker
    carried = [c for c in _cells() if c["cell_type"] == "code" and "embedded_module" in c.get("metadata", {}).get("dimer", {})]
    assert len(carried) == 3 and all(c["metadata"].get("jupyter", {}).get("source_hidden") for c in carried)


def test_no_doubled_braces_and_the_id_pattern_renders_correctly():
    markdown = _markdown()
    assert "{{" not in markdown and "}}" not in markdown
    assert "[A-Za-z0-9_.:-]{1,64}" in markdown and "{id, image, captions}" in markdown


def test_runtime_figures_name_their_environment():
    markdown = _markdown()
    assert "about fifteen minutes" not in markdown and "the build record measured" not in markdown
    for figure in ("879 s", "487 s", "32 s"):
        for match in re.finditer(re.escape(figure), markdown):
            window = markdown[max(0, match.start() - 400) : match.end() + 50]
            assert "CPU pre-flight" in window, figure
