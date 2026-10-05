"""The stale-figure checker, including its strict mode for the Devpost writeup.

The writeup is the text judges read first, so in it every decimal figure must
carry a citation (an uncited one fails), placeholders are counted, and the
pre-submission mode fails while any remain.
"""

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_doc_numbers", ROOT / "tools" / "check_doc_numbers.py")
cdn = importlib.util.module_from_spec(SPEC)
sys.modules["check_doc_numbers"] = cdn
SPEC.loader.exec_module(cdn)


def setup(tmp_path, monkeypatch, text, data=None):
    (tmp_path / "bench" / "out").mkdir(parents=True)
    (tmp_path / "bench" / "out" / "x.json").write_text(json.dumps(data or {"a": 0.611, "b": [12.5, 13.75]}))
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs" / "devpost-writeup.md"
    doc.write_text(text)
    monkeypatch.setattr(cdn, "ROOT", tmp_path)
    monkeypatch.setattr(cdn, "BENCH_OUT", tmp_path / "bench" / "out")
    return doc


def test_writeup_is_always_in_the_checked_set():
    assert "docs/devpost-writeup.md" in cdn.REQUIRED_DOCS
    assert "docs/devpost-writeup.md" in cdn.STRICT_DOCS


def test_cited_figures_pass_in_strict_mode(tmp_path, monkeypatch):
    doc = setup(tmp_path, monkeypatch,
                "Baseline 0.611 <!--bench:x.a--> and 12.50 <!--bench:x.b[0]-->–13.75 <!--bench:x.b[1]--> ms.")
    assert cdn.check([doc], strict={doc}) == []


def test_an_uncited_decimal_fails_in_strict_mode(tmp_path, monkeypatch):
    doc = setup(tmp_path, monkeypatch, "Baseline 0.611 <!--bench:x.a--> but also 4.12x from memory.")
    problems = cdn.check([doc], strict={doc})
    assert any("4.12" in p and "uncited" in p for p in problems)


def test_one_uncited_end_of_a_range_fails(tmp_path, monkeypatch):
    doc = setup(tmp_path, monkeypatch, "12.50 <!--bench:x.b[0]-->–13.75 ms")
    assert any("13.75" in p for p in cdn.check([doc], strict={doc}))


def test_code_spans_and_versions_are_not_figures(tmp_path, monkeypatch):
    doc = setup(tmp_path, monkeypatch, "Run `uv run x --value 12.5` with H.264 on OpenCV 5.")
    assert cdn.check([doc], strict={doc}) == []


def test_non_strict_docs_allow_uncited_decimals(tmp_path, monkeypatch):
    doc = setup(tmp_path, monkeypatch, "Roughly 4.12x.")
    assert cdn.check([doc], strict=set()) == []


def test_placeholders_are_counted_and_fail_only_for_submission(tmp_path, monkeypatch):
    doc = setup(tmp_path, monkeypatch, "- [TBD — later]\n- [update after W6]\n")
    assert cdn.placeholders([doc]) == 2
    assert cdn.main_with([str(doc)], submission=False) == 0
    assert cdn.main_with([str(doc)], submission=True) == 1


def test_a_figure_at_the_end_of_a_sentence_is_still_a_figure(tmp_path, monkeypatch):
    """Found on the real draft: '... 10.94.' and '... 0.295.' slipped through."""
    doc = setup(tmp_path, monkeypatch, "JPEG quality 7.97 <!--bench:x.a-->–10.94. mAP falls to 0.295.")
    problems = cdn.check([doc], strict={doc})
    assert any("10.94" in p for p in problems) and any("0.295" in p for p in problems)
