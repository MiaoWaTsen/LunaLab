"""
Tests for the interface translations (frontend/i18n/messages.json).

The file is a table: one row per English source string, one column per
language, so every language has the same keys by construction.  These
tests check what the structure cannot: that no cell is empty and that a
translation keeps the placeholders of its source, since the engine
(frontend/i18n.js) refuses a translation whose placeholders differ.
"""

import glob
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGUAGES = ["en", "zh-TW", "zh-CN", "ja", "de", "fr"]


@pytest.fixture(scope="module")
def table():
    with open(os.path.join(ROOT, "frontend", "i18n", "messages.json"), encoding="utf-8") as f:
        return json.load(f)


def marks(text):
    """The placeholders in a message: interpolated values and inline elements."""
    return (
        sorted(re.findall(r"\{\d+\}", text)),
        sorted(re.findall(r"<\d+/>", text)),
        sorted(re.findall(r"<\d+>", text)),
        sorted(re.findall(r"</\d+>", text)),
    )


def test_languages_are_the_ones_the_selector_offers(table):
    assert table["languages"] == LANGUAGES
    with open(os.path.join(ROOT, "frontend", "i18n.js"), encoding="utf-8") as f:
        offered = re.findall(r"\{ code: '([^']+)'", f.read())
    assert offered == LANGUAGES


def test_every_row_has_a_cell_per_language(table):
    assert table["messages"], "no messages"
    for source, row in table["messages"].items():
        assert len(row) == len(LANGUAGES), source


def test_no_translation_is_missing_or_empty(table):
    for source, row in table["messages"].items():
        for language, cell in zip(LANGUAGES[1:], row[1:]):
            assert isinstance(cell, str) and cell.strip(), f"[{language}] {source}"


def test_english_column_is_only_for_non_english_sources(table):
    # English is the source language: a null cell means "as written"
    for source, row in table["messages"].items():
        assert row[0] is None or row[0] != source, source


def test_translations_keep_their_placeholders(table):
    for source, row in table["messages"].items():
        for language, cell in zip(LANGUAGES, row):
            if cell is not None:
                assert marks(cell) == marks(source), f"[{language}] {source}"


def test_inline_placeholders_are_well_formed(table):
    for source, row in table["messages"].items():
        for cell in [source, *[c for c in row if c is not None]]:
            depth = []
            for close, number in re.findall(r"<(/?)(\d+)>", cell):
                if close:
                    assert depth and depth.pop() == number, cell
                else:
                    depth.append(number)
            assert not depth, cell


def test_keys_are_normalised_the_way_the_engine_reads_them(table):
    for source in table["messages"]:
        assert source == " ".join(source.split()), repr(source)


def test_chinese_columns_use_their_own_script(table):
    # Pairs of characters that differ between the two scripts
    pairs = ["擊击", "體体", "學学", "進进", "這这", "個个", "們们", "為为", "與与", "實实", "驗验",
             "結结", "構构", "說说", "選选", "項项", "關关", "開开", "數数", "據据", "載载", "執执"]
    traditional_only = {pair[0] for pair in pairs}
    simplified_only = {pair[1] for pair in pairs}
    for source, row in table["messages"].items():
        assert not (set(row[1]) & simplified_only), f"Simplified characters in zh-TW: {source}"
        assert not (set(row[2]) & traditional_only), f"Traditional characters in zh-CN: {source}"


def test_every_page_loads_the_translation_script_before_app_js():
    pages = sorted(glob.glob(os.path.join(ROOT, "frontend", "*.html")))
    assert len(pages) >= 11
    for page in pages:
        with open(page, encoding="utf-8") as f:
            html = f.read()
        i18n, app = html.find("/static/i18n.js"), html.find("/static/app.js")
        assert 0 <= i18n < app, os.path.basename(page)
