"""Tests for the PDTB / TED-MDB pipe parser."""

from __future__ import annotations

import pytest

from long_context_discourse.preprocess.pipe import _parse_span_list, parse_pipe_rows


def test_parse_span_list_simple() -> None:
    assert _parse_span_list("9..35") == ((9, 35),)


def test_parse_span_list_discontinuous() -> None:
    assert _parse_span_list("9..35;195..239") == ((9, 35), (195, 239))


def test_parse_span_list_handles_garbage() -> None:
    assert _parse_span_list("") == ()
    assert _parse_span_list("not-a-span") == ()
    assert _parse_span_list("9..a") == ()


def test_parse_pipe_rows_drops_invalid() -> None:
    raw = "X" * 200 + "ARG1 TEXT" + "X" * 50 + "ARG2 TEXT" + "X" * 100
    arg1_span = f"{200}..{200 + len('ARG1 TEXT')}"
    arg2_span = f"{200 + len('ARG1 TEXT') + 50}..{200 + len('ARG1 TEXT') + 50 + len('ARG2 TEXT')}"
    cols_explicit = [""] * 30
    cols_explicit[0] = "Explicit"
    cols_explicit[8] = "Comparison.Concession.Arg2-as-denier"
    cols_explicit[14] = arg1_span
    cols_explicit[20] = arg2_span
    line_explicit = "|".join(cols_explicit)

    cols_entrel = [""] * 30
    cols_entrel[0] = "EntRel"
    cols_entrel[8] = "Comparison.Concession.Arg2-as-denier"
    cols_entrel[14] = arg1_span
    cols_entrel[20] = arg2_span
    line_entrel = "|".join(cols_entrel)

    pipe_text = "\n".join([line_explicit, line_entrel])
    rels = parse_pipe_rows(pipe_text, raw, document_id="doc")
    assert len(rels) == 1
    rel = rels[0]
    assert rel.rel_type == "Explicit"
    assert rel.sense_l1 == "Comparison"
    assert rel.arg1 == "ARG1 TEXT"
    assert rel.arg2 == "ARG2 TEXT"


def test_parse_pipe_rows_skips_unsupported_sense() -> None:
    cols = [""] * 30
    cols[0] = "Implicit"
    cols[8] = "Hypophora"  # not a Level-1 sense in our taxonomy
    cols[14] = "0..3"
    cols[20] = "4..7"
    rels = parse_pipe_rows("|".join(cols), "ABCDEFGH", document_id="doc")
    assert rels == []
