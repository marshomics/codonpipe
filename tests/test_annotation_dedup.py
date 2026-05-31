"""Gene-ID deduplication for external (NCBI/RefSeq) annotation files.

External annotations can assign one locus tag to two CDS — genes with a
programmed ribosomal frameshift or stop-codon readthrough emit two products
under a single tag (dnaX/b0470 -> tau + gamma; prfB). Duplicate IDs crash
KofamScan ("Non-unique query name") and become duplicate pandas index labels
downstream (Cartesian merge blow-up, ragged .loc arrays, gene-track length
mismatches). These tests lock in that duplicates are detected and collapsed to
the longest record per ID, and that clean (Prokka) input is passed through
untouched.
"""

import tempfile
from pathlib import Path

from codonpipe.modules.prokka import (
    _dedupe_fasta_keep_longest,
    _fasta_has_duplicate_ids,
    deduplicate_annotation_files,
)


def _ids(path: Path) -> list[str]:
    return [ln[1:].split()[0] for ln in open(path) if ln.startswith(">")]


def _write(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")


class TestDuplicateDetection:
    def test_detects_duplicates(self, tmp_path):
        p = tmp_path / "dup.faa"
        _write(p, [("b1", "AAA"), ("b2", "CCC"), ("b1", "GGG")])
        assert _fasta_has_duplicate_ids(p) is True

    def test_unique_is_clean(self, tmp_path):
        p = tmp_path / "uniq.faa"
        _write(p, [("b1", "AAA"), ("b2", "CCC"), ("b3", "GGG")])
        assert _fasta_has_duplicate_ids(p) is False

    def test_id_is_first_header_token(self, tmp_path):
        # "b1 description" and "b1 other" share the ID b1.
        p = tmp_path / "tok.faa"
        with open(p, "w") as fh:
            fh.write(">b1 dnaA\nAAA\n>b1 dnaA isoform\nGGG\n")
        assert _fasta_has_duplicate_ids(p) is True


class TestDedupeKeepLongest:
    def test_keeps_longest_and_preserves_order(self, tmp_path):
        src = tmp_path / "src.ffn"
        dst = tmp_path / "dst.ffn"
        # b0470 appears twice: the longer record (tau) must survive.
        _write(src, [
            ("b0001", "ATGAAA"),
            ("b0470", "ATG" * 4),       # 12 nt
            ("b0002", "ATGCCC"),
            ("b0470", "ATG" * 10),      # 30 nt (longer -> keep this one)
        ])
        dropped = _dedupe_fasta_keep_longest(src, dst)
        assert dropped == ["b0470"]
        ids = _ids(dst)
        assert ids == ["b0001", "b0470", "b0002"]   # order preserved, unique
        # the 30-nt record was kept
        seqs = {}
        cur = None
        for ln in open(dst):
            if ln.startswith(">"):
                cur = ln[1:].split()[0]
                seqs[cur] = ""
            else:
                seqs[cur] += ln.strip()
        assert len(seqs["b0470"]) == 30

    def test_triple_duplicate(self, tmp_path):
        src = tmp_path / "s.faa"
        dst = tmp_path / "d.faa"
        _write(src, [("g", "A" * 5), ("g", "A" * 20), ("g", "A" * 9)])
        dropped = _dedupe_fasta_keep_longest(src, dst)
        assert dropped == ["g"]
        assert _ids(dst) == ["g"]
        assert sum(len(ln.strip()) for ln in open(dst) if not ln.startswith(">")) == 20


class TestDeduplicateAnnotationFiles:
    def test_noop_when_unique(self, tmp_path):
        faa = tmp_path / "x.faa"
        ffn = tmp_path / "x.ffn"
        _write(faa, [("b1", "MKV"), ("b2", "MQA")])
        _write(ffn, [("b1", "ATGAAA"), ("b2", "ATGCAA")])
        out_faa, out_ffn = deduplicate_annotation_files(faa, ffn, tmp_path / "out", "s")
        # Clean input is returned unchanged (no sanitized copy written).
        assert out_faa == faa
        assert out_ffn == ffn

    def test_rewrites_when_duplicates(self, tmp_path):
        faa = tmp_path / "x.faa"
        ffn = tmp_path / "x.ffn"
        _write(faa, [("b1", "MKV"), ("b1", "MK"), ("b2", "MQA")])
        _write(ffn, [("b1", "ATGAAAGTT"), ("b1", "ATGAAA"), ("b2", "ATGCAA")])
        out_faa, out_ffn = deduplicate_annotation_files(faa, ffn, tmp_path / "out", "s")
        assert out_faa != faa and out_ffn != ffn
        assert _fasta_has_duplicate_ids(out_faa) is False
        assert _fasta_has_duplicate_ids(out_ffn) is False
        assert set(_ids(out_faa)) == {"b1", "b2"}
        assert set(_ids(out_ffn)) == {"b1", "b2"}
