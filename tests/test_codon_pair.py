"""Tests for the codon-pair-bias module (Coleman et al. 2008 CPS / CPB).

The core formula tests use a hand-constructed corpus where every count is known
on paper, plus a *marginal-preserving swap* that enriches one codon pair while
holding all single-codon and dipeptide counts fixed — the clean demonstration
that codon pair bias is independent of single-codon usage by construction.
"""

import math

import numpy as np
import pandas as pd
import pytest


def pytest_approx(value, rel=1e-6):
    """Approx comparison. abs=1e-6 matches the 6-decimal rounding applied to
    per-gene CPB columns (the CPS dict itself is full-precision)."""
    return pytest.approx(value, rel=rel, abs=1e-6)


from codonpipe.modules.codon_pair import (
    count_codon_pairs,
    compute_cps_table,
    compute_gene_cpb,
    compute_bridge_dinucleotide_bias,
    score_sequence_cpb,
    compare_cpb_by_expression_tier,
    run_codon_pair_analysis,
    _N_POSSIBLE_PAIRS,
)
from codonpipe.utils.codon_tables import CODON_TABLE_11


def _write(tmp_path, genes, name="t.ffn"):
    fa = tmp_path / name
    fa.write_text("\n".join(f">g{i}\n{s}" for i, s in enumerate(genes)))
    return fa


# ── Counting ─────────────────────────────────────────────────────────────────

class TestCounting:
    def test_hand_counts(self, tmp_path):
        # gene UUU-AAA-UUC: Phe Lys Phe
        fa = _write(tmp_path, ["UUUAAAUUC"])
        c = count_codon_pairs(fa, min_length=0)
        assert dict(c["codon_counts"]) == {"UUU": 1, "AAA": 1, "UUC": 1}
        assert dict(c["aa_counts"]) == {"Phe": 2, "Lys": 1}
        assert c["pair_counts"][("UUU", "AAA")] == 1
        assert c["pair_counts"][("AAA", "UUC")] == 1
        assert c["aa_pair_counts"][("Phe", "Lys")] == 1
        assert c["aa_pair_counts"][("Lys", "Phe")] == 1
        assert c["n_pairs"] == 2

    def test_stop_codon_excluded_from_pairs(self, tmp_path):
        # UUU-AAA-UAA(stop): the (AAA, stop) pair must not be counted.
        fa = _write(tmp_path, ["UUUAAAUAA"])
        c = count_codon_pairs(fa, min_length=0)
        assert c["n_pairs"] == 1
        assert c["pair_counts"][("UUU", "AAA")] == 1
        assert ("AAA", "UAA") not in c["pair_counts"]
        # stop codon contributes to neither codon nor aa counts
        assert "UAA" not in c["codon_counts"]
        assert "*" not in c["aa_counts"]

    def test_min_length_filter(self, tmp_path):
        fa = _write(tmp_path, ["UUUAAAUUC"])  # 9 nt
        c = count_codon_pairs(fa, min_length=240)
        assert c["n_genes"] == 0
        assert c["n_pairs"] == 0

    def test_nonstandard_codon_skipped(self, tmp_path):
        # NNN is not in the codon table; the pairs touching it are skipped.
        fa = _write(tmp_path, ["UUUNNNAAA"])
        c = count_codon_pairs(fa, min_length=0)
        assert c["pair_counts"].get(("UUU", "NNN")) is None
        assert c["pair_counts"].get(("NNN", "AAA")) is None
        assert c["n_pairs"] == 0
        assert dict(c["codon_counts"]) == {"UUU": 1, "AAA": 1}


# ── CPS formula ──────────────────────────────────────────────────────────────

def _balanced(k=10):
    return ["UUUAAA"] * k + ["UUUAAG"] * k + ["UUCAAA"] * k + ["UUCAAG"] * k


def _swapped(k=10):
    # Marginal-preserving swap vs balanced: +1 UUUAAA / -1 UUUAAG / -1 UUCAAA /
    # +1 UUCAAG. Single-codon and dipeptide counts are IDENTICAL to balanced;
    # only the four pair counts change.
    return ["UUUAAA"] * (k + 1) + ["UUUAAG"] * (k - 1) + \
           ["UUCAAA"] * (k - 1) + ["UUCAAG"] * (k + 1)


class TestCPSFormula:
    def test_balanced_corpus_cps_zero(self, tmp_path):
        fa = _write(tmp_path, _balanced())
        cps, _ = compute_cps_table(count_codon_pairs(fa, min_length=0))
        for pair in (("UUU", "AAA"), ("UUU", "AAG"), ("UUC", "AAA"), ("UUC", "AAG")):
            assert abs(cps[pair]) < 1e-9, (pair, cps[pair])

    def test_marginal_preserving_swap_isolates_pair_bias(self, tmp_path):
        bal = count_codon_pairs(_write(tmp_path, _balanced(), "bal.ffn"), min_length=0)
        swp = count_codon_pairs(_write(tmp_path, _swapped(), "swp.ffn"), min_length=0)
        # single-codon and dipeptide marginals are unchanged by the swap
        assert dict(bal["codon_counts"]) == dict(swp["codon_counts"])
        assert dict(bal["aa_counts"]) == dict(swp["aa_counts"])
        assert dict(bal["aa_pair_counts"]) == dict(swp["aa_pair_counts"])
        cps, _ = compute_cps_table(swp)
        # enriched pair (obs 11 vs expected 10) -> ln(1.1); depleted -> negative
        assert cps[("UUU", "AAA")] == pytest_approx(math.log(11 / 10))
        assert cps[("UUU", "AAG")] < 0
        assert cps[("UUC", "AAG")] == pytest_approx(math.log(11 / 10))
        assert cps[("UUC", "AAA")] < 0

    def test_cps_matches_formula_re_derivation(self, tmp_path):
        # Internal consistency on a varied corpus: module CPS == the Coleman
        # formula applied to the module's own counts, for every observed pair.
        rng = np.random.default_rng(0)
        codons = [c for c, aa in CODON_TABLE_11.items() if aa != "*"]
        genes = ["".join(rng.choice(codons, size=120)) for _ in range(40)]
        counts = count_codon_pairs(_write(tmp_path, genes), min_length=0)
        cc, ac = counts["codon_counts"], counts["aa_counts"]
        pc, apc = counts["pair_counts"], counts["aa_pair_counts"]
        cps, _ = compute_cps_table(counts)
        for (c1, c2), n_ab in pc.items():
            a1, a2 = CODON_TABLE_11[c1], CODON_TABLE_11[c2]
            exp = (cc[c1] * cc[c2] / (ac[a1] * ac[a2])) * apc[(a1, a2)]
            assert cps[(c1, c2)] == pytest_approx(math.log(n_ab / exp))

    def test_table_only_observed_pairs(self, tmp_path):
        fa = _write(tmp_path, _balanced())
        counts = count_codon_pairs(fa, min_length=0)
        cps, cps_df = compute_cps_table(counts)
        assert len(cps) == len(counts["pair_counts"])
        assert len(cps_df) == len(cps)
        assert len(cps) <= _N_POSSIBLE_PAIRS


# ── Per-gene CPB ─────────────────────────────────────────────────────────────

class TestGeneCPB:
    def test_single_pair_gene_cpb_equals_cps(self, tmp_path):
        fa = _write(tmp_path, _swapped())
        cps, _ = compute_cps_table(count_codon_pairs(fa, min_length=0))
        gene_cpb = compute_gene_cpb(fa, cps, min_length=0)
        g0 = gene_cpb.iloc[0]  # UUUAAA
        assert g0["cpb"] == pytest_approx(cps[("UUU", "AAA")])
        assert g0["frac_scored"] == 1.0
        assert g0["n_pairs"] == 1

    def test_cpb_is_mean_of_pair_scores(self, tmp_path):
        # A 3-codon gene: CPB = mean of its two pair scores.
        corpus = _swapped() + ["UUUAAAUUC"] * 3  # ensure all pairs observed
        fa = _write(tmp_path, corpus)
        counts = count_codon_pairs(fa, min_length=0)
        cps, _ = compute_cps_table(counts)
        gene_cpb = compute_gene_cpb(fa, cps, min_length=0)
        # find a 3-codon gene
        long_gene = gene_cpb[gene_cpb["n_pairs"] == 2].iloc[0]
        expected = np.mean([cps[("UUU", "AAA")], cps[("AAA", "UUC")]])
        assert long_gene["cpb"] == pytest_approx(expected)


# ── Foreign-gene scoring (engineering use) ───────────────────────────────────

class TestForeignScoring:
    def _host_table(self, tmp_path):
        # host uses UUU-AAA but never GGG-CCC
        fa = _write(tmp_path, _swapped())
        return compute_cps_table(count_codon_pairs(fa, min_length=0))[0]

    def test_unobserved_pair_skipped_and_counted(self, tmp_path):
        cps = self._host_table(tmp_path)
        res = score_sequence_cpb("GGGCCC", cps)  # pair absent from host
        assert res["n_pairs"] == 1
        assert res["n_unobserved"] == 1
        assert res["n_scored"] == 0
        assert np.isnan(res["cpb"])
        assert res["frac_scored"] == 0.0

    def test_observed_pair_scored(self, tmp_path):
        cps = self._host_table(tmp_path)
        res = score_sequence_cpb("UUUAAA", cps)
        assert res["n_scored"] == 1
        assert res["frac_scored"] == 1.0
        assert res["cpb"] == pytest_approx(cps[("UUU", "AAA")])

    def test_unobserved_penalty_applied(self, tmp_path):
        cps = self._host_table(tmp_path)
        res = score_sequence_cpb("GGGCCC", cps, unobserved_penalty=-5.0)
        assert res["n_scored"] == 1
        assert res["cpb"] == pytest_approx(-5.0)


# ── Bridge dinucleotide (the CPB confounder) ─────────────────────────────────

class TestBridgeDinucleotide:
    def test_sixteen_rows_and_flags(self, tmp_path):
        fa = _write(tmp_path, _swapped())
        counts = count_codon_pairs(fa, min_length=0)
        b = compute_bridge_dinucleotide_bias(counts["pair_counts"])
        assert len(b) == 16
        assert int(b["is_CpG"].sum()) == 1
        assert int(b["is_UpA"].sum()) == 1
        cpg = b.loc[b["is_CpG"]].iloc[0]
        assert cpg["dinucleotide"] == "CG"

    def test_absent_cpg_bridge_has_zero_rho(self, tmp_path):
        # CpG junction avoided but its marginals present: C occurs at junction
        # position 1 (UUC->...) and G at junction position 2 (...->GUU), yet C is
        # never directly followed by G, so observed CpG = 0 while expected > 0,
        # giving rho = 0 (the scientifically meaningful "avoided" case).
        # UUC|AAA junction = CA; AAA|GUU junction = AG.
        fa = _write(tmp_path, ["UUCAAA", "AAAGUU"] * 10)
        counts = count_codon_pairs(fa, min_length=0)
        b = compute_bridge_dinucleotide_bias(counts["pair_counts"])
        cpg = b.loc[b["is_CpG"]].iloc[0]
        assert cpg["observed"] == 0
        assert cpg["exp_freq"] > 0          # marginals present, so expected > 0
        assert cpg["rho"] == 0.0            # avoided: obs/exp = 0

    def test_undefined_rho_when_marginal_absent(self, tmp_path):
        # When a junction base is entirely absent (no C ever at position 1),
        # expected is 0 too and rho is genuinely undefined -> NaN, not 0.
        fa = _write(tmp_path, ["UUUAAA", "AAAUUU"] * 10)  # no C at junction pos 1
        counts = count_codon_pairs(fa, min_length=0)
        b = compute_bridge_dinucleotide_bias(counts["pair_counts"])
        cpg = b.loc[b["is_CpG"]].iloc[0]
        assert cpg["observed"] == 0
        assert cpg["exp_freq"] == 0.0
        assert np.isnan(cpg["rho"])

    def test_rho_is_obs_over_exp(self, tmp_path):
        fa = _write(tmp_path, _swapped())
        counts = count_codon_pairs(fa, min_length=0)
        b = compute_bridge_dinucleotide_bias(counts["pair_counts"]).dropna(subset=["rho"])
        for r in b.itertuples():
            if r.exp_freq > 0:
                assert r.rho == pytest_approx(r.obs_freq / r.exp_freq, rel=1e-3)


# ── Expression-tier comparison ───────────────────────────────────────────────

class TestExpressionTier:
    def test_runs_and_reports_columns(self):
        rng = np.random.default_rng(1)
        n = 60
        gene_cpb = pd.DataFrame({
            "gene": [f"g{i}" for i in range(n)],
            "cpb": np.concatenate([rng.normal(0.1, 0.05, n // 2),
                                   rng.normal(-0.1, 0.05, n // 2)]),
        })
        expr = pd.DataFrame({
            "gene": [f"g{i}" for i in range(n)],
            "expression_class": ["high"] * (n // 2) + ["low"] * (n // 2),
        })
        out = compare_cpb_by_expression_tier(gene_cpb, expr)
        assert not out.empty
        for col in ("cliffs_delta", "p_value", "n_high", "n_low", "caveat"):
            assert col in out.columns

    def test_too_small_tiers_returns_empty(self):
        gene_cpb = pd.DataFrame({"gene": ["a", "b", "c"], "cpb": [0.1, 0.0, -0.1]})
        expr = pd.DataFrame({"gene": ["a", "b", "c"],
                             "expression_class": ["high", "low", "high"]})
        assert compare_cpb_by_expression_tier(gene_cpb, expr).empty


# ── Integration ──────────────────────────────────────────────────────────────

def _realistic_corpus(n_genes=40, codons_per_gene=100, seed=0):
    rng = np.random.default_rng(seed)
    codons = [c for c, aa in CODON_TABLE_11.items() if aa != "*"]
    return ["".join(rng.choice(codons, size=codons_per_gene)) for _ in range(n_genes)]


class TestRunCodonPairAnalysis:
    def test_emits_expected_files(self, tmp_path):
        fa = _write(tmp_path, _realistic_corpus())
        out = run_codon_pair_analysis(fa, tmp_path, "S", make_figures=False)
        assert "cps_table" in out
        assert "gene_cpb" in out
        assert "bridge_dinucleotide" in out
        assert "summary" in out
        for key, path in out.items():
            assert path.exists(), key
        summ = pd.read_csv(out["summary"], sep="\t")
        for col in ("mean_gene_cpb", "n_distinct_pairs_observed",
                    "frac_pairs_observed", "bridge_CpG_rho", "bridge_UpA_rho"):
            assert col in summ.columns

    def test_with_expression_tiers(self, tmp_path):
        corpus = _realistic_corpus(n_genes=60)
        fa = _write(tmp_path, corpus)
        expr = pd.DataFrame({
            "gene": [f"g{i}" for i in range(60)],
            "expression_class": ["high"] * 30 + ["low"] * 30,
        })
        out = run_codon_pair_analysis(fa, tmp_path, "S", expr_df=expr, make_figures=False)
        assert "expression_tiers" in out
        assert out["expression_tiers"].exists()

    def test_empty_input_returns_empty(self, tmp_path):
        fa = _write(tmp_path, ["AUG"])  # below min length, no pairs
        out = run_codon_pair_analysis(fa, tmp_path, "S", make_figures=False)
        assert out == {}
