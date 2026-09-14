"""
Regression tests for the MEE-review feature-order bug (fixed 2026-08-31).

Background: match_dataset() in taxonmatch/matching.py used to build its own hardcoded
`relevant_features` list, which had drifted out of sync with the column order that
compute_similarity_metrics() actually produces during training. Because the shipped
XGBoost model has no feature names, it scores its input positionally, so the mismatch
silently dropped real-world accuracy from ~0.97 to ~0.82 without raising any error.

These tests pin down the fix from multiple angles so a future refactor can't
reintroduce the same silent failure:
  1. The column order stored in the shipped training_set.txt matches the canonical
     SIMILARITY_FEATURE_ORDER.
  2. generate_training_test()'s guardrail assertion actually fires on drift.
  3. match_dataset() is wired to the canonical order (not a private hardcoded list).
  4. End-to-end: the shipped model, given the canonical column order, reproduces the
     paper's reported ~0.97-0.98 accuracy on the shipped training set -- not the ~0.82
     the bug produced.
"""

import inspect
from pathlib import Path

import pandas as pd
import pytest

from taxonmatch import matching, model_training as mt
from taxonmatch.loader import load_training_set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "taxonmatch" / "files" / "models" / "xgb_model.json"

# The old, buggy hardcoded order that used to live in matching.match_dataset().
OLD_BUGGY_ORDER = [
    'rank_similarity', 'levenshtein_distance', 'damerau_levenshtein_distance', 'ratio',
    'q_ratio', 'token_sort_ratio', 'w_ratio', 'token_set_ratio', 'jaro_winkler_similarity',
    'partial_ratio', 'hamming_distance', 'jaro_similarity'
]


def test_old_buggy_order_is_a_real_permutation_not_the_same_order():
    # Sanity check on the regression fixture itself: same columns, different order.
    assert set(OLD_BUGGY_ORDER) == set(mt.SIMILARITY_FEATURE_ORDER)
    assert OLD_BUGGY_ORDER != mt.SIMILARITY_FEATURE_ORDER


def test_training_set_column_order_matches_canonical_order():
    df = load_training_set()
    feature_columns = [
        c for c in df.columns
        if c not in ["query_name", "target_name", "taxonRank", "ncbi_rank", "match"]
    ]
    assert feature_columns == mt.SIMILARITY_FEATURE_ORDER


def test_generate_training_test_rejects_column_order_drift():
    # Build a tiny synthetic df_output whose feature columns are deliberately shuffled
    # relative to SIMILARITY_FEATURE_ORDER, and confirm the guardrail catches it.
    n = 10
    data = {"query_name": ["a"] * n, "target_name": ["b"] * n,
            "taxonRank": ["genus"] * n, "ncbi_rank": ["genus"] * n,
            "match": [0, 1] * (n // 2)}
    shuffled = list(reversed(mt.SIMILARITY_FEATURE_ORDER))
    for col in shuffled:
        data[col] = [0.5] * n
    df_output = pd.DataFrame(data)

    with pytest.raises(AssertionError):
        mt.generate_training_test(df_output, test_size=0.3, random_state=0)


def test_match_dataset_is_wired_to_canonical_feature_order():
    # Guard against someone reintroducing a private hardcoded feature list in
    # match_dataset() instead of using the single source of truth. Uses the AST (not a
    # substring check) so a leftover comment mentioning SIMILARITY_FEATURE_ORDER can't
    # make this test pass while the actual code has reverted to a hardcoded list.
    import ast

    source = inspect.getsource(matching.match_dataset)
    tree = ast.parse(source)

    assign_node = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "relevant_features" for t in node.targets)),
        None,
    )
    assert assign_node is not None, "no `relevant_features = ...` assignment found in match_dataset()"

    assert not isinstance(assign_node.value, ast.List), (
        "match_dataset() assigns relevant_features from a hardcoded list literal again -- "
        "this is exactly the pattern that caused the original column-order bug. It must "
        "reference model_training.SIMILARITY_FEATURE_ORDER instead."
    )

    names_used = {n.id for n in ast.walk(assign_node.value) if isinstance(n, ast.Name)}
    assert "SIMILARITY_FEATURE_ORDER" in names_used, (
        "relevant_features in match_dataset() no longer derives from SIMILARITY_FEATURE_ORDER."
    )


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="shipped model file not found")
def test_shipped_model_reproduces_reported_accuracy_with_canonical_order():
    from xgboost import XGBClassifier

    df = load_training_set()
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    y = df["match"].values
    X_correct = df[mt.SIMILARITY_FEATURE_ORDER].values
    X_buggy = df[OLD_BUGGY_ORDER].values

    acc_correct = (model.predict(X_correct) == y).mean()
    acc_buggy = (model.predict(X_buggy) == y).mean()

    # Paper reports ~0.97 cross-val accuracy; the bug measured ~0.82. Assert we're
    # solidly back in the correct regime, and that the buggy order is measurably worse
    # (proving this test would have caught the original regression).
    assert acc_correct > 0.95, f"expected ~0.97-0.98 accuracy with canonical order, got {acc_correct:.4f}"
    assert acc_buggy < acc_correct - 0.1, (
        f"expected the old buggy column order to score markedly worse "
        f"(correct={acc_correct:.4f}, buggy={acc_buggy:.4f})"
    )
