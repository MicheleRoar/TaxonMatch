from pathlib import Path


def test_import_taxonmatch():
    import taxonmatch

    assert taxonmatch is not None


def test_import_main_modules():
    from taxonmatch import (
        analysis_utils,
        downloader,
        loader,
        matching,
        model_training,
        tree_utils,
    )

    assert analysis_utils is not None
    assert downloader is not None
    assert loader is not None
    assert matching is not None
    assert model_training is not None
    assert tree_utils is not None


def test_model_file_exists():
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "taxonmatch"
        / "files"
        / "models"
        / "xgb_model.json"
    )

    assert model_path.exists(), f"Model file not found: {model_path}"


def test_training_set_exists():
    project_root = Path(__file__).resolve().parents[1]

    training_set_path = (
        project_root
        / "taxonmatch"
        / "files"
        / "training_set"
        / "training_set.txt"
    )

    assert training_set_path.exists(), (
        f"Training set not found: {training_set_path}"
    )
