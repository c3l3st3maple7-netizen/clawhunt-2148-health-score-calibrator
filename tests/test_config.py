"""Configuration validation: every rejected case names the offending key."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from calib import miniyaml
from calib.config import ConfigError, config_from_dict, load_config


@pytest.fixture()
def document(project_root: Path) -> dict:
    return copy.deepcopy(miniyaml.load(str(project_root / "configs" / "default.yaml")))


def test_default_config_values(default_config) -> None:
    assert default_config.seed == 20260158
    assert default_config.dataset.companies == 300
    assert default_config.dataset.sites == 3
    assert default_config.dataset.months == 24
    assert default_config.window_months == 12
    assert default_config.dataset.label_sampling == "bernoulli"
    assert set(default_config.true_weights) == {
        "event",
        "interaction",
        "resource",
        "opportunity",
        "space",
    }
    assert default_config.missing_data == default_config.missing_data.__class__(
        site_index=2, start_month_index=12, months=3
    )


def test_unknown_root_key(document: dict) -> None:
    document["surprise"] = 1
    with pytest.raises(ConfigError, match="unknown key.*surprise"):
        config_from_dict(document)


def test_unknown_dataset_key(document: dict) -> None:
    document["dataset"]["surprise"] = 1
    with pytest.raises(ConfigError, match="unknown key.*surprise"):
        config_from_dict(document)


def test_missing_required_value(document: dict) -> None:
    del document["dataset"]["companies"]
    with pytest.raises(ConfigError, match="companies must be an integer"):
        config_from_dict(document)


@pytest.mark.parametrize("target", [0.0, 0.6, -0.1])
def test_churn_target_range(document: dict, target: float) -> None:
    document["dataset"]["annual_churn_target"] = target
    with pytest.raises(ConfigError):
        config_from_dict(document)


def test_months_must_cover_a_year(document: dict) -> None:
    document["dataset"]["months"] = 11
    with pytest.raises(ConfigError, match="months must be between 12"):
        config_from_dict(document)


def test_missing_window_must_fit(document: dict) -> None:
    document["missing_data"]["start_month_index"] = 22
    with pytest.raises(ConfigError, match="does not fit"):
        config_from_dict(document)


def test_missing_site_must_exist(document: dict) -> None:
    document["missing_data"]["site_index"] = 3
    with pytest.raises(ConfigError, match="outside 0..2"):
        config_from_dict(document)


def test_attribute_weights_are_normalised(document: dict) -> None:
    document["attributes"]["industry"] = {"a": 2, "b": 2}
    config = config_from_dict(document)
    assert dict(config.attributes["industry"]) == {"a": 0.5, "b": 0.5}


def test_attribute_weights_must_be_positive(document: dict) -> None:
    document["attributes"]["tier"] = {"basic": 1, "plus": 0}
    with pytest.raises(ConfigError, match="greater than 0"):
        config_from_dict(document)


def test_unknown_label_sampling(document: dict) -> None:
    document["dataset"]["label_sampling"] = "quota"
    with pytest.raises(ConfigError, match="label_sampling must be one of"):
        config_from_dict(document)


def test_score_saturation_is_rejected(document: dict) -> None:
    document["standardisation"]["score_sd"] = 40.0
    with pytest.raises(ConfigError, match="must not exceed 100"):
        config_from_dict(document)


def test_explosive_latent_walk_is_rejected(document: dict) -> None:
    document["latent_activity"]["phi"] = 1.0
    with pytest.raises(ConfigError, match="phi must be"):
        config_from_dict(document)


def test_duplicate_seasonal_month_is_rejected(document: dict) -> None:
    document["latent_activity"]["seasonal_months"] = [1, 1]
    with pytest.raises(ConfigError, match="duplicates"):
        config_from_dict(document)


def test_missing_out_dir_falls_back_to_default(document: dict) -> None:
    del document["out_dir"]
    assert config_from_dict(document).out_dir == "runs/default"


def test_unreadable_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(str(tmp_path / "absent.yaml"))


def test_yaml_error_is_wrapped(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("a: 1\na: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(str(path))
