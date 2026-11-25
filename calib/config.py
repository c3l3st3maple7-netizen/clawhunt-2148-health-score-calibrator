"""Configuration schema, loading and validation for the generator.

Every value the generator uses comes from the YAML file (or an explicit CLI
override).  An unknown key, an out-of-range value or a malformed weight map is
a hard error: the run never falls back to a hidden default, so a configuration
review is enough to know what a run will produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from . import miniyaml

SUB_ITEMS: tuple[str, ...] = (
    "event",
    "interaction",
    "resource",
    "opportunity",
    "space",
)
ATTRIBUTES: tuple[str, ...] = ("industry", "stage", "tier")
LABEL_SAMPLING = ("exact_count", "bernoulli")

#: Every key understood in each section; anything else is rejected.
_SECTIONS = {
    "root": {
        "seed",
        "out_dir",
        "dataset",
        "attributes",
        "latent_activity",
        "sub_items",
        "true_weights",
        "standardisation",
        "missing_data",
    },
    "dataset": {
        "companies",
        "sites",
        "months",
        "start_year",
        "start_month",
        "annual_churn_target",
        "label_sampling",
    },
    "latent_activity": {
        "phi",
        "innovation_sd",
        "initial_sd",
        "seasonal_months",
        "seasonal_effect",
    },
    "sub_item": {"base", "loading", "noise_sd", "company_bias_sd"},
    "standardisation": {"clip_z", "score_mean", "score_sd"},
    "missing_data": {"site_index", "start_month_index", "months"},
}


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or out of range."""


@dataclass(frozen=True)
class DatasetConfig:
    companies: int = 300
    sites: int = 3
    months: int = 24
    start_year: int = 2024
    start_month: int = 1
    annual_churn_target: float = 0.12
    label_sampling: str = "bernoulli"


@dataclass(frozen=True)
class LatentConfig:
    phi: float = 0.85
    innovation_sd: float = 0.45
    initial_sd: float = 1.0
    seasonal_months: tuple[int, ...] = (1, 2)
    seasonal_effect: float = -0.70


@dataclass(frozen=True)
class SubItemConfig:
    base: float
    loading: float
    noise_sd: float
    company_bias_sd: float


@dataclass(frozen=True)
class StandardisationConfig:
    clip_z: float = 3.0
    score_mean: float = 50.0
    score_sd: float = 15.0


@dataclass(frozen=True)
class MissingDataConfig:
    site_index: int = 2
    start_month_index: int = 12
    months: int = 3


@dataclass(frozen=True)
class Config:
    seed: int
    out_dir: str
    dataset: DatasetConfig
    attributes: Mapping[str, tuple[tuple[str, float], ...]]
    latent: LatentConfig
    sub_items: Mapping[str, SubItemConfig]
    true_weights: Mapping[str, float]
    standardisation: StandardisationConfig
    missing_data: MissingDataConfig
    source_path: str | None = field(default=None, compare=False)

    # -- derived helpers -------------------------------------------------
    @property
    def window_months(self) -> int:
        """Length of the annual label window (12, or the whole run if shorter)."""
        return min(12, self.dataset.months)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form with every value the run actually used."""
        return {
            "seed": self.seed,
            "dataset": {
                "companies": self.dataset.companies,
                "sites": self.dataset.sites,
                "months": self.dataset.months,
                "start_year": self.dataset.start_year,
                "start_month": self.dataset.start_month,
                "annual_churn_target": self.dataset.annual_churn_target,
                "label_sampling": self.dataset.label_sampling,
            },
            "attributes": {
                name: {label: weight for label, weight in pairs}
                for name, pairs in self.attributes.items()
            },
            "latent_activity": {
                "phi": self.latent.phi,
                "innovation_sd": self.latent.innovation_sd,
                "initial_sd": self.latent.initial_sd,
                "seasonal_months": list(self.latent.seasonal_months),
                "seasonal_effect": self.latent.seasonal_effect,
            },
            "sub_items": {
                name: {
                    "base": spec.base,
                    "loading": spec.loading,
                    "noise_sd": spec.noise_sd,
                    "company_bias_sd": spec.company_bias_sd,
                }
                for name, spec in self.sub_items.items()
            },
            "true_weights": dict(self.true_weights),
            "standardisation": {
                "clip_z": self.standardisation.clip_z,
                "score_mean": self.standardisation.score_mean,
                "score_sd": self.standardisation.score_sd,
            },
            "missing_data": {
                "site_index": self.missing_data.site_index,
                "start_month_index": self.missing_data.start_month_index,
                "months": self.missing_data.months,
            },
        }


def load_config(path: str) -> Config:
    """Read and validate a YAML configuration file."""
    try:
        document = miniyaml.load(path)
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path!r}: {exc}") from exc
    except miniyaml.YAMLError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    config = config_from_dict(document)
    object.__setattr__(config, "source_path", path)
    return config


def config_from_dict(document: Any) -> Config:
    """Validate an already-parsed mapping and build a :class:`Config`."""
    root = _mapping(document, "config")
    _check_keys(root, _SECTIONS["root"], "config")

    dataset_doc = _mapping(root.get("dataset"), "dataset")
    _check_keys(dataset_doc, _SECTIONS["dataset"], "dataset")
    dataset = DatasetConfig(
        companies=_integer(dataset_doc, "companies", 1, 1_000_000),
        sites=_integer(dataset_doc, "sites", 1, 10_000),
        months=_integer(dataset_doc, "months", 12, 600),
        start_year=_integer(dataset_doc, "start_year", 1900, 2999),
        start_month=_integer(dataset_doc, "start_month", 1, 12),
        annual_churn_target=_number(dataset_doc, "annual_churn_target", 0.0, 0.5),
        label_sampling=_choice(dataset_doc, "label_sampling", LABEL_SAMPLING),
    )
    if dataset.annual_churn_target <= 0.0:
        raise ConfigError("dataset.annual_churn_target must be greater than 0")
    if dataset.sites > dataset.companies:
        raise ConfigError("dataset.sites cannot exceed dataset.companies")

    attributes_doc = _mapping(root.get("attributes"), "attributes")
    _check_keys(attributes_doc, set(ATTRIBUTES), "attributes")
    attributes: dict[str, tuple[tuple[str, float], ...]] = {}
    for name in ATTRIBUTES:
        weights_doc = _mapping(attributes_doc.get(name), f"attributes.{name}")
        if not weights_doc:
            raise ConfigError(f"attributes.{name} must list at least one category")
        pairs: list[tuple[str, float]] = []
        for label, weight in weights_doc.items():
            if not isinstance(label, str) or not label.strip():
                raise ConfigError(f"attributes.{name}: category names must be non-empty strings")
            pairs.append((label, _positive_number(weight, f"attributes.{name}.{label}")))
        total = sum(weight for _, weight in pairs)
        attributes[name] = tuple((label, weight / total) for label, weight in pairs)

    latent_doc = _mapping(root.get("latent_activity"), "latent_activity")
    _check_keys(latent_doc, _SECTIONS["latent_activity"], "latent_activity")
    phi = _number(latent_doc, "phi", -1.0, 0.999)
    if abs(phi) >= 1.0:
        raise ConfigError("latent_activity.phi must satisfy |phi| < 1 for a stationary walk")
    seasonal_months = _integer_list(latent_doc, "seasonal_months", 1, 12)
    latent = LatentConfig(
        phi=phi,
        innovation_sd=_positive_number(latent_doc.get("innovation_sd"), "latent_activity.innovation_sd"),
        initial_sd=_positive_number(latent_doc.get("initial_sd"), "latent_activity.initial_sd"),
        seasonal_months=seasonal_months,
        seasonal_effect=_number(latent_doc, "seasonal_effect", -5.0, 5.0),
    )

    sub_items_doc = _mapping(root.get("sub_items"), "sub_items")
    _check_keys(sub_items_doc, set(SUB_ITEMS), "sub_items")
    sub_items: dict[str, SubItemConfig] = {}
    for name in SUB_ITEMS:
        item_doc = _mapping(sub_items_doc.get(name), f"sub_items.{name}")
        _check_keys(item_doc, _SECTIONS["sub_item"], f"sub_items.{name}")
        base = _number(item_doc, "base", 0.0, 1e9)
        sub_items[name] = SubItemConfig(
            base=base,
            loading=_positive_number(item_doc.get("loading"), f"sub_items.{name}.loading"),
            noise_sd=_positive_number(item_doc.get("noise_sd"), f"sub_items.{name}.noise_sd"),
            company_bias_sd=_non_negative_number(
                item_doc.get("company_bias_sd"), f"sub_items.{name}.company_bias_sd"
            ),
        )

    weights_doc = _mapping(root.get("true_weights"), "true_weights")
    _check_keys(weights_doc, set(SUB_ITEMS), "true_weights")
    true_weights = {
        name: _positive_number(weights_doc.get(name), f"true_weights.{name}")
        for name in SUB_ITEMS
    }

    standardisation_doc = _mapping(root.get("standardisation"), "standardisation")
    _check_keys(standardisation_doc, _SECTIONS["standardisation"], "standardisation")
    standardisation = StandardisationConfig(
        clip_z=_positive_number(
            standardisation_doc.get("clip_z"), "standardisation.clip_z"
        ),
        score_mean=_number(standardisation_doc, "score_mean", 0.0, 100.0),
        score_sd=_positive_number(
            standardisation_doc.get("score_sd"), "standardisation.score_sd"
        ),
    )
    if standardisation.clip_z > 6.0:
        raise ConfigError("standardisation.clip_z above 6 offers no protection against outliers")
    if standardisation.score_sd * standardisation.clip_z > 100.0:
        raise ConfigError(
            "standardisation.score_sd * clip_z must not exceed 100 points, "
            "otherwise every score saturates at 0 or 100"
        )

    missing_doc = _mapping(root.get("missing_data"), "missing_data")
    _check_keys(missing_doc, _SECTIONS["missing_data"], "missing_data")
    missing = MissingDataConfig(
        site_index=_integer(missing_doc, "site_index", 0, 10_000),
        start_month_index=_integer(missing_doc, "start_month_index", 0, 10_000),
        months=_integer(missing_doc, "months", 1, 120),
    )
    if missing.site_index >= dataset.sites:
        raise ConfigError(
            f"missing_data.site_index {missing.site_index} is outside 0..{dataset.sites - 1}"
        )
    if missing.start_month_index + missing.months > dataset.months:
        raise ConfigError(
            "missing_data window "
            f"[{missing.start_month_index}, {missing.start_month_index + missing.months}) "
            f"does not fit in dataset.months={dataset.months}"
        )

    seed = root.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("seed must be a non-negative integer")
    out_dir = root.get("out_dir", "runs/default")
    if not isinstance(out_dir, str) or not out_dir.strip():
        raise ConfigError("out_dir must be a non-empty string")

    return Config(
        seed=seed,
        out_dir=out_dir,
        dataset=dataset,
        attributes=attributes,
        latent=latent,
        sub_items=sub_items,
        true_weights=true_weights,
        standardisation=standardisation,
        missing_data=missing,
    )


# -- validation helpers ---------------------------------------------------
def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _check_keys(document: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(repr(key) for key in unknown)}")


def _integer(document: Mapping[str, Any], key: str, low: int, high: int) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    if not low <= value <= high:
        raise ConfigError(f"{key} must be between {low} and {high}")
    return value


def _number(document: Mapping[str, Any], key: str, low: float, high: float) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    number = float(value)
    if not low <= number <= high:
        raise ConfigError(f"{key} must be between {low} and {high}")
    return number


def _positive_number(value: Any, where: str) -> float:
    number = _as_number(value, where)
    if number <= 0.0:
        raise ConfigError(f"{where} must be greater than 0")
    return number


def _non_negative_number(value: Any, where: str) -> float:
    number = _as_number(value, where)
    if number < 0.0:
        raise ConfigError(f"{where} must be zero or greater")
    return number


def _as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    return float(value)


def _choice(document: Mapping[str, Any], key: str, allowed: tuple[str, ...]) -> str:
    value = document.get(key)
    if value not in allowed:
        raise ConfigError(f"{key} must be one of {', '.join(allowed)}")
    return str(value)


def _integer_list(
    document: Mapping[str, Any], key: str, low: int, high: int
) -> tuple[int, ...]:
    value = document.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{key} must be a non-empty list of integers")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or not low <= item <= high:
            raise ConfigError(f"{key} entries must be integers between {low} and {high}")
        result.append(item)
    if len(set(result)) != len(result):
        raise ConfigError(f"{key} contains duplicates")
    return tuple(result)
