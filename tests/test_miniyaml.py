"""The YAML subset loader: what it accepts, and what it refuses loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from calib import miniyaml


def test_default_config_parses(project_root: Path) -> None:
    document = miniyaml.load(str(project_root / "configs" / "default.yaml"))
    assert document["dataset"]["companies"] == 300
    assert document["dataset"]["sites"] == 3
    assert document["dataset"]["months"] == 24
    assert document["dataset"]["annual_churn_target"] == 0.12
    assert document["latent_activity"]["seasonal_months"] == [1, 2]
    assert document["true_weights"]["event"] == 0.55
    assert isinstance(document["seed"], int)


def test_nested_sequence_of_mappings() -> None:
    document = miniyaml.loads(
        """
        outer:
          items:
            - name: first
              values: [1, 2]
            - name: second
              values:
                - 3
                - 4
        """
    )
    assert document == {
        "outer": {
            "items": [
                {"name": "first", "values": [1, 2]},
                {"name": "second", "values": [3, 4]},
            ]
        }
    }


def test_flow_mapping() -> None:
    assert miniyaml.loads('m: {a: 1, b: "two", c: [1, 2]}') == {
        "m": {"a": 1, "b": "two", "c": [1, 2]}
    }


def test_comments_and_quoted_hash() -> None:
    document = miniyaml.loads(
        """
        # leading comment
        plain: value   # trailing comment
        quoted: "a # b"
        single: 'c # d'
        """
    )
    assert document == {"plain": "value", "quoted": "a # b", "single": "c # d"}


def test_scalar_types() -> None:
    document = miniyaml.loads(
        """
        yes_value: yes
        off_value: off
        null_value: null
        tilde: ~
        integer: -3
        float_value: 1e-3
        plain_text: hello world
        """
    )
    assert document["yes_value"] is True
    assert document["off_value"] is False
    assert document["null_value"] is None
    assert document["tilde"] is None
    assert document["integer"] == -3
    assert document["float_value"] == 0.001
    assert document["plain_text"] == "hello world"


def test_json_document_is_accepted() -> None:
    assert miniyaml.loads('{"a": [1, 2], "b": null}') == {"a": [1, 2], "b": None}


def test_duplicate_key_is_rejected() -> None:
    with pytest.raises(miniyaml.YAMLError, match="duplicate key"):
        miniyaml.loads("a: 1\na: 2\n")


def test_tab_indentation_is_rejected() -> None:
    with pytest.raises(miniyaml.YAMLError, match="tabs"):
        miniyaml.loads("a:\n\tb: 1\n")


def test_block_scalar_is_rejected() -> None:
    with pytest.raises(miniyaml.YAMLError, match="block scalars"):
        miniyaml.loads("a: |\n  text\n")


def test_sequence_at_the_same_indent_as_its_key() -> None:
    document = miniyaml.loads("keys:\n- a\n- b\nnext: 1\n")
    assert document == {"keys": ["a", "b"], "next": 1}


def test_nested_sequence_on_one_line() -> None:
    assert miniyaml.loads("- - 1\n  - 2\n- 3\n") == [[1, 2], 3]


def test_unterminated_quote_is_rejected() -> None:
    with pytest.raises(miniyaml.YAMLError):
        miniyaml.loads('a: "unterminated\n')


def test_unknown_yaml_feature_is_rejected() -> None:
    with pytest.raises(miniyaml.YAMLError, match="anchors"):
        miniyaml.loads("a: &anchor 1\n")
