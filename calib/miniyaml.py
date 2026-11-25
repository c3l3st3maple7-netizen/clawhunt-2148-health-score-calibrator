"""A small, dependency-free YAML subset loader.

The task requires a YAML configuration file and a standard-library-only
generator, so this module implements exactly the YAML subset that the
configuration files use.  It is deliberately strict: anything outside the
subset raises :class:`YAMLError` with a line number instead of being silently
misread.

Supported
---------
* block mappings and block sequences, nested by indentation (spaces only)
* a sequence entry that starts a mapping (``- key: value``) or a nested
  sequence (``- - item``)
* flow mappings ``{a: 1, b: 2}`` and flow sequences ``[1, 2, 3]`` on one line
* plain, single-quoted and double-quoted scalars (double quotes use JSON
  escape rules)
* comments introduced by ``#`` at the start of a line or after whitespace
* integers, floats (``1.5``, ``.5``, ``1e-3``), ``true/false/yes/no/on/off``,
  ``null``/``~``
* a whole-document JSON object or array (YAML is a superset of JSON)

Not supported
-------------
anchors/aliases, tags, multi-line block scalars (``|``, ``>``), multi-document
streams, and tabs used for indentation.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["YAMLError", "loads", "load"]


class YAMLError(ValueError):
    """Raised when the document is outside the supported YAML subset."""

    def __init__(self, message: str, lineno: int | None = None) -> None:
        self.lineno = lineno
        super().__init__(f"line {lineno}: {message}" if lineno else message)


_INT = re.compile(r"^[+-]?(0|[1-9][0-9]*)$")
_FLOAT = re.compile(
    r"^[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+(?:\.[0-9]*)?[eE][+-]?[0-9]+)$"
)


def load(path: str) -> Any:
    """Read ``path`` as UTF-8 and parse it with :func:`loads`."""
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())


def loads(text: str) -> Any:
    """Parse a YAML-subset document held in ``text``."""
    text = text.lstrip("\ufeff")
    if not text.strip():
        return None

    stripped = text.lstrip()
    if stripped[0] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass  # a YAML flow collection; fall through to the YAML reader

    lines: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        leading = raw[: len(raw) - len(raw.lstrip(" \t"))]
        if "\t" in leading:
            raise YAMLError("tabs are not allowed for indentation", lineno)
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip(), lineno))

    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise YAMLError("unexpected content", lines[index][2])
    return value


def _strip_comment(line: str) -> str:
    out: list[str] = []
    quote: str | None = None
    for char in line:
        if quote is not None:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char == "#" and (not out or out[-1] in " \t"):
            break
        if char in "'\"":
            quote = char
        out.append(char)
    return "".join(out).rstrip()


def _parse_block(
    lines: list[tuple[int, str, int]], index: int, indent: int
) -> tuple[Any, int]:
    if index >= len(lines) or lines[index][0] != indent:
        lineno = lines[index][2] if index < len(lines) else None
        raise YAMLError("unexpected indentation", lineno)
    if _is_sequence_entry(lines[index][1]):
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _is_sequence_entry(content: str) -> bool:
    return content == "-" or content.startswith("- ")


def _parse_mapping(
    lines: list[tuple[int, str, int]], index: int, indent: int
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content, lineno = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YAMLError("unexpected indentation", lineno)
        if _is_sequence_entry(content):
            raise YAMLError("sequence entry where a mapping key was expected", lineno)

        pair = _split_key(content, lineno)
        if pair is None:
            raise YAMLError("expected 'key: value'", lineno)
        raw_key, rest = pair
        key = _scalar(raw_key, lineno)
        if not isinstance(key, str):
            key = str(key)
        if key in result:
            raise YAMLError(f"duplicate key {key!r}", lineno)

        if rest == "":
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                result[key], index = _parse_block(lines, index, lines[index][0])
            elif index < len(lines) and lines[index][0] == indent and _is_sequence_entry(
                lines[index][1]
            ):
                # YAML allows a sequence to sit at the same indent as its key.
                result[key], index = _parse_sequence(lines, index, indent)
            else:
                result[key] = None
        else:
            result[key] = _parse_flow_or_scalar(rest, lineno)
            index += 1
    return result, index


def _parse_sequence(
    lines: list[tuple[int, str, int]], index: int, indent: int
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        line_indent, content, lineno = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YAMLError("unexpected indentation", lineno)
        if not _is_sequence_entry(content):
            break

        rest = content[1:].strip() if content != "-" else ""
        if rest == "":
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_block(lines, index, lines[index][0])
            else:
                value = None
            items.append(value)
            continue

        if _is_sequence_entry(rest):
            items.append(_parse_inline_block(lines, index, indent, rest, "sequence"))
            index = _next_index(lines, index, indent)
            continue

        pair = None if rest[0] in "[{" else _split_key(rest, lineno)
        if pair is None:
            items.append(_parse_flow_or_scalar(rest, lineno))
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                raise YAMLError("unexpected indentation", lines[index][2])
        else:
            items.append(_parse_inline_block(lines, index, indent, rest, "mapping"))
            index = _next_index(lines, index, indent)
    return items, index


def _next_index(
    lines: list[tuple[int, str, int]], index: int, indent: int
) -> int:
    cursor = index + 1
    while cursor < len(lines) and lines[cursor][0] > indent:
        cursor += 1
    return cursor


def _parse_inline_block(
    lines: list[tuple[int, str, int]],
    index: int,
    indent: int,
    first_content: str,
    kind: str,
) -> Any:
    """Parse a block whose first line follows ``- `` on the same line."""
    child_indent = indent + 2
    end = _next_index(lines, index, indent)
    sub = [(child_indent, first_content, lines[index][2])] + list(
        lines[index + 1 : end]
    )
    parser = _parse_sequence if kind == "sequence" else _parse_mapping
    value, consumed = parser(sub, 0, child_indent)
    if consumed != len(sub):
        raise YAMLError("unexpected indentation", sub[consumed][2])
    return value


def _split_key(content: str, lineno: int) -> tuple[str, str] | None:
    quote: str | None = None
    depth = 0
    for position, char in enumerate(content):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
            if depth < 0:
                raise YAMLError("unbalanced brackets", lineno)
        elif char == ":" and depth == 0:
            if position + 1 == len(content) or content[position + 1] in " \t":
                return content[:position].strip(), content[position + 1 :].strip()
    return None


def _parse_flow_or_scalar(text: str, lineno: int) -> Any:
    stripped = text.strip()
    if stripped.startswith("|") or stripped.startswith(">"):
        raise YAMLError("block scalars (| and >) are not supported", lineno)
    if stripped.startswith("&") or stripped.startswith("*"):
        raise YAMLError("anchors and aliases are not supported", lineno)
    if stripped.startswith("[") or stripped.startswith("{"):
        return _parse_flow(stripped, lineno)
    return _scalar(stripped, lineno)


def _parse_flow(text: str, lineno: int) -> Any:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        return [_parse_flow_or_scalar(part, lineno) for part in _split_flow(body, lineno)]
    if text.startswith("{") and text.endswith("}"):
        body = text[1:-1].strip()
        if not body:
            return {}
        result: dict[str, Any] = {}
        for part in _split_flow(body, lineno):
            pair = _split_key(part, lineno)
            if pair is None:
                raise YAMLError("expected 'key: value' in flow mapping", lineno)
            key, rest = pair
            parsed_key = _scalar(key, lineno)
            key_text = parsed_key if isinstance(parsed_key, str) else str(parsed_key)
            if key_text in result:
                raise YAMLError(f"duplicate key {key_text!r}", lineno)
            result[key_text] = _parse_flow_or_scalar(rest, lineno) if rest else None
        return result
    raise YAMLError("unterminated flow collection", lineno)


def _split_flow(body: str, lineno: int) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for char in body:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            current.append(char)
        elif char in "[{":
            depth += 1
            current.append(char)
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise YAMLError("unbalanced brackets", lineno)
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    if any(not part for part in parts):
        raise YAMLError("empty element in flow collection", lineno)
    return parts


def _scalar(text: str, lineno: int) -> Any:
    value = text.strip()
    if value == "":
        return None
    if value.startswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise YAMLError(f"invalid double-quoted string: {exc.msg}", lineno) from exc
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise YAMLError("unterminated single-quoted string", lineno)
        return value[1:-1].replace("''", "'")
    lowered = value.lower()
    if lowered in ("null", "~"):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if _INT.match(value):
        return int(value)
    if _FLOAT.match(value):
        return float(value)
    return value
