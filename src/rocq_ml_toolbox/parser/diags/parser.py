from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DiagnosticParseError(ValueError):
    """Base class for all diagnostic parsing errors."""


class DiagnosticJSONError(DiagnosticParseError):
    """Raised when the input is not valid JSON."""


class DiagnosticValidationError(DiagnosticParseError):
    """Raised when JSON is valid but does not match the expected schema."""


@dataclass(frozen=True, slots=True)
class Position:
    line: int
    character: int

    def to_json(self):
        return {
            "line": self.line,
            "character": self.character
        }

@dataclass(frozen=True, slots=True)
class Range:
    start: Position
    end: Position

    def to_json(self):
        return {
            "start": self.start.to_json(),
            "end": self.end.to_json()
        }

@dataclass(frozen=True, slots=True)
class Diagnostic:
    range: Range
    severity: int
    message: str

    def to_json(self):
        return {
            "range": self.range.to_json(),
            "severity": self.severity,
            "message": self.message
        }


def parse_diagnostics(text: str) -> list[Diagnostic]:
    """
    Parse a string containing multiple JSON objects written one after another.

    Example accepted format:
        { ... }
        { ... }
        { ... }

    Returns:
        A list of Diagnostic objects.

    Raises:
        DiagnosticJSONError:
            If one of the JSON objects is malformed.
        DiagnosticValidationError:
            If an object is valid JSON but does not follow the expected schema.
    """
    decoder = json.JSONDecoder()
    diagnostics: list[Diagnostic] = []
    index = 0
    length = len(text)

    while True:
        index = _skip_whitespace(text, index)
        if index >= length:
            break

        diagnostic_number = len(diagnostics) + 1

        try:
            obj, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise DiagnosticJSONError(
                _format_json_error(text, exc, diagnostic_number)
            ) from exc

        diagnostics.append(_build_diagnostic(obj, diagnostic_number))
        index = next_index

    return diagnostics


def parse_diagnostics_file(path: str | Path, encoding: str = "utf-8") -> list[Diagnostic]:
    """
    Read and parse diagnostics from a file.

    Raises:
        DiagnosticParseError:
            If the file cannot be read.
        DiagnosticJSONError / DiagnosticValidationError:
            If parsing fails.
    """
    file_path = Path(path)

    try:
        text = file_path.read_text(encoding=encoding)
    except OSError as exc:
        raise DiagnosticParseError(f"Unable to read '{file_path}': {exc}") from exc

    return parse_diagnostics(text)


def _build_diagnostic(obj: Any, diagnostic_number: int) -> Diagnostic:
    if not isinstance(obj, dict):
        raise DiagnosticValidationError(
            f"Invalid diagnostic #{diagnostic_number}: expected a JSON object at the top level, "
            f"got {_type_name(obj)}."
        )

    range_obj = _require_dict(obj, "range", diagnostic_number, path="diagnostic")
    start_obj = _require_dict(range_obj, "start", diagnostic_number, path="diagnostic.range")
    end_obj = _require_dict(range_obj, "end", diagnostic_number, path="diagnostic.range")

    start = Position(
        line=_require_non_negative_int(start_obj, "line", diagnostic_number, path="diagnostic.range.start"),
        character=_require_non_negative_int(start_obj, "character", diagnostic_number, path="diagnostic.range.start"),
    )

    end = Position(
        line=_require_non_negative_int(end_obj, "line", diagnostic_number, path="diagnostic.range.end"),
        character=_require_non_negative_int(end_obj, "character", diagnostic_number, path="diagnostic.range.end"),
    )

    if (end.line, end.character) < (start.line, start.character):
        raise DiagnosticValidationError(
            f"Invalid diagnostic #{diagnostic_number}: 'range.end' must not be before 'range.start'."
        )

    severity = _require_int(obj, "severity", diagnostic_number, path="diagnostic")
    message = _require_str(obj, "message", diagnostic_number, path="diagnostic")

    return Diagnostic(
        range=Range(start=start, end=end),
        severity=severity,
        message=message,
    )


def _require_dict(mapping: dict[str, Any], key: str, diagnostic_number: int, *, path: str) -> dict[str, Any]:
    value = _require_key(mapping, key, diagnostic_number, path=path)
    if not isinstance(value, dict):
        raise DiagnosticValidationError(
            f"Invalid diagnostic #{diagnostic_number}: field '{path}.{key}' must be an object, "
            f"got {_type_name(value)}."
        )
    return value


def _require_int(mapping: dict[str, Any], key: str, diagnostic_number: int, *, path: str) -> int:
    value = _require_key(mapping, key, diagnostic_number, path=path)
    if type(value) is not int:
        raise DiagnosticValidationError(
            f"Invalid diagnostic #{diagnostic_number}: field '{path}.{key}' must be an integer, "
            f"got {_type_name(value)}."
        )
    return value


def _require_non_negative_int(
    mapping: dict[str, Any], key: str, diagnostic_number: int, *, path: str
) -> int:
    value = _require_int(mapping, key, diagnostic_number, path=path)
    if value < 0:
        raise DiagnosticValidationError(
            f"Invalid diagnostic #{diagnostic_number}: field '{path}.{key}' must be >= 0, got {value}."
        )
    return value


def _require_str(mapping: dict[str, Any], key: str, diagnostic_number: int, *, path: str) -> str:
    value = _require_key(mapping, key, diagnostic_number, path=path)
    if not isinstance(value, str):
        raise DiagnosticValidationError(
            f"Invalid diagnostic #{diagnostic_number}: field '{path}.{key}' must be a string, "
            f"got {_type_name(value)}."
        )
    return value


def _require_key(mapping: dict[str, Any], key: str, diagnostic_number: int, *, path: str) -> Any:
    if key not in mapping:
        raise DiagnosticValidationError(
            f"Invalid diagnostic #{diagnostic_number}: missing required field '{path}.{key}'."
        )
    return mapping[key]


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _type_name(value: Any) -> str:
    return type(value).__name__


def _format_json_error(text: str, exc: json.JSONDecodeError, diagnostic_number: int) -> str:
    lines = text.splitlines()
    line_text = lines[exc.lineno - 1] if 1 <= exc.lineno <= len(lines) else ""
    pointer = " " * max(exc.colno - 1, 0) + "^"

    return (
        f"Invalid JSON while parsing diagnostic #{diagnostic_number} at "
        f"line {exc.lineno}, column {exc.colno}: {exc.msg}\n"
        f"{line_text}\n"
        f"{pointer}"
    )