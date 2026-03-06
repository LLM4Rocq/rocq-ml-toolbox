from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

class DiagnosticParseError(ValueError):
    """Base class for all diagnostic parsing errors."""


class DiagnosticJSONError(DiagnosticParseError):
    """Raised when the input is not valid JSON."""


class DiagnosticValidationError(DiagnosticParseError):
    """Raised when JSON is valid but does not match the expected schema."""

class DiagnosticFormatError(ValueError):
    """Raised when a JSON object does not match the expected diagnostic schema."""


def _type_name(value: Any) -> str:
    return type(value).__name__

def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiagnosticFormatError(
            f"Field '{path}' must be an object, got {_type_name(value)}."
        )
    return value


def _require_int(value: Any, path: str) -> int:
    if type(value) is not int:
        raise DiagnosticFormatError(
            f"Field '{path}' must be an integer, got {_type_name(value)}."
        )
    return value


def _require_non_negative_int(value: Any, path: str) -> int:
    value = _require_int(value, path)
    if value < 0:
        raise DiagnosticFormatError(
            f"Field '{path}' must be >= 0, got {value}."
        )
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise DiagnosticFormatError(
            f"Field '{path}' must be a string, got {_type_name(value)}."
        )
    return value

def _require_field(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise DiagnosticFormatError(f"Missing required field '{path}.{key}'.")
    return data[key]


@dataclass(frozen=True, slots=True)
class Position:
    line: int
    character: int

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Position:
        data = _require_mapping(data, "position")

        return cls(
            line=_require_non_negative_int(
                _require_field(data, "line", "position"),
                "position.line",
            ),
            character=_require_non_negative_int(
                _require_field(data, "character", "position"),
                "position.character",
            ),
        )

    def to_json(self) -> dict[str, int]:
        return {
            "line": self.line,
            "character": self.character,
        }


@dataclass(frozen=True, slots=True)
class Range:
    start: Position
    end: Position

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Range:
        data = _require_mapping(data, "range")

        start = Position.from_json(
            _require_mapping(_require_field(data, "start", "range"), "range.start")
        )
        end = Position.from_json(
            _require_mapping(_require_field(data, "end", "range"), "range.end")
        )

        if (end.line, end.character) < (start.line, start.character):
            raise DiagnosticFormatError(
                "Field 'range.end' must not be before 'range.start'."
            )

        return cls(start=start, end=end)

    def to_json(self) -> dict[str, dict[str, int]]:
        return {
            "start": self.start.to_json(),
            "end": self.end.to_json(),
        }


@dataclass(frozen=True, slots=True)
class Diagnostic:
    range: Range
    severity: int
    message: str

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Diagnostic:
        data = _require_mapping(data, "diagnostic")

        return cls(
            range=Range.from_json(
                _require_mapping(_require_field(data, "range", "diagnostic"), "diagnostic.range")
            ),
            severity=_require_int(
                _require_field(data, "severity", "diagnostic"),
                "diagnostic.severity",
            ),
            message=_require_str(
                _require_field(data, "message", "diagnostic"),
                "diagnostic.message",
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "range": self.range.to_json(),
            "severity": self.severity,
            "message": self.message,
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
    try:
        return Diagnostic.from_json(obj)
    except DiagnosticFormatError as exc:
        raise DiagnosticValidationError(
            f"Invalid diagnostic #{diagnostic_number}: {exc}"
        ) from exc


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


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