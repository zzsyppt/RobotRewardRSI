"""Task-neutral contracts for structured subprocess argv."""

from __future__ import annotations

from collections.abc import Sequence


def require_string_argv(argv: Sequence[str], *, context: str = "argv") -> None:
    if isinstance(argv, str | bytes) or not isinstance(argv, Sequence):
        raise ValueError(f"{context} must be a sequence of strings")
    if not all(isinstance(part, str) for part in argv):
        raise ValueError(f"{context} entries must be strings")


def option_occurrences(argv: Sequence[str], option: str) -> list[int]:
    require_string_argv(argv)
    inline_prefix = f"{option}="
    return [
        index
        for index, part in enumerate(argv)
        if part == option or part.startswith(inline_prefix)
    ]


def require_option_absent(argv: Sequence[str], option: str) -> None:
    count = len(option_occurrences(argv, option))
    if count:
        raise ValueError(f"argv option {option} must be absent; found {count}")


def unique_option_value(argv: Sequence[str], option: str) -> str:
    occurrences = option_occurrences(argv, option)
    if len(occurrences) != 1:
        raise ValueError(
            f"argv option {option} must appear exactly once; "
            f"found {len(occurrences)}"
        )
    index = occurrences[0]
    if argv[index] != option:
        raise ValueError(f"argv option {option} must use a separate value token")
    if index + 1 >= len(argv):
        raise ValueError(f"argv option {option} is missing its value")
    value = argv[index + 1]
    if not value:
        raise ValueError(f"argv option {option} must have a non-empty value")
    return value


def require_exact_option_value(
    argv: Sequence[str],
    option: str,
    *,
    expected: str,
) -> str:
    actual = unique_option_value(argv, option)
    if actual != expected:
        raise ValueError(f"argv option {option} does not match its job contract")
    return actual


__all__ = [
    "option_occurrences",
    "require_exact_option_value",
    "require_option_absent",
    "require_string_argv",
    "unique_option_value",
]
