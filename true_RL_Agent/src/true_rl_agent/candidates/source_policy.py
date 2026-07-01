"""Closed source-pack access policy."""

from __future__ import annotations


class SourceAccessPolicy:
    def __init__(self, allowed_sources: dict[str, str]) -> None:
        self._allowed_sources = dict(allowed_sources)

    def resolve(self, logical_id: str) -> str:
        if logical_id not in self._allowed_sources:
            raise KeyError(f"source id is not allowlisted: {logical_id}")
        return self._allowed_sources[logical_id]
