"""Error classes for the cchvae package.

These mirror the structure of CELIA's error classes but rely only on the
standard library, so the package has no extra dependency on CELIA.
"""

from __future__ import annotations

from typing import Any


class CCHVAEError(Exception):
    """Base error for the cchvae package.

    Parameters
    ----------
    message : str
        Human-readable description of what went wrong.
    config : dict[str, Any] | None
        Optional structured context (e.g. observed types, shapes).
    param : str | None
        Name of the offending parameter, when applicable.
    hint : str | None
        Suggested fix for the user.
    source : str | None
        Fully-qualified location where the error originated
        (e.g. ``"CCHVAE._validate_init"``).
    """

    def __init__(
        self,
        message: str,
        *,
        config: dict[str, Any] | None = None,
        param: str | None = None,
        hint: str | None = None,
        source: str | None = None,
    ) -> None:
        self.message = message
        self.config = config or {}
        self.param = param
        self.hint = hint
        self.source = source
        super().__init__(self._format())

    def _format(self) -> str:
        parts = [self.message]
        if self.param:
            parts.append(f"[param={self.param}]")
        if self.source:
            parts.append(f"[source={self.source}]")
        if self.hint:
            parts.append(f"\n  hint: {self.hint}")
        if self.config:
            parts.append(f"\n  context: {self.config}")
        return " ".join(parts)


class CCHVAEValueError(CCHVAEError, ValueError):
    """Raised when a user-supplied value is invalid."""
