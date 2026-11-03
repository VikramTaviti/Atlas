"""Authentication integration seam.

The default implementation intentionally permits local unauthenticated access. It
makes no identity or authorization claim; deployments must inject a real verifier.
"""

from __future__ import annotations

from typing import Protocol

from fastapi import Request


class Authenticator(Protocol):
    async def authenticate(self, request: Request) -> str | None:
        """Return a trusted principal, or None when access is allowed anonymously."""


class UnauthenticatedLocalAuthenticator:
    """Explicit development default: accepts requests without authenticating them."""

    async def authenticate(self, request: Request) -> str | None:
        del request
        return None
