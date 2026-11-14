"""Gzip for the dashboard's large JSON routes.

Claw reads ``/api/cron/jobs?profile=all`` across a long, lossy link. The list
carries every job's full prompt, so it runs to hundreds of KB, and on that link
it did not finish inside Claw's 15 s fetch budget. JSON compresses roughly ten
to one, which brings it well under. Only the listed prefixes are compressed, so
file downloads, the PTY and websocket routes are left as they were.
"""

from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

GZIP_PATH_PREFIXES = ("/api/cron/",)
GZIP_MINIMUM_SIZE = 1024


class ScopedGZipMiddleware:
    def __init__(self, app: ASGIApp, prefixes: tuple[str, ...] = GZIP_PATH_PREFIXES) -> None:
        self.app = app
        self.prefixes = prefixes
        self.gzip = GZipMiddleware(app, minimum_size=GZIP_MINIMUM_SIZE)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if path.startswith(self.prefixes):
            await self.gzip(scope, receive, send)
            return
        await self.app(scope, receive, send)
