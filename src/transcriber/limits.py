"""Request-body size limit enforced while the body streams in.

FastAPI/Starlette parse multipart uploads *before* the endpoint runs, spooling
the whole file to a temp file with no size cap; a check inside the endpoint
only runs once a 20 GB upload has already landed on disk. This ASGI
middleware sits in front of that parser:

  * a declared Content-Length over the limit is rejected before a byte is read;
  * chunked/undeclared bodies are counted as they arrive and cut off at the limit.

In production the same limit also belongs at the edge (ingress/proxy), and big
files should skip the API entirely via pre-signed object-storage uploads.
"""

from __future__ import annotations

import json


class _TooLarge(Exception):
    pass


class BodySizeLimit:
    def __init__(self, app, max_bytes: int, detail: str | None = None):
        self.app = app
        self.max_bytes = max_bytes
        self.detail = detail or f"request body exceeds {max_bytes // (1 << 20)} MB"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(send)

        seen = 0
        exceeded = False
        responded = False

        async def counting_receive():
            nonlocal seen, exceeded
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    exceeded = True
                    raise _TooLarge
            return message

        async def guarded_send(message):
            nonlocal responded
            if exceeded:
                # The framework turned our abort into its own error (FastAPI reports body
                # parse failures as 400); answer with the real reason instead, once.
                if message["type"] == "http.response.start" and not responded:
                    responded = True
                    await self._reject(send)
                return
            if message["type"] == "http.response.start":
                responded = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _TooLarge:
            if not responded:
                await self._reject(send)

    async def _reject(self, send) -> None:
        body = json.dumps({"detail": self.detail}).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"connection", b"close")],
        })
        await send({"type": "http.response.body", "body": body})
