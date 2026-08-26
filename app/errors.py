"""
Error shape. Everything the API returns on failure is {"detail": "..."}.

That is FastAPI's own convention and, more importantly, what the Next.js client
already expects -- lib/api-client.ts reads `detail` and nothing else. A handler that
leaks a traceback into the response body would render as a wall of Python in the UI.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logging_utils import _log


class ApiError(HTTPException):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)


def not_found(what: str = "Resource") -> ApiError:
    return ApiError(404, f"{what} not found")


def bad_request(detail: str) -> ApiError:
    return ApiError(400, detail)


def unauthorized(detail: str = "Not authenticated") -> ApiError:
    return ApiError(401, detail)


def forbidden(detail: str = "Not permitted") -> ApiError:
    return ApiError(403, detail)


def install(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):  # noqa: ANN202
        # FastAPI's default is a list of {loc,msg,type} dicts. The client expects a
        # string, and "[object Object]" in a toast helps nobody -- so flatten it here
        # rather than making every consumer handle two shapes.
        parts = []
        for error in exc.errors():
            location = ".".join(str(p) for p in error.get("loc", []) if p != "body")
            message = error.get("msg", "invalid")
            parts.append(f"{location}: {message}" if location else message)
        return JSONResponse(status_code=422, content={"detail": "; ".join(parts) or "Invalid request"})

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException):  # noqa: ANN202
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN202
        # Log the real error server-side; return a generic message. An unhandled
        # exception's text can carry SQL fragments and file paths.
        _log(f"UNHANDLED {request.method} {request.url.path}: {exc!r}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
