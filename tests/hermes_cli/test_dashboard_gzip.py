"""ScopedGZipMiddleware compresses only the listed dashboard routes."""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from hermes_cli.dashboard_gzip import ScopedGZipMiddleware

BIG = {"prompt": "Chấm điểm lead " * 500}


def _client() -> TestClient:
    async def big(request):
        return JSONResponse(BIG)

    app = Starlette(routes=[
        Route("/api/cron/jobs", big),
        Route("/api/files/download", big),
    ])
    app.add_middleware(ScopedGZipMiddleware)
    return TestClient(app)


def test_cron_list_is_gzipped_and_decodes_to_the_same_json():
    res = _client().get("/api/cron/jobs", headers={"Accept-Encoding": "gzip"})
    assert res.headers.get("content-encoding") == "gzip"
    assert res.json() == BIG


def test_routes_outside_the_prefixes_are_not_compressed():
    res = _client().get("/api/files/download", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in res.headers
    assert res.json() == BIG


def test_client_without_gzip_gets_plain_json():
    res = _client().get("/api/cron/jobs", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in res.headers
    assert res.json() == BIG
