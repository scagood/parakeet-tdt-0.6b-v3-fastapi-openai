"""GET /compare: the comparison page, served only with PARAKEET_COMPARE_UI on."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import HTTPException

from parakeet_service import routes


def test_compare_page_is_a_404_when_off(monkeypatch):
    monkeypatch.setattr(routes, "COMPARE_UI", False)
    with pytest.raises(HTTPException) as caught:
        routes.compare_page()
    assert caught.value.status_code == 404


def test_compare_page_calls_only_routes_that_exist(monkeypatch):
    monkeypatch.setattr(routes, "COMPARE_UI", True)
    page = routes.compare_page()
    assert page.media_type == "text/html"
    urls = set(re.findall(r'"(v1/[\w/]+)"', Path(page.path).read_text(encoding="utf-8")))
    assert urls == {"v1/models", "v1/aligners", "v1/audio/transcriptions"}
    assert {"/" + url for url in urls} <= {route.path for route in routes.router.routes}
