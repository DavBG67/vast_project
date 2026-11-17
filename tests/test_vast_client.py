import httpx
import pytest

from vast_upscaler.vast_client import (
    VastAuthError,
    VastClient,
    VastRateLimitError,
    VastReadOnlyViolation,
    VastResponseError,
    VastServerError,
    VastTimeoutError,
)


class FakeHttpClient:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0

    def request(self, method, path, json=None):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No fake response left")
        return self.responses.pop(0)

    def close(self):
        return None


def _response(status_code: int, payload):
    request = httpx.Request("POST", "https://console.vast.ai/api/v0/bundles/")
    return httpx.Response(status_code=status_code, json=payload, request=request)


def test_missing_auth_raises() -> None:
    with pytest.raises(VastAuthError):
        VastClient(api_key="")


def test_valid_search_response() -> None:
    fake = FakeHttpClient(responses=[_response(200, {"offers": [{"id": 1, "gpu_name": "RTX 4090"}]})])
    client = VastClient(api_key="key", http_client=fake)

    offers = client.search_offers({"limit": 1})

    assert len(offers) == 1
    assert offers[0]["id"] == 1


def test_malformed_response_raises() -> None:
    fake = FakeHttpClient(responses=[_response(200, {"offers": "bad"})])
    client = VastClient(api_key="key", http_client=fake)

    with pytest.raises(VastResponseError):
        client.search_offers({})


def test_timeout_raises() -> None:
    fake = FakeHttpClient(exc=httpx.TimeoutException("timeout"))
    client = VastClient(api_key="key", http_client=fake)

    with pytest.raises(VastTimeoutError):
        client.search_offers({})


def test_401_raises_auth_error() -> None:
    fake = FakeHttpClient(responses=[_response(401, {"msg": "Invalid user key"})])
    client = VastClient(api_key="key", http_client=fake)

    with pytest.raises(VastAuthError):
        client.search_offers({})


def test_429_retries_then_raises(monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.vast_client.time.sleep", lambda _seconds: None)
    fake = FakeHttpClient(responses=[_response(429, {"msg": "API requests too frequent"})] * 3)
    client = VastClient(api_key="key", http_client=fake)

    with pytest.raises(VastRateLimitError):
        client.search_offers({})
    assert fake.calls == 3


def test_500_retries_then_raises(monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.vast_client.time.sleep", lambda _seconds: None)
    fake = FakeHttpClient(responses=[_response(500, {"msg": "error"})] * 3)
    client = VastClient(api_key="key", http_client=fake)

    with pytest.raises(VastServerError):
        client.search_offers({})
    assert fake.calls == 3


def test_non_readonly_blocked() -> None:
    fake = FakeHttpClient(responses=[])
    client = VastClient(api_key="key", http_client=fake)

    with pytest.raises(VastReadOnlyViolation):
        client._request_json("PUT", "/api/v0/asks/123/")
