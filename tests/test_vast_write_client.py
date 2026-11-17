import httpx
import pytest

from vast_upscaler.vast_write_client import (
    CreateInstanceRequest,
    VastCreateAmbiguousError,
    VastCreateRejectedError,
    VastWriteAuthError,
    VastWriteClient,
    VastWriteResponseError,
    VastWriteTimeoutError,
)


class FakeHttpClient:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = []

    def request(self, method, path, json=None, params=None):
        self.calls.append((method, path, json, params))
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No fake response left")
        return self.responses.pop(0)

    def close(self):
        return None


def _response(status_code: int, payload):
    request = httpx.Request("PUT", "https://console.vast.ai/api/v0/asks/123/")
    return httpx.Response(status_code=status_code, json=payload, request=request)


def test_missing_auth_raises() -> None:
    with pytest.raises(VastWriteAuthError):
        VastWriteClient(api_key="")


def test_create_instance_success() -> None:
    fake = FakeHttpClient(responses=[_response(200, {"success": True, "new_contract": 789})])
    client = VastWriteClient(api_key="key", http_client=fake)

    req = CreateInstanceRequest(ask_id=123, image="ghcr.io/x/y:tag", label="sess-1")
    instance_id = client.create_instance(req)

    assert instance_id == 789
    method, path, body, _params = fake.calls[0]
    assert method == "PUT"
    assert path == "/api/v0/asks/123/"
    assert body["label"] == "sess-1"
    assert body["runtype"] == "ssh_direct"


def test_create_instance_rejected() -> None:
    fake = FakeHttpClient(responses=[_response(400, {"error": "invalid_args", "msg": "bad"})])
    client = VastWriteClient(api_key="key", http_client=fake)

    with pytest.raises(VastCreateRejectedError):
        client.create_instance(CreateInstanceRequest(ask_id=123, image="img:tag"))


def test_create_timeout_is_ambiguous() -> None:
    fake = FakeHttpClient(exc=httpx.TimeoutException("timeout"))
    client = VastWriteClient(api_key="key", http_client=fake)

    with pytest.raises(VastCreateAmbiguousError):
        client.create_instance(CreateInstanceRequest(ask_id=123, image="img:tag"))


def test_get_instance_success() -> None:
    payload = {
        "instances": {
            "id": 999,
            "actual_status": "running",
            "intended_status": "running",
            "cur_state": "running",
            "label": "sess-abc",
            "image_runtype": "ssh_direct",
            "ssh_host": "ssh1.vast.ai",
            "ssh_port": 10222,
        }
    }
    fake = FakeHttpClient(responses=[_response(200, payload)])
    client = VastWriteClient(api_key="key", http_client=fake)

    summary = client.get_instance(999)

    assert summary.instance_id == 999
    assert summary.actual_status == "running"
    assert summary.ssh_port == 10222


def test_get_instance_malformed_raises() -> None:
    fake = FakeHttpClient(responses=[_response(200, {"instances": []})])
    client = VastWriteClient(api_key="key", http_client=fake)

    with pytest.raises(VastWriteResponseError):
        client.get_instance(1)


def test_destroy_success() -> None:
    fake = FakeHttpClient(responses=[_response(200, {"success": True})])
    client = VastWriteClient(api_key="key", http_client=fake)

    ok = client.destroy_instance(4321)

    assert ok is True


def test_destroy_failure_response_false() -> None:
    fake = FakeHttpClient(responses=[_response(200, {"success": False, "msg": "nope"})])
    client = VastWriteClient(api_key="key", http_client=fake)

    ok = client.destroy_instance(4321)

    assert ok is False


def test_non_create_timeout_raises_timeout_error() -> None:
    fake = FakeHttpClient(exc=httpx.TimeoutException("timeout"))
    client = VastWriteClient(api_key="key", http_client=fake)

    with pytest.raises(VastWriteTimeoutError):
        client.get_instance(321)
