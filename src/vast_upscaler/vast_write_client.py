"""Writable Vast.ai API client for phase 4 instance lifecycle operations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

LOGGER = logging.getLogger(__name__)


class VastWriteApiError(RuntimeError):
    """Base exception for writable Vast API client errors."""


class VastWriteAuthError(VastWriteApiError):
    """Raised when API key is missing or rejected."""


class VastWriteRateLimitError(VastWriteApiError):
    """Raised on HTTP 429 after retry budget."""


class VastWriteServerError(VastWriteApiError):
    """Raised on HTTP 5xx after retry budget."""


class VastWriteTimeoutError(VastWriteApiError):
    """Raised when HTTP timeout occurs."""


class VastWriteResponseError(VastWriteApiError):
    """Raised when payload shape is malformed."""


class VastCreateRejectedError(VastWriteApiError):
    """Raised when create request is rejected by API (4xx)."""


class VastCreateAmbiguousError(VastWriteApiError):
    """Raised when create result is unknown and requires reconciliation."""


@dataclass(frozen=True)
class VastWriteClientConfig:
    base_url: str = "https://console.vast.ai"
    timeout_seconds: float = 20.0
    max_retries: int = 2
    backoff_base_seconds: float = 0.5


@dataclass(frozen=True)
class CreateInstanceRequest:
    ask_id: int
    image: str | None = None
    template_hash_id: str | None = None
    disk_gb: float | None = None
    label: str | None = None
    runtype: str = "ssh_direct"
    target_state: str = "running"
    cancel_unavail: bool = True
    env: str | None = None


@dataclass(frozen=True)
class InstanceSummary:
    instance_id: int
    actual_status: str | None
    intended_status: str | None
    cur_state: str | None
    label: str | None
    image_runtype: str | None
    ssh_host: str | None
    ssh_port: int | None


class VastWriteClient:
    def __init__(
        self,
        api_key: str,
        config: VastWriteClientConfig | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise VastWriteAuthError("Missing VAST_API_KEY. Set it before paid Vast operations.")

        self._config = config or VastWriteClientConfig()
        timeout = httpx.Timeout(self._config.timeout_seconds)
        self._client = http_client or httpx.Client(
            base_url=self._config.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()

    def create_instance(self, req: CreateInstanceRequest) -> int:
        body: dict[str, Any] = {
            "runtype": req.runtype,
            "target_state": req.target_state,
            "cancel_unavail": req.cancel_unavail,
        }
        if req.template_hash_id:
            body["template_hash_id"] = req.template_hash_id
        if req.image:
            body["image"] = req.image
        if req.disk_gb is not None:
            body["disk"] = req.disk_gb
        if req.label:
            body["label"] = req.label
        if req.env:
            body["env"] = req.env

        payload = self._request_json(
            "PUT",
            f"/api/v0/asks/{req.ask_id}/",
            json_body=body,
            retryable=True,
            ambiguous_on_timeout=True,
            create_call=True,
        )

        new_contract = payload.get("new_contract")
        if isinstance(new_contract, int):
            return new_contract
        raise VastWriteResponseError("Create instance response missing integer new_contract.")

    def get_instance(self, instance_id: int) -> InstanceSummary:
        payload = self._request_json("GET", f"/api/v0/instances/{instance_id}/", retryable=True)
        raw = payload.get("instances")
        if not isinstance(raw, dict):
            raise VastWriteResponseError("Malformed show-instance response: expected object field 'instances'.")

        rid = raw.get("id")
        if not isinstance(rid, int):
            raise VastWriteResponseError("Malformed show-instance response: missing integer id.")

        return InstanceSummary(
            instance_id=rid,
            actual_status=_as_optional_str(raw.get("actual_status")),
            intended_status=_as_optional_str(raw.get("intended_status")),
            cur_state=_as_optional_str(raw.get("cur_state")),
            label=_as_optional_str(raw.get("label")),
            image_runtype=_as_optional_str(raw.get("image_runtype")),
            ssh_host=_as_optional_str(raw.get("ssh_host")),
            ssh_port=_as_optional_int(raw.get("ssh_port")),
        )

    def list_instances(self, *, limit: int = 25, select_filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params: dict[str, str] = {
            "limit": str(max(1, min(limit, 25))),
            "select_cols": '["id","label","actual_status","cur_state","intended_status"]',
        }
        if select_filters:
            params["select_filters"] = __import__("json").dumps(select_filters)

        payload = self._request_json("GET", "/api/v1/instances/", params=params, retryable=True)
        rows = payload.get("instances")
        if not isinstance(rows, list):
            raise VastWriteResponseError("Malformed show-instances response: expected list field 'instances'.")
        return rows

    def reconcile_ambiguous_create(self, *, label: str) -> int | None:
        if not label:
            return None
        rows = self.list_instances(limit=25, select_filters={"label": {"eq": label}})
        if len(rows) != 1:
            return None
        rid = rows[0].get("id")
        return rid if isinstance(rid, int) else None

    def destroy_instance(self, instance_id: int) -> bool:
        payload = self._request_json("DELETE", f"/api/v0/instances/{instance_id}/", retryable=True)
        success = payload.get("success")
        return bool(success is True)

    def _request_json(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        retryable: bool = False,
        ambiguous_on_timeout: bool = False,
        create_call: bool = False,
    ) -> dict[str, Any]:
        attempts = self._config.max_retries + 1
        method_upper = method.upper()

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.request(method_upper, path, json=json_body, params=params)
            except httpx.TimeoutException as exc:
                if ambiguous_on_timeout and create_call:
                    raise VastCreateAmbiguousError(
                        "Create instance timed out. Result is ambiguous; reconcile before retrying create."
                    ) from exc
                if retryable and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise VastWriteTimeoutError("Vast API request timed out.") from exc
            except httpx.HTTPError as exc:
                if retryable and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise VastWriteApiError(f"HTTP transport error: {exc}") from exc

            if response.status_code in (401, 403):
                raise VastWriteAuthError("Vast API authentication failed (401/403).")
            if response.status_code == 429:
                if retryable and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise VastWriteRateLimitError("Vast API rate limit reached (429).")
            if response.status_code >= 500:
                if retryable and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise VastWriteServerError(f"Vast API server error ({response.status_code}).")
            if response.status_code >= 400:
                if create_call:
                    raise VastCreateRejectedError(self._extract_error_message(response))
                raise VastWriteApiError(self._extract_error_message(response))

            try:
                payload = response.json()
            except ValueError as exc:
                raise VastWriteResponseError("Vast API response is not valid JSON.") from exc

            if not isinstance(payload, dict):
                raise VastWriteResponseError("Vast API JSON root must be an object.")

            LOGGER.debug("Vast write call succeeded: %s %s status=%s", method_upper, path, response.status_code)
            return payload

        raise VastWriteApiError("Vast API request failed after retries.")

    def _sleep_backoff(self, attempt: int) -> None:
        seconds = self._config.backoff_base_seconds * (2 ** (attempt - 1))
        time.sleep(seconds)

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = payload.get("msg") or payload.get("message") or payload.get("error")
                if message:
                    return f"Vast API error {response.status_code}: {message}"
        except ValueError:
            pass
        return f"Vast API error {response.status_code}: {response.text.strip() or 'unknown error'}"


def _as_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _as_optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
