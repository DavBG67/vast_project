"""Read-only Vast.ai API client for phase 2 offer discovery."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, ClassVar, Self

import httpx

LOGGER = logging.getLogger(__name__)


class VastApiError(RuntimeError):
    """Base exception for Vast API client errors."""


class VastAuthError(VastApiError):
    """Raised when API key is missing or rejected."""


class VastRateLimitError(VastApiError):
    """Raised when the API returns HTTP 429."""


class VastServerError(VastApiError):
    """Raised for 5xx API errors after retry budget exhaustion."""


class VastTimeoutError(VastApiError):
    """Raised when a request times out."""


class VastResponseError(VastApiError):
    """Raised when API payloads are malformed."""


class VastReadOnlyViolation(VastApiError):
    """Raised when code tries to call a non-read-only endpoint."""


@dataclass(frozen=True)
class VastClientConfig:
    base_url: str = "https://console.vast.ai"
    timeout_seconds: float = 15.0
    max_retries: int = 2
    backoff_base_seconds: float = 0.5


class VastClient:
    """Strictly read-only Vast.ai client.

    In phase 2 this client only supports offer search and account info checks.
    It intentionally rejects arbitrary endpoints to eliminate accidental paid operations.
    """

    _ALLOWED_ROUTES: ClassVar[set[tuple[str, str]]] = {
        ("POST", "/api/v0/bundles/"),
        ("GET", "/api/v0/users/current/"),
    }

    def __init__(
        self,
        api_key: str,
        config: VastClientConfig | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise VastAuthError("Missing VAST_API_KEY. Set it in environment before using live Vast API calls.")

        self._config = config or VastClientConfig()
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

    def get_current_user(self) -> dict[str, Any]:
        """Read-only auth check endpoint."""
        return self._request_json("GET", "/api/v0/users/current/", retryable=True)

    def search_offers(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """Query Vast.ai marketplace offers in read-only mode."""
        payload = self._request_json("POST", "/api/v0/bundles/", json_body=query, retryable=True)
        offers = payload.get("offers")
        if not isinstance(offers, list):
            raise VastResponseError("Malformed API response: expected 'offers' to be a list.")
        return offers

    def _request_json(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> dict[str, Any]:
        method_upper = method.upper()
        if (method_upper, path) not in self._ALLOWED_ROUTES:
            raise VastReadOnlyViolation(f"Blocked non-read-only Vast API call: {method_upper} {path}")

        attempts = self._config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.request(method_upper, path, json=json_body)
            except httpx.TimeoutException as exc:
                if retryable and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise VastTimeoutError("Vast API request timed out.") from exc
            except httpx.HTTPError as exc:
                if retryable and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise VastApiError(f"HTTP transport error: {exc}") from exc

            if response.status_code in (401, 403):
                raise VastAuthError("Vast API authentication failed (401/403).")
            if response.status_code == 429:
                if retryable and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise VastRateLimitError("Vast API rate limit reached (429).")
            if response.status_code >= 500:
                if retryable and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise VastServerError(f"Vast API server error ({response.status_code}).")
            if response.status_code >= 400:
                raise VastApiError(self._extract_error_message(response))

            try:
                payload = response.json()
            except ValueError as exc:
                raise VastResponseError("Vast API response is not valid JSON.") from exc

            if not isinstance(payload, dict):
                raise VastResponseError("Vast API JSON root must be an object.")

            LOGGER.debug("Vast read-only call succeeded: %s %s status=%s", method_upper, path, response.status_code)
            return payload

        raise VastApiError("Vast API request failed after retries.")

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
