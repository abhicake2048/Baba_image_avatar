from __future__ import annotations

from typing import Any

import httpx


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def friendly_api_error(response: httpx.Response) -> OpenRouterError:
    status = response.status_code
    request_id = response.headers.get("x-request-id")
    suffix = f" (request {request_id})" if request_id else ""
    try:
        payload = response.json()
        detail = payload.get("error", payload) if isinstance(payload, dict) else payload
        if isinstance(detail, dict):
            provider_message = str(detail.get("message") or detail.get("code") or "").strip()
        else:
            provider_message = str(detail).strip()
    except Exception:
        provider_message = ""

    if status == 401:
        message = "OpenRouter rejected the API key. Check OPENROUTER_API_KEY."
    elif status == 402:
        message = "OpenRouter reports insufficient balance for this request."
    elif status == 403:
        message = "OpenRouter refused this request because of account, provider, or safety restrictions."
    elif status == 404:
        message = "The selected OpenRouter model or endpoint is unavailable."
    elif status == 429:
        message = "OpenRouter rate-limited the request. Wait briefly and try again."
    elif status >= 500:
        message = "The image provider is temporarily unavailable."
    else:
        message = f"OpenRouter returned HTTP {status}."
    if provider_message and status not in {401, 402}:
        message += f" {provider_message[:300]}"
    return OpenRouterError(message + suffix, status_code=status, retryable=status == 429 or status >= 500)


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Private Portrait Studio",
        }
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def post(
        self, path: str, payload: dict[str, Any], *, retry_on_timeout: bool = False
    ) -> dict[str, Any]:
        try:
            response = await self._client.post(
                f"{self.base_url}/{path.lstrip('/')}", headers=self._headers, json=payload
            )
        except httpx.TimeoutException as exc:
            message = "The OpenRouter request timed out."
            if not retry_on_timeout:
                message += " It was not retried to avoid duplicate image charges."
            raise OpenRouterError(message, retryable=retry_on_timeout) from exc
        except httpx.RequestError as exc:
            raise OpenRouterError(f"Could not reach OpenRouter: {exc}", retryable=True) from exc
        if not response.is_success:
            raise friendly_api_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise OpenRouterError("OpenRouter returned a non-JSON response.") from exc
        if not isinstance(data, dict):
            raise OpenRouterError("OpenRouter returned an unexpected response shape.")
        return data

    async def get(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(f"{self.base_url}/{path.lstrip('/')}", headers=self._headers)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise OpenRouterError(f"Could not validate OpenRouter models: {exc}", retryable=True) from exc
        if not response.is_success:
            raise friendly_api_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise OpenRouterError("OpenRouter model catalog returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise OpenRouterError("OpenRouter model catalog returned an unexpected response.")
        return data


def catalog_supports_references(catalog: dict[str, Any], model_id: str) -> bool:
    models = catalog.get("data")
    if not isinstance(models, list):
        return False
    for model in models:
        if not isinstance(model, dict) or model.get("id") != model_id:
            continue
        params = model.get("supported_parameters", {})
        if isinstance(params, dict):
            return "input_references" in params
        if isinstance(params, list):
            return "input_references" in params
    return False

