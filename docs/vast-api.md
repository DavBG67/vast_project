# Vast API Notes (Phase 2)

## Sources checked

- https://docs.vast.ai/api-reference/search/search-offers.md
- https://docs.vast.ai/api-reference/openapi/yaml/search_offers.yaml
- https://docs.vast.ai/api-reference/authentication.md
- https://docs.vast.ai/api-reference/rate-limits-and-errors.md

## Read-only endpoints used

- `POST /api/v0/bundles/` for offer search.
- `GET /api/v0/users/current/` for optional auth verification.

Base URL:

- `https://console.vast.ai`

Auth:

- Header `Authorization: Bearer <VAST_API_KEY>`

## Search offers request shape

Phase 2 query uses only read filters:

- `type: "ondemand"`
- `limit`
- `verified.eq`
- `rentable.eq`
- `gpu_name.in`
- `num_gpus.eq`
- `reliability.gte`
- `dph_total.lte`
- `disk_space.gte`
- `direct_port_count.gte`
- `order`

## Fields consumed from offers

- Identity and location: `id`, `gpu_name`, `geolocation`
- Cost: `dph_total`, `search.totalHour`, `search.gpuCostPerHour`, `search.diskHour`, `storage_cost`
- Constraints: `reliability`, `verification`/`verified`, `num_gpus`, `disk_space`, `direct_port_count`
- Secondary ranking signals: `total_flops`, `dlperf`, `dlperf_per_dphtotal`, `inet_down`, `inet_up`, `cpu_cores_effective`

## Error and rate-limit behavior

- Auth failures: HTTP 401/403.
- Rate-limit: HTTP 429, often without `Retry-After` header.
- Server errors: HTTP 5xx.
- Error payloads are not perfectly uniform (`msg`, `message`, or `error`).

Client policy in phase 2:

- Bounded retries with exponential backoff for read-only idempotent operations.
- Strict endpoint allowlist to prevent accidental paid operations.
- Response validation before use; malformed payloads raise explicit errors.

## Explicitly excluded in phase 2

- Create/start/stop/destroy instance endpoints.
- Any endpoint that can trigger billing side effects.
