# ADR 0001 - Separate read-only Vast API from paid operations

## Status

Accepted

## Context

Phase 2 requires real offer discovery and estimation, but must not perform any billing action.
Vast API includes both read and write endpoints under the same base URL and auth model.
Without strict boundaries, integration mistakes could create paid instances.

## Decision

Implement a dedicated read-only client with:

- Endpoint allowlist limited to:
  - `POST /api/v0/bundles/`
  - `GET /api/v0/users/current/`
- Explicit runtime rejection for any non-whitelisted route.
- No generic public request API exposed by this module.
- Offer filtering/ranking logic isolated from HTTP layer.

## Consequences

Positive:

- Strong protection against accidental paid operations.
- Testable domain logic (offers and cost) independent of network.
- Easier migration to a future paid-operations client with separate review.

Trade-offs:

- Additional code for explicit route guarding.
- Future write operations require a separate implementation path in phase 4.

## Alternatives considered

- Single generic API client with read/write methods and call-site discipline.
  - Rejected: too easy to misuse and violate financial safety constraints.
- Rely only on CLI dry-run flag.
  - Rejected: insufficient guarantee against accidental write calls.
