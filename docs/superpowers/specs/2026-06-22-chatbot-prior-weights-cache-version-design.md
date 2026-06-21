# Chatbot Prior-Weight Injection and Cache Version Design

## Scope

Improve only the production chatbot governance path. Automatically pass trusted session portfolio weights into governance optimization and prevent pre-audit cached governance payloads from being reused.

## Prior-Weight Resolution

Add one API helper that reads `resolved_memory["session_state"]["active_weights"]` and returns normalized ticker weights only when:

- `weights` is a non-empty dictionary;
- every retained value is finite and non-negative;
- the positive total is greater than zero; and
- an `equal_weight_proxy` has explicit user approval.

The helper returns an empty dictionary when these conditions are not met. It does not use `latest_governance_run` as current holdings because a previous recommendation is not necessarily the user's actual portfolio.

REST chat, streaming chat, and background chat jobs add the resolved dictionary to LangGraph's configurable context as `previous_weights`. Existing session and model configuration remain unchanged.

## Tool Resolution

`governance_pipeline_with_cache` keeps its explicit optional `previous_weights` argument. When that argument is absent, it reads `config["configurable"]["previous_weights"]`. An explicit tool argument takes precedence over configuration.

When prior weights are available, the wrapper bypasses plan-cache reads and writes because turnover is session-specific. The resolved weights are forwarded to `run_full_governance_pipeline` and then to the optimizer.

## Cache Versioning

Define a constant cache contract version in the chatbot orchestrator:

```text
GOVERNANCE_CACHE_VERSION = "optimizer-audit-v2"
```

For requests without prior weights, include this version in the material supplied to `compute_query_hash`. This creates new cache keys without deleting old cache rows and guarantees that legacy payloads missing optimization-audit fields are not returned.

Future governance payload contract changes increment this constant.

## Logging

Log whether prior weights came from an explicit tool argument, session configuration, or were unavailable. Log the cache contract version on cache hits and writes. Do not log the weight values themselves.

## Testing

Add deterministic tests for:

1. Valid current session weights are normalized and returned.
2. Unapproved equal-weight proxies are rejected.
3. Invalid or empty session weights are rejected.
4. REST, streaming, and background assistant configurations include resolved previous weights.
5. The governance wrapper uses session-configured weights when the tool argument is absent.
6. Explicit tool weights override session-configured weights.
7. Requests with previous weights bypass cache reads and writes.
8. Requests without previous weights use a hash containing `optimizer-audit-v2`.

## Compatibility

No notebook, report, frontend, chart, database schema, or optimizer equation changes are included. Existing callers that do not provide previous weights continue to work. Existing cache rows remain intact but are naturally bypassed by the new versioned key.
