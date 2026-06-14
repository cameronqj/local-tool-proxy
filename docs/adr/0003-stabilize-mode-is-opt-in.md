# ADR 0003: Stabilize Mode Is Opt-In

## Status

Accepted

## Context

When a model stops using tools, the proxy can sometimes nudge it back with a
single retry. This changes the upstream model interaction and should not happen
silently.

## Decision

Stabilization is only active with `--mode stabilize`.

The default retry count is conservative, and failed retries fall back to the
original response.

## Consequences

Users can reproduce baseline behavior with `--mode compat`. Experiments remain
clearly marked in logs.
