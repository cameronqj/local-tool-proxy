# ADR 0004: Non-Stream Tool Path For Compat Models

## Status

Accepted

## Context

Streaming tool-call deltas are one of the hardest compatibility surfaces for
small local models and OpenAI-compatible harnesses.

## Decision

For configured compatibility models, tool-using requests may be forced to a
non-streaming upstream call before repair.

## Consequences

The proxy gets a complete response to inspect and repair, which improves the
first useful compatibility path. The tradeoff is that full streaming tool-call
reconstruction remains future work.
