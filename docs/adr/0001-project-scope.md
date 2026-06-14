# ADR 0001: Project Scope

## Status

Accepted

## Context

Local models can be useful with agent harnesses, but tool-call compatibility is
fragile. There is a temptation to solve the whole agent loop inside middleware.

## Decision

`local-tool-proxy` is scoped to OpenAI-compatible protocol repair for local-model
tool calls.

It will not become:

- A general model router.
- A tool executor.
- An autonomous coding agent.
- A task planner.

## Consequences

The project stays small and inspectable. Users who need broader routing should
use tools such as LiteLLM. Users who need task execution should use an agent
harness.
