# ADR 0002: Compat Mode First

## Status

Accepted

## Context

Users need a safe default that does not surprise them with hidden behavioral
changes.

## Decision

`compat` mode is the default. It performs transparent forwarding plus targeted
tool-call normalization for configured models.

## Consequences

The default behavior remains easy to reason about. More active interventions must
be explicitly enabled.
