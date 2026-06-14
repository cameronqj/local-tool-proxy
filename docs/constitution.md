# Project Constitution

This document describes the engineering principles for `local-tool-proxy`.

## 1. Compatibility First

The project exists to make local models easier to use with existing
OpenAI-compatible clients. The default path should preserve client expectations
and upstream semantics whenever possible.

## 2. Transparent By Default

`compat` mode is the default. It should be easy to understand what the proxy did,
what it changed, and what it passed through unchanged.

Diagnostics should be useful for local debugging without turning normal use into
raw prompt capture.

## 3. No Hidden Agent Runtime

The proxy must not become an autonomous agent. It does not own the task, plan the
task, execute tools, edit files, or decide whether a project is complete.

## 4. Experimental Behavior Is Opt-In

Behavior-changing features such as stabilization retries must be explicit,
documented, and easy to disable.

Experimental modes may help investigate a model failure. They should not be
documented as a guarantee of task success.

## 5. Never Execute Tools

The proxy may rewrite a model's representation of a tool call. It must never run
the tool call itself.

## 6. Logs Should Help Without Betraying The User

Logs should identify routing decisions, rewrites, collapse categories, and trace
ids. Raw prompts, messages, and tool schemas should only be logged behind an
explicit debug flag.

## 7. Real Failures Are First-Class Test Fixtures

The most valuable tests are the strange outputs real local models produce. Add
them as focused fixtures or tests when possible.

## 8. Honest Documentation Beats Hype

The README should be clear about what works, what is experimental, and what has
not been proven yet.

Public framing should use the `local-tool-proxy` name. Earlier internal names
belong only in compatibility or history notes when they help existing users
migrate.

## 9. Small, Inspectable Code Wins

Prefer simple functions, explicit strategy order, and readable logs over clever
frameworks. This project should remain understandable to someone debugging a
local model at 1 a.m.

## 10. Local-First Matters

The target user is often running on consumer hardware with local models. Keep the
proxy lightweight and easy to run alongside Ollama or a similar local server.

## 11. Public Surface Stays Small

The public API should stay close to OpenAI-compatible endpoints and a small CLI.
New configuration should earn its place by solving a real compatibility problem
or making experiments easier to reproduce.
