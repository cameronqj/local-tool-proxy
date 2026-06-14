# Security Policy

`local-tool-proxy` is local-first experimental middleware. It is designed to run
on a trusted developer machine between a local client and a local model server.

It has not been hardened for untrusted networks, multi-user hosting, or
production traffic.

## Supported Use

The expected deployment is:

```text
agent harness -> local-tool-proxy -> local model server
```

The default host is `127.0.0.1`, which keeps the proxy local to your machine:

```bash
local-tool-proxy --host 127.0.0.1
```

If you need LAN access, opt into a broader bind address with `--host 0.0.0.0`.
If you expose the proxy beyond your machine, put it behind controls you trust and
review what your client harness and upstream model server can access.

## Sensitive Data

By default, the proxy does not log full prompts, messages, or tool schemas.

The `--trace-file` flag writes sanitized JSONL metadata about request, rewrite,
collapse, and stabilization events. It is intended for shareable diagnostics, but
you should still review traces before publishing them.

The `--debug-log-model-outputs` flag is intentionally loud and dangerous. It can
log raw harness requests and model outputs. Do not use it with private code,
credentials, customer data, or anything you would not put in a local plaintext
log file.

Evaluation logs, debug output, and copied model traces can contain private code
or credentials even when the proxy defaults are conservative. Review artifacts
before sharing them.

## Tool Execution

The proxy does not execute tools. It only forwards and rewrites model responses.
Actual tool execution remains owned by the client harness.

That boundary matters for security review: permissions, filesystem access,
shell execution, and approval flows belong to the harness, not this proxy.

## Reporting Vulnerabilities

For now, please report security issues privately to the repository owner rather
than opening a public issue. If this project becomes broadly used, this file
should be updated with a dedicated reporting address.
