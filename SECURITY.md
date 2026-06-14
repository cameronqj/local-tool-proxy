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

The default host is currently `0.0.0.0` for convenience while experimenting. If
you do not need LAN access, prefer binding to loopback:

```bash
local-tool-proxy --host 127.0.0.1
```

If you expose the proxy beyond your machine, put it behind controls you trust and
review what your client harness and upstream model server can access.

## Sensitive Data

By default, the proxy does not log full prompts, messages, or tool schemas.

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
