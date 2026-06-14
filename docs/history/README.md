# History Notes

This directory is for curated experiment history, not raw artifact dumps or
private planning prompts.

The public repo should keep enough history for readers to audit the project
claim:

- What failure was observed.
- Which harness/model combination produced it.
- Which proxy behavior changed the outcome.
- Whether artifact-level verification passed.

Prefer short narrative notes, reduced fixtures, and links to reproducible commands
over wholesale logs. Raw `eval/logs/proxy_*.log`, full `eval/results/*.json`, and
large run transcripts are useful while developing, but they should not be
committed wholesale unless they are intentionally curated and reviewed for noisy
machine-specific paths or prompt content.

Current public narrative:

- Mock and automated tests show that JSON-in-content tool intent can be repaired
  into OpenAI-compatible `tool_calls`.
- Real OpenCode runs show partial progress: compatibility paths, tool turns,
  drift/collapse classifications, and one stabilize retry were observed.
- Captured real-harness artifact verification remains unsolved; several runs
  exited cleanly while expected task files were absent from verification output.

That is the line to preserve: protocol repair works in controlled tests, real
harness behavior is improving, and verified task artifacts remain the bar.
