# Manual driver scripts

Ad-hoc scripts for driving the proxy against a **real, locally running** Ollama +
model. They are not part of the automated test suite (they need live inference and
are non-deterministic), and are kept here for manual reproduction and debugging.

Run the proxy first (see the top-level [README](../../README.md)), then, from the
repo root:

```bash
python3 eval/manual/test_rigid_prompt.py        # rigid-prompt rewrite drive
python3 eval/manual/test_with_clean_task.py     # single clean-task tool loop
python3 eval/manual/test_task02_via_proxy.py    # multi-turn Task 02 bugfix loop
```

For a deterministic, no-Ollama demonstration of the repair path, use the
top-level demo instead:

```bash
make demo   # or: python3 demo.py
```
