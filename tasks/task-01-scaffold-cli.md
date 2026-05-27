TASK: Scaffold a minimal, correct Python CLI tool.

Goal: Create a new small project in the current directory (or a fresh subdir `wordcount-cli`) that does the following:

- A CLI `wordcount` that takes a file path and prints line count, word count, and character count.
- Use only the Python standard library + argparse.
- Include a `requirements.txt` (even if empty or just for future).
- Include a `README.md` with usage example.
- Include one simple unit test in `tests/test_wordcount.py` using pytest (add pytest to requirements).
- The code must run: `python -m wordcount_cli --help` and `python -m wordcount_cli somefile.txt` must work after `pip install -e .` or direct run.

Constraints for success:
- All files must be created using the available tools (write/edit/run).
- After creation, run the test and show it passes.
- The final state must have a working, importable package or script.
- Do not add extra commentary or numbered lists in your final summary — just confirm the verification commands and their output.

Start by exploring the current directory, then create the necessary structure.
