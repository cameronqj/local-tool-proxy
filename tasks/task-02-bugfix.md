TASK: Fix a small but real bug in provided code and add a test.

You are given (or will create) a tiny Python file `buggy.py` containing this function:

```python
def find_longest_word(text: str) -> str:
    """Return the longest word in the text. Words are split on whitespace and punctuation should be stripped."""
    words = text.split()
    if not words:
        return ""
    return max(words, key=len)
```

Known issues (do not mention them in reasoning):
- It does not strip common punctuation (.,!? etc.) so "hello," beats "hello".
- No handling for ties (should return the first occurrence in case of tie).
- No docstring update or type hints improvement.

Your job:
1. Create the file `buggy.py` with the above function (if not already present).
2. Write a clear, minimal fix.
3. Add or update `test_buggy.py` with at least 3 pytest cases that would have failed before the fix (including punctuation and tie cases).
4. Verify: run pytest and show all tests pass.
5. Show the final diff or the corrected function.

Success criteria: The tests pass and demonstrate the bug is fixed. Keep changes minimal and correct.
