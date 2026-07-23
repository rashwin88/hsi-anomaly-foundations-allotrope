"""Module entry point: `python -m allotrope_worker`.

Kept thin so the real loop in `runner.py` is independently testable.
"""

from .runner import main

if __name__ == "__main__":
    main()
