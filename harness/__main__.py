"""Allow ``python -m harness run "..."`` and ``python -m harness serve``.

Thin dispatch into harness.cli.main - the package itself has no other reason
to be run as a module."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
