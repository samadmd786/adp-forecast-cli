"""Allow running the CLI as `python -m adp_ner`."""

from adp_ner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
