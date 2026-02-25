"""CLI entrypoint: python -m figures_pipeline --pdf <path> --out <dir> --paper_id <id>"""

from __future__ import annotations

import sys

from src.alignment_atlas.figures.cli import main

if __name__ == "__main__":
    sys.exit(main())
