# Figures pipeline

Extract figures from PDFs, attach captions, and generate structured explanations via a vision API. Output is JSONL suitable for graph ingestion.

## Requirements

- Python 3.10+
- PyMuPDF (`pymupdf`), Pillow, `requests`
- OpenAI API key (or compatible endpoint) for vision

## Install

From project root:

```bash
uv sync
# or
pip install -e ".[dev]"
```

**Requirements snippet** (figures pipeline only; already in `pyproject.toml`):

```
PyMuPDF>=1.24.0
Pillow>=10.0.0
requests>=2.32
openai>=2.21.0
```

## Usage

### CLI

```bash
python -m figures_pipeline --pdf path/to/paper.pdf --out ./figures_out --paper_id my_paper_001
```

Options:

- `--min_side_px 200` — minimum side length for embedded images (default 200)
- `--max_figures N` — cap number of figures processed
- `--force` — ignore cache and re-call vision API for all figures
- `-v` — verbose logging

Environment:

- `OPENAI_API_KEY` — API key for vision
- `FIGURE_VLM_MODEL` — model name (default `gpt-4o-mini`)
- `FIGURE_VLM_ENDPOINT` — optional; default is `https://api.openai.com/v1/chat/completions`
- `FIGURE_PROMPT_VERSION` — cache-busting prompt version (default `v1`)

### Library

```python
from src.alignment_atlas.figures import process_pdf_figures

records = process_pdf_figures(
    "path/to/paper.pdf",
    "output_dir",
    "paper_001",
    min_side_px=200,
    max_figures=10,
    force_refresh=False,
)
# records is list[dict]; same as figure_records.jsonl lines
```

### Output

- **Images**: `out_dir/p{N}_fig{M}.png` (or `.jpg`) for each extracted figure
- **JSONL**: `out_dir/figure_records.jsonl` — one JSON object per line with:
  - `paper_id`, `pdf_path`, `page`, `figure_id`, `image_path`, `image_bbox`
  - `caption`, `api_model`, `explanation`, `structured_json`
- **Cache**: `out_dir/vision_cache/` — one JSON file per (image+caption+model+prompt) hash to avoid re-calling the API

## Linking figures to the knowledge graph

Claims are already connected to papers in the KG via `paper -> claim` edges (`has_claim`). To also attach figure (image-to-text) records to the same papers:

1. **Use the same ID for papers and figures**  
   Run the figures pipeline with `--paper_id` equal to the document’s `doc_id` (e.g. the same id used in `docs.jsonl` and `claims.jsonl`). For example, if the paper’s `doc_id` is `perez_2022`, run:
   ```bash
   python -m figures_pipeline --pdf data/raw_pdfs/perez_2022.pdf --out data/figures_out/perez_2022 --paper_id perez_2022
   ```

2. **Merge figure records into processed data**  
   Copy or concatenate `figure_records.jsonl` from your figure output directories into a single file that the ingest pipeline reads:
   ```bash
   # Example: merge one paper’s figures into processed
   cp data/figures_out/perez_2022/figure_records.jsonl data/processed/figures.jsonl
   # Or append multiple: cat data/figures_out/*/figure_records.jsonl > data/processed/figures.jsonl
   ```

3. **Rebuild the KG**  
   The `build_kg` stage (Stage 05) reads `data/processed/figures.jsonl` when present. It adds:
   - **Nodes**: `figure:{paper_id}:{figure_id}` with type `"figure"`, and attributes such as `caption`, `explanation`, `image_path`, `structured_json`.
   - **Edges**: `paper -> figure` with `rel="has_figure"`.
   So each figure is linked to the same paper node as its claims. Run the stage (or full reingest) after updating `figures.jsonl`:
   ```bash
   python -m src.ingest.cli run-stage --stage build_kg
   ```

## Tests

From project root (with the project installed, e.g. `uv sync`):

```bash
uv run pytest src/alignment_atlas/figures/tests/ -v
```

If the package is not installed, set `PYTHONPATH` for the **whole** command (before `uv run`):

```bash
PYTHONPATH=src uv run pytest src/alignment_atlas/figures/tests/ -v
```

- `test_captioner.py` — caption matching heuristic (synthetic bbox + text blocks)
- `test_cache.py` — cache key, cache save/load, JSONL schema validation
