# aero-spec-rag

Ask a natural-language question about an aerospace or guidance parameter — a drag
coefficient, ISA density at altitude, a proportional-navigation gain — and get
back a **structured, cited, machine-usable** answer instead of prose. A LangChain
ingestion pipeline chunks and embeds a small markdown corpus into a local Chroma
store; a LangGraph state machine then retrieves the relevant passages, checks any
number it extracts against a hardcoded table of physical plausibility bounds,
discards values that fail, and emits a JSON object carrying the value, its unit,
the source document, a confidence score, a `verified` flag, and the passages it
relied on. It is built as a companion to a missile-guidance simulator: the
intended consumer is a simulation config loader, not a human reading paragraphs.

> ⚠️ **Illustrative data, not real specs.** Every number in `src/corpus/` was
> written for this repository. The atmosphere and aerodynamics documents are
> standard textbook physics restated in original wording; the interceptor and
> target parameter sets are **invented order-of-magnitude placeholders**, clearly
> labelled `source_type: illustrative`. Nothing here describes, or is derived
> from, any real weapon system. Do not use it for anything but simulation
> plumbing and portfolio demonstration.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API key is required. The default embedding backend is a deterministic offline
hashing embedder, so ingestion, the graph, and the test suite all run with no
network access (see [DECISIONS.md](DECISIONS.md)).

## Ingest the corpus

```bash
python -m src.ingest
```

Chunk ids are content-addressed, so this is idempotent — re-running upserts the
same ids and prunes chunks whose source text has been deleted. Use
`python -m src.ingest --rebuild` to start from an empty store.

## Run the API

```bash
uvicorn src.api:app --reload --port 8000
```

Interactive docs are then at `http://127.0.0.1:8000/docs`.

## Example request

```bash
curl -s -X POST http://127.0.0.1:8000/ground-spec \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the ISA air density at 10 km altitude?"}'
```

```json
{
  "query": "What is the ISA air density at 10 km altitude?",
  "quantity": "air_density",
  "value": 0.4135,
  "value_low": null,
  "value_high": null,
  "unit": "kg/m^3",
  "source_doc": "isa-density-vs-altitude.md",
  "confidence": 0.787,
  "verified": true,
  "answer": "air density = 0.4135 kg/m^3 (ISA at 10000 m altitude), from isa-density-vs-altitude.md.",
  "context": "ISA at 10000 m altitude",
  "plausible_bounds": [1e-06, 1.5],
  "flags": [
    {
      "code": "wide_spread",
      "severity": "warning",
      "message": "Retrieved values for this quantity span more than an order of magnitude; the query may be under-specified (for example, missing an altitude or a body shape)."
    }
  ],
  "citations": [
    {
      "source_doc": "isa-density-vs-altitude.md",
      "title": "ISA Density and Pressure Against Altitude",
      "source_type": "derived",
      "chunk_index": 2,
      "snippet": "| Quantity | Value | Unit | Notes |\n| air density | 1.225 | kg/m^3 | ISA at 0 m altitude |...",
      "retrieval_score": 0.4191
    }
  ],
  "disclaimer": "Illustrative corpus. Values are teaching or order-of-magnitude figures, not the specifications of any real system."
}
```

`GET /health` reports the collection name, the active embedding backend, and the
configured `top_k`.

You can also query the graph directly without the server:

```bash
python -m src.graph "What proportional navigation gain should I use?"
```

## How the graph works

```
START → retrieve → verify → propose → finalize → END
```

| Node | Responsibility |
| --- | --- |
| `retrieve` | Vector search against Chroma (`top_k=5` by default), over-fetched and lexically rescored; also maps the query onto a registry quantity. |
| `verify` | Parses candidate `\| Quantity \| Value \| Unit \| Notes \|` rows out of the retrieved chunks and checks each against `src/bounds.py`. Out-of-range values are **discarded**, not merely annotated; unit mismatches and order-of-magnitude spreads are flagged. |
| `propose` | Ranks surviving candidates by how well their row context answers the query, then scores confidence from match strength, retrieval rank, provenance, and any warnings raised. |
| `finalize` | Re-validates the payload against the `GroundedSpec` pydantic model before it leaves the graph. |

If the query matches no quantity in the registry, or no value survives
verification, the pipeline returns `verified: false` with `value: null` and the
citations it found — it does not guess.

## Layout

```
src/
  corpus/            10 markdown documents with YAML frontmatter
  config.py          env-overridable settings
  corpus_loader.py   frontmatter loading and validation
  embeddings.py      offline hashing embedder + optional hosted backends
  ingest.py          chunk → embed → persist to .chroma/ (idempotent)
  bounds.py          quantity registry and plausibility bounds
  extract.py         numeric candidate extraction from chunk text
  graph.py           the LangGraph StateGraph
  schemas.py         pydantic response models
  api.py             FastAPI app
tests/
```

## Tests

```bash
pytest
```

Tests build their own vector store under `.chroma-test/` and never touch a
developer's `.chroma/`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AERO_EMBEDDINGS` | `local` | `local`, `openai`, or `voyage` |
| `AERO_CHROMA_DIR` | `.chroma` | Vector store location |
| `AERO_CORPUS_DIR` | `src/corpus` | Corpus location |
| `AERO_CHUNK_SIZE` | `800` | Splitter chunk size |
| `AERO_CHUNK_OVERLAP` | `100` | Splitter overlap |
| `AERO_TOP_K` | `5` | Retrieval depth |

## License

MIT.
