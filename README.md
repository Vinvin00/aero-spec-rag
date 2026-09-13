# aero-spec-rag

Ask a natural-language question about an aerospace or guidance parameter — a drag
coefficient, ISA density at altitude, a proportional-navigation gain — and get
back a **structured, cited, machine-usable** answer instead of prose. A LangChain
ingestion pipeline chunks and embeds a small markdown corpus into a local Chroma
store; a small retrieve → verify → propose → finalize pipeline then retrieves the relevant passages, checks any
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
hashing embedder, so ingestion, the pipeline, and the test suite all run with no
network access (see [DECISIONS.md](DECISIONS.md)).

### Optional: run a real model locally with Ollama

Both model slots can be served by [Ollama](https://ollama.com), so the project
gains a genuine LLM and learned embeddings without any API key or cloud call:

```bash
ollama pull llama3.1:8b        # chat model
ollama pull nomic-embed-text   # embedding model

export AERO_LLM=ollama                 # enables classification + narration
export AERO_EMBEDDINGS=ollama          # learned embeddings instead of hashing
python -m src.ingest --rebuild         # required: the store is dimension-specific
uvicorn src.api:app --reload --port 8000
```

> **If `ollama pull` times out**, your network may not have a route to
> Cloudflare R2, where Ollama hosts model blobs (`dial tcp 172.64.x.x:443: i/o
> timeout`). Pulling the equivalent GGUF from HuggingFace goes over a different
> CDN and works:
>
> ```bash
> ollama pull hf.co/CompendiumLabs/bge-small-en-v1.5-gguf
> export AERO_OLLAMA_EMBED_MODEL=hf.co/CompendiumLabs/bge-small-en-v1.5-gguf
> ```
>
> bge-small-en-v1.5 is 24 MB and 384-dimensional; this is the combination the
> project was verified against.

Enabling the LLM adds exactly two capabilities, and **neither can produce a
number**:

1. **Quantity classification fallback.** The alias registry in `src/bounds.py`
   is exact-match. When it misses a phrasing — *"how heavy is the bird at
   launch?"* — the LLM maps the query onto a registry key, and is constrained to
   return a key that already exists. The identified quantity then steers
   retrieval, and the value still comes from a parsed, bounds-checked table row.
2. **Answer narration.** The `answer` sentence is rephrased in natural language
   from fields that are *already* selected and verified. The sentence is
   rejected unless the verified value survives into it verbatim, and the
   citation is appended deterministically rather than asked of the model. On
   rejection or backend failure the deterministic sentence is kept and a
   `narration_unavailable` flag is raised.

`narration_model` in the response records which model phrased the answer, and is
`null` whenever the deterministic sentence was used. Every structured field
(`value`, `unit`, `source_doc`, `verified`, `confidence`) is produced the same
way regardless of backend.

With the LLM off — the default — no model is contacted at all.

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

### Or with Docker

```bash
docker build -t aero-spec-rag .
docker run --rm -p 8001:8001 aero-spec-rag
```

The corpus is ingested at build time (deterministic offline embeddings, no
API key, no network call), so the container starts serving immediately on
`:8001`. This image is also what
[missile-sim-viz](https://github.com/Vinvin00/missile_guidance_sim)'s
`docker-compose.yml` builds as its `rag-backend` service, as a sibling
checkout — see that repo's README.

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

`GET /health` reports the collection name, the active embedding and LLM
backends, and the configured `top_k`.

You can also query the pipeline directly without the server:

```bash
python -m src.graph "What proportional navigation gain should I use?"
```

## How the pipeline works

```
START → retrieve → verify → propose → finalize → END
```

| Node | Responsibility |
| --- | --- |
| `retrieve` | Vector search against Chroma (`top_k=5` by default), over-fetched and lexically rescored; also maps the query onto a registry quantity. |
| `verify` | Parses candidate `\| Quantity \| Value \| Unit \| Notes \|` rows out of the retrieved chunks and checks each against `src/bounds.py`. Out-of-range values are **discarded**, not merely annotated; unit mismatches and order-of-magnitude spreads are flagged. |
| `propose` | Ranks surviving candidates by how well their row context answers the query, then scores confidence from match strength, retrieval rank, provenance, and any warnings raised. |
| `finalize` | Re-validates the payload against the `GroundedSpec` pydantic model, and — only when an LLM backend is configured — rephrases the `answer` sentence under the guards described above. |

If the query matches no quantity in the registry, or no value survives
verification, the pipeline returns `verified: false` with `value: null` and the
citations it found — it does not guess.

## Layout

```
src/
  corpus/            10 markdown documents with YAML frontmatter
  config.py          env-overridable settings
  corpus_loader.py   frontmatter loading and validation
  embeddings.py      offline hashing embedder + ollama backend
  llm.py             optional LLM: constrained classification + guarded narration
  ingest.py          chunk → embed → persist to .chroma/ (idempotent)
  bounds.py          quantity registry and plausibility bounds
  extract.py         numeric candidate extraction from chunk text
  graph.py           the retrieve/verify/propose/finalize pipeline
  schemas.py         pydantic response models
  api.py             FastAPI app
tests/
```

## Tests

```bash
pytest
```

Tests build their own vector store under `.chroma-test/` and never touch a
developer's `.chroma/`. The LLM guard tests stub the model, so the whole suite
runs offline; the live Ollama tests skip themselves automatically when the
server or the model is not present.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AERO_EMBEDDINGS` | `local` | `local`, `ollama`, `openai`, or `voyage` |
| `AERO_LLM` | `none` | `none`, `ollama`, or `anthropic` |
| `AERO_LLM_MODEL` | `llama3.1:8b` | Chat model name for the active LLM backend |
| `AERO_LLM_TIMEOUT` | `30` | Seconds before an LLM call is abandoned |
| `AERO_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `AERO_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model when `AERO_EMBEDDINGS=ollama` |
| `AERO_CHROMA_DIR` | `.chroma` | Vector store location |
| `AERO_CORPUS_DIR` | `src/corpus` | Corpus location |
| `AERO_CHUNK_SIZE` | `800` | Splitter chunk size |
| `AERO_CHUNK_OVERLAP` | `100` | Splitter overlap |
| `AERO_TOP_K` | `5` | Retrieval depth |

## License

MIT.
