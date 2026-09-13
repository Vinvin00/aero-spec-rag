# Decisions

Every default chosen while building this repository, and why. The task brief left
these open; each was resolved without pausing, as instructed.

## Environment

**Python 3.14.6, not 3.11 or 3.12.** The brief asked for 3.11+. The only
interpreter on this machine is Homebrew's 3.14, and no pyenv or uv was
installed. Rather than spend ten minutes building an older interpreter, 3.14 was
tried first — `chromadb`, `langchain`, `langgraph`, and `fastapi` all have
working wheels for it, and the full test suite passes. `requires-python` is
declared as `>=3.11` because nothing in the code uses 3.12+ syntax.

**Both `pyproject.toml` and `requirements.txt`.** `pyproject.toml` carries loose
floors for anyone installing the package; `requirements.txt` carries the exact
versions this was developed against, which is what the README tells you to
install. `requirements.lock.txt` is a full `pip freeze` of the working venv.

## Model and embedding provider

**No LLM call is made anywhere in the pipeline by default.**
`langchain-anthropic` is installed as the brief specified, but with
`AERO_LLM=none` (the default) the graph is fully deterministic: retrieval, table
parsing, bounds checking, and ranking are all rule-based. This was chosen
deliberately over an LLM-synthesised answer because the deliverable is a
*machine-usable grounded value* feeding a simulation, and a parsed table cell
with a citation is strictly more trustworthy than a generated number.

**When an LLM is enabled it is confined to two jobs, neither of which can
produce a number.** This was the design constraint for the Ollama work:

*Quantity classification (`retrieve`).* The alias registry is exact-match, so
"how heavy is the bird at launch?" matched nothing and the query failed. The LLM
now maps such a query onto a registry key, and is **constrained to return a key
that already exists** — an invented label like `warhead_yield` is rejected. The
value still comes from a parsed, bounds-checked table row.

*Answer narration (`finalize`).* The `answer` sentence is rephrased from fields
that are already selected and verified. Three guards apply: the verified value
must survive into the sentence verbatim or the prose is discarded; the citation
is appended deterministically rather than asked of the model; and any backend
failure falls back to the deterministic sentence with a `narration_unavailable`
flag. `narration_model` records which model phrased the answer, and is null when
it was the deterministic one.

The guards are not theoretical — llama3.1:8b *did* drop the source filename when
asked to include it, which the first version of the guard correctly rejected.
Rather than fight the prompt, the citation was moved out of the model's hands
entirely. That is the more robust design and would have been worth doing even
with a stronger model.

**Retrieval is steered by the identified quantity.** Once a quantity is known
(by alias or by LLM), its canonical vocabulary is appended to the vector search
text and added to the lexical rescoring with a lower weight. Without this, a
paraphrased query that the LLM classified correctly still retrieved the wrong
chunks, because "how heavy is the bird" shares no words with a "launch mass"
table row. This helps registry-matched queries too, and is not Ollama-specific.

**Ollama is supported for both model slots, but is opt-in rather than the
default.** `AERO_LLM=ollama` and `AERO_EMBEDDINGS=ollama` give the project a real
LLM and learned embeddings with no API key and no cloud call, which fits the
no-key constraint far better than the hosted options. They are not the *default*
because defaults must work on a clean checkout: a default of `ollama` would make
`pytest` fail for anyone without a running server and the right models pulled.
So the deterministic offline path stays the default, Ollama is one environment
variable away, and the live Ollama tests skip themselves when the server or
model is absent.

**Default embeddings are a local deterministic hashing embedder, not a hosted
model.** Anthropic does not offer an embeddings API at all, so "use Anthropic"
cannot be satisfied for the vector step. `ANTHROPIC_API_KEY` was not set in the
environment and no OpenAI or Voyage key was present either, so the alternative
was a pipeline that cannot run. The offline embedder in `src/embeddings.py`
hashes unigrams and bigrams into a 512-dimension unit vector — weaker than a
learned model on paraphrase, but deterministic, free, and dependency-light, so
ingestion and the whole test suite run anywhere with no network. `AERO_EMBEDDINGS=openai`
or `=voyage` swaps in a learned model without touching anything else.

**The retriever compensates with a lexical rescoring pass.** Because the hashing
embedder is weak at paraphrase, `retrieve` over-fetches `top_k * 4` chunks by
vector distance and re-ranks them by query-term overlap before truncating to
`top_k`. This is documented in the node rather than hidden, and would still be
reasonable (as a cheap hybrid retriever) with a learned embedding in place.

## Corpus

**Ten documents, all written from scratch.** The brief asked for 6-10. No source
text was copied. The atmosphere and aerodynamics documents restate standard
textbook physics in original wording; the vehicle parameter documents are
invented placeholders and say so in bold in their own body text as well as in
`source_type: illustrative`.

**Frontmatter is `title` / `topic` / `source_type`**, exactly as specified.
`source_type` is constrained to `illustrative` or `derived` and validated at load
time — a malformed document raises rather than silently ingesting.

**Numbers live in four-column markdown tables** (`| Quantity | Value | Unit |
Notes |`) embedded in prose. This is the key structural decision: it gives the
`verify` node something parseable to extract without an LLM, while keeping the
documents readable as documents. The parser accepts any four-cell row whose
second cell is a number or a range, so a table split away from its header by
chunking still parses.

## Ingestion

**Idempotency via content-addressed ids.** Each chunk's id is
`sha1(source_doc:index:content)`. Re-ingesting upserts identical ids (no
duplicates), and ids present in the store but absent from the new chunk set are
pruned, so editing a document cleans up its stale chunks. `--rebuild` deletes the
store outright.

**Custom splitter separators.** `RecursiveCharacterTextSplitter` is used at the
specified `chunk_size=800` / `chunk_overlap=100`, but with markdown heading
separators (`\n## `, `\n### `) ahead of the defaults so chunks break at section
boundaries rather than mid-table where possible.

**Store path is `.chroma/` and is gitignored.** It is fully rebuildable from the
corpus in about a second, so committing it would only add churn.

## Graph

**Linear `retrieve → verify → propose → finalize`, no interrupt.** As specified.
**A human-in-the-loop interrupt before `propose` is the natural v2 addition** —
LangGraph's `interrupt_before` plus a checkpointer would let an engineer approve
or override a low-confidence or flagged value before it reaches a simulation
config. It was deliberately left out of v1 per the brief.

**State is a `TypedDict`**, not a pydantic model, because that is the LangGraph
idiom and it keeps node returns as partial dicts. The *response* is a pydantic
model (`GroundedSpec`), validated in `finalize` and again by FastAPI's
`response_model`, so the external contract is strictly typed even though the
internal state is not.

**`verify` discards out-of-bounds values rather than passing them through with a
flag.** The brief said "flags if retrieved values fall outside bounds"; flagging
alone would still hand a physically impossible number to a caller that might not
read flags. A discarded value is both flagged *and* excluded, and if nothing
survives the response is `verified: false` with `value: null`.

**An unidentifiable query returns citations, not a guess.** If the query matches
no quantity in the registry, extraction is skipped entirely — otherwise the
pipeline would return whatever number happened to be nearby, which is exactly the
failure mode this project exists to prevent.

**Bounds are generous on purpose.** The table in `src/bounds.py` is sized to
catch retrieval and parsing failures (a Cd of 47, a density of 900), not to
adjudicate whether a plausible value is the *best* value. Cd is bounded
`0.01-1.5` and navigation gain `2-6`, as suggested in the brief.

**Ranges are stored as midpoint plus explicit endpoints.** A row reading
`3 to 5` yields `value: 4.0`, `value_low: 3.0`, `value_high: 5.0`, so a caller
can use the scalar directly or respect the interval. Range answers take a small
confidence penalty.

**Confidence is a transparent heuristic, not a calibrated probability.** It
starts at 0.35, adds up to 0.45 for query-to-row-context term overlap and up to
0.2 for retrieval rank, and subtracts for illustrative provenance, active
warnings, and range-valued answers. The formula is in one function
(`propose_node`) so it can be replaced wholesale.

## API

**`POST /ground-spec` and `GET /health` only**, as specified. `top_k` is exposed
as an optional per-request override because it is genuinely useful when probing
retrieval quality.

**CORS is `allow_origins=["*"]` with `allow_credentials=False`.** Open for local
dev as the brief asked; credentials are disabled so the wildcard cannot be abused
into an authenticated cross-origin read, and the code comment says to restrict
origins before deploying anywhere.

**Pipeline failures return 503, not 500**, with a message pointing at
`python -m src.ingest` — the overwhelmingly likely cause is an un-ingested store.

## Tests

**Tests run against a real Chroma store, not mocks.** A session-scoped fixture
ingests the corpus into `.chroma-test/` once. Because embeddings are offline and
deterministic, this is fast (sub-second) and fully reproducible, and it exercises
the actual retrieval path rather than a stub. The three queries named in the
brief — drag coefficient, ISA density at altitude, PN gain — assert on exact
expected values, not just on schema shape.

**Adversarial cases are covered too**: an out-of-bounds row is injected directly
into `verify_node` to prove it is discarded, and an unanswerable query is
asserted to degrade to `verified: false` rather than fabricate.

## Known gaps

**`nomic-embed-text` could not be pulled on this machine.** `AERO_EMBEDDINGS=ollama`
is implemented and wired through, but the download from the Ollama registry
timed out at the manifest stage (`dial tcp 172.64.66.1:443: i/o timeout`) across
repeated attempts — a network path problem, not a code one. The chat-model path
(`AERO_LLM=ollama` with `llama3.1:8b`) *was* verified end to end against a live
server. The embeddings test is written and skips itself until the model is
present; run `ollama pull nomic-embed-text` on a working network, then
`python -m src.ingest --rebuild`, and it will execute.

## Remote

The GitHub CLI was authenticated, so a **private** repository was created and the
initial commit pushed. Private rather than public because the brief said private;
it can be flipped with `gh repo edit --visibility public`.
