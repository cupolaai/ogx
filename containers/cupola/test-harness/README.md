# OGX image harness

Runs an OGX image locally — no cluster, no network — against a fake OpenAI-compatible provider
(`fake_provider.py`, inside the same container) and measures what Cupola depends on. Run it on every image build;
Cupola's backend tests replay the frames it records (k8s-ai `backend/internal/handler/testdata/ogx_frames/provider_errors/`).

```sh
cd containers/cupola/test-harness
IMG=cupola-cr.io/cupola/cupola-distribution-starter:<tag>

# provider errors on every path (chat completions and Responses, with and without a stream)
docker run -d --name ogx-h --platform linux/amd64 -p 18321:8321 -v "$PWD":/h:ro --entrypoint sh "$IMG" /h/start.sh
# wait for: curl -s localhost:18321/v1/health
MODELS="openai-r429/gemma openai-r401/bad401 openai-r403/bad403 openai-rmid/mid openai-ok/good" \
  sh measure.sh http://localhost:18321
docker rm -f ogx-h

# RAG on inline milvus, then the same search after a container restart (restart rehydration)
sh rag_restart.sh "$IMG" 18331

# seconds from container start to /v1/health, three runs
sh startup_time.sh "$IMG" 18334 3

# inference_store purge at startup (CUPOLA_INFERENCE_STORE=off): 1,323 rows of ~19 KB, then 50,000 rows of
# 4 KB (the 10 s limit stops it, the next starts finish it); last argument "on" = store env unset (timing)
sh inference_store_purge.sh "$IMG" 18341 1323
sh inference_store_purge.sh "$IMG" 18342 50000 4000
BACKEND=postgres sh inference_store_purge.sh "$IMG" 18343 1323     # same on a postgres:17 container
```

The fake provider answers by path prefix: `r429` OpenRouter's live 429 body (2026-10-12, free shared pool), `r401`
OpenRouter's 401 for an unknown key, `r403` a 403 whose text holds a key fragment, URLs and an IP (the backend's
cleaner test; the key is not real), `rmid` HTTP 200 then OpenRouter's mid-stream `error` chunk, `ok` a normal answer
and 8-dimension embeddings (`openai-ok/emb`).

Expected on `1.1.3-cupola.9`: every provider error keeps the provider's HTTP status (Responses without a stream too)
and carries `provider_status`, `provider` (when the provider names one) and `provider_detail`; a 429 has code
`rate_limit_exceeded`; `openai-ok/good` answers normally; the RAG search finds the document before and after the
restart. From `1.1.3-cupola.10`, with `CUPOLA_INFERENCE_STORE=off` the first start logs `Cupola: inference_store off,
deleted N stored chat completions` (or `…, more remain (time limit 10 s), continues at the next start`), the next one
`deleted 0`; each line ends with the backend's scrub (`sqlite: secure_delete on, WAL truncated` or `postgres: table
rewritten by VACUUM FULL; …`). The marker is then found 0 times in the SQLite database file and WAL, or in PostgreSQL's
`base/` (its `pg_wal` is the server's and keeps older copies). The openai SDK retries a 429 twice with backoff, so a full `measure.sh` run takes a few minutes under
emulation.
