#!/bin/sh
# RAG through OGX on inline milvus, then the same search after a container restart
# (milvus restart rehydration). Usage: rag_restart.sh <image> <port>
set -u
IMG=$1; PORT=$2; NAME=perr-rag-$PORT; B=http://localhost:$PORT
H=$(cd "$(dirname "$0")" && pwd)
wait_up() { for i in $(seq 1 90); do curl -s -m 3 $B/v1/health >/dev/null && return 0; sleep 3; done; echo "OGX did not start"; return 1; }
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --platform linux/amd64 -p $PORT:8321 -v "$H":/h:ro --entrypoint sh "$IMG" /h/start.sh >/dev/null
wait_up || exit 1
VS=$(curl -s $B/v1/vector_stores -H 'Content-Type: application/json' \
  -d '{"name":"rag-test","embedding_model":"openai-ok/emb","embedding_dimension":8}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
echo "vector store $VS"
printf 'Cupola keeps milvus vector stores across restarts.\n\nThe quarterly revenue of the blue division grew by twelve percent.\n' > /tmp/rag-doc.txt
FID=$(curl -s $B/v1/files -F purpose=assistants -F file=@/tmp/rag-doc.txt | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
echo "file $FID"
curl -s $B/v1/vector_stores/$VS/files -H 'Content-Type: application/json' -d "{\"file_id\":\"$FID\"}" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("attach status", d.get("status"), d.get("last_error"))'
search() {
  curl -s $B/v1/vector_stores/$VS/search -H 'Content-Type: application/json' \
    -d '{"query":"blue division quarterly revenue","max_num_results":3}' | python3 -c '
import json,sys
d=json.load(sys.stdin)
rows=d.get("data") or []
txt=" ".join(c.get("text","") for r in rows for c in r.get("content",[]))
found = "blue division" in txt
print("  results=%d found_text=%s" % (len(rows), found) if rows else "  NO RESULTS: " + json.dumps(d)[:300])'
}
echo "search before restart:"; search
docker restart $NAME >/dev/null; wait_up || exit 1
echo "search after restart (cold cache):"; search
docker logs $NAME 2>&1 | grep -iE "VectorStoreNotFound|chunk_content|Traceback" | head -3
docker rm -f $NAME >/dev/null; rm -f /tmp/rag-doc.txt
