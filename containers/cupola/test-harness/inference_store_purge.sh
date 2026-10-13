#!/bin/sh
# inference_store purge at startup (CUPOLA_INFERENCE_STORE=off): OGX writes one chat
# completion with the store on, the table is filled with <rows> copies carrying a marker,
# then OGX starts with the store off until a start deletes 0 (at most <starts> starts).
# Each start: seconds from container start to /v1/health, OGX's Cupola log line, rows
# left. At the end: rows and marker occurrences in the database files.
#   BACKEND=sqlite (default): sql_store.db on the harness volume; the image's SQLite has
#     SECURE_DELETE and the WAL is truncated, so the marker must be found 0 times in the
#     database file and in the WAL.
#   BACKEND=postgres: sql_default on a postgres:17 container (run.yaml copy with only
#     that block changed); after VACUUM FULL the marker must be found 0 times in the
#     database's files (base/); pg_wal is the server's and is reported, not expected 0.
# LOGDIR=<dir> keeps each start's OGX log.
# Usage: inference_store_purge.sh <image> <port> <rows> [bytes per row] [starts] [store env]
#   store env "on" leaves CUPOLA_INFERENCE_STORE unset (an image without the purge, for timing).
set -u
IMG=$1; PORT=$2; ROWS=$3; SIZE=${4:-19000}; STARTS=${5:-5}; MODE=${6:-off}; BACKEND=${BACKEND:-sqlite}
NAME=ogx-purge-$PORT; VOL=ogx-purge-$PORT; NET=ogx-purge-$PORT; PG=ogx-purge-pg-$PORT; B=http://localhost:$PORT
H=$(cd "$(dirname "$0")" && pwd)
CONF=$(mktemp -d)
cp "$H"/*.py "$H"/start.sh "$H"/run.yaml "$CONF"/
now() { python3 -c 'import time;print(time.time())'; }
psql_() { docker exec $PG psql -q -X -U ogx -d ogx -tA -c "$1"; }
cleanup() {
  docker rm -f $NAME $PG >/dev/null 2>&1; docker volume rm $VOL >/dev/null 2>&1
  docker network rm $NET >/dev/null 2>&1; rm -rf "$CONF"
}
cleanup; CONF=$(mktemp -d); cp "$H"/*.py "$H"/start.sh "$H"/run.yaml "$CONF"/
docker volume create $VOL >/dev/null; docker network create $NET >/dev/null
if [ "$BACKEND" = postgres ]; then
  docker run -d --name $PG --network $NET -e POSTGRES_USER=ogx -e POSTGRES_PASSWORD=ogx -e POSTGRES_DB=ogx postgres:17 >/dev/null
  until docker exec $PG pg_isready -U ogx -d ogx >/dev/null 2>&1; do sleep 1; done; sleep 2
  python3 - "$CONF/run.yaml" "$PG" <<'PY'
import sys
p, host = sys.argv[1], sys.argv[2]
s = open(p).read()
a = "    sql_default:\n      type: sql_sqlite\n      db_path: /tmp/ogx/sql_store.db\n"
assert s.count(a) == 1
open(p, "w").write(s.replace(a, f"    sql_default:\n      type: sql_postgres\n      host: {host}\n      port: 5432\n"
                              "      db: ogx\n      user: ogx\n      password: ogx\n"))
PY
fi
start() { # env args...
  docker rm -f $NAME >/dev/null 2>&1
  t0=$(now)
  docker run -d --name $NAME --network $NET --platform linux/amd64 -p $PORT:8321 -v "$CONF":/h:ro -v $VOL:/tmp/ogx "$@" \
    --entrypoint sh "$IMG" /h/start.sh >/dev/null
  for i in $(seq 1 600); do curl -s -m 2 $B/v1/health >/dev/null && break; sleep 0.5; done
  UP=$(python3 -c "print(f'{$(now)-$t0:.1f}')")
}
stop() { docker stop -t 30 $NAME >/dev/null; }
inspect() {
  if [ "$BACKEND" = postgres ]; then
    psql_ checkpoint
    D=$(docker exec $PG sh -c 'grep -rao CUPOLA-PURGE-MARKER "$PGDATA/base" | wc -l')
    W=$(docker exec $PG sh -c 'grep -rao CUPOLA-PURGE-MARKER "$PGDATA/pg_wal" | wc -l')
    echo "rows $(psql_ 'select count(*) from inference_store'); table $(psql_ "select pg_size_pretty(pg_total_relation_size('inference_store'))"), marker in base/ x$D; pg_wal/ (the server's) x$W"
  else
    docker run --rm --platform linux/amd64 -v $VOL:/tmp/ogx -v "$CONF":/h:ro --entrypoint python3 "$IMG" -I /h/inspect_inference_store.py /tmp/ogx/sql_store.db
  fi
}
seed() {
  if [ "$BACKEND" = postgres ]; then
    psql_ "insert into inference_store select (jsonb_populate_record(null::inference_store, to_jsonb(s) || jsonb_build_object(
      'id', 'chatcmpl-seed-' || g, 'created', s.created + g,
      'input_messages', json_build_array(json_build_object('role', 'user', 'content', 'CUPOLA-PURGE-MARKER ' || g || ' ' ||
        (select string_agg(md5(random()::text || g || i), '') from generate_series(1, $SIZE / 32) i)))))).*
      from inference_store s, generate_series(1, $ROWS - 1) g"
    echo "rows $(psql_ 'select count(*) from inference_store')"
  else
    docker run --rm --platform linux/amd64 -v $VOL:/tmp/ogx -v "$CONF":/h:ro --entrypoint python3 "$IMG" -I \
      /h/seed_inference_store.py /tmp/ogx/sql_store.db "$ROWS" "$SIZE"
  fi
}

echo "backend $BACKEND"
start
curl -s -o /dev/null -w "store on, one chat completion: HTTP %{http_code}\n" $B/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"openai-ok/good","messages":[{"role":"user","content":"CUPOLA-PURGE-MARKER hello"}]}'
sleep 2; stop
echo "seeded: $(seed)"
echo "before: $(inspect)"
n=1; while [ $n -le "$STARTS" ]; do
  if [ "$MODE" = on ]; then start; else start -e CUPOLA_INFERENCE_STORE=off; fi
  LINE=$(docker logs $NAME 2>&1 | grep -A3 -E "Cupola: (inference_store off|could not delete)" | tr -s ' ' | tr '\n' ' ' | sed -E 's/^.*(Cupola: inference_store off|Cupola: could not)/\1/; s/ category=core::routers.*$//')
  [ -n "${LOGDIR:-}" ] && docker logs $NAME > "$LOGDIR/start-$n.log" 2>&1
  stop
  echo "start $n: up in ${UP}s | ${LINE:-no Cupola delete line} | $(inspect)"
  case "$LINE" in *"deleted 0 "*) break ;; esac
  n=$((n + 1))
done
cleanup
