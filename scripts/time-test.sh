#!/usr/bin/env bash

function workload() {
    ./scripts/wr-table.sh "$@" >/dev/null 2>/dev/null
    ./scripts/ws-table.sh "$@" >/dev/null 2>/dev/null
}

time -- workload "$@"
