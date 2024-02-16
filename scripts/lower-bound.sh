#!/usr/bin/env bash
set -eu

declare -a params=("1/4 1/3" "1/3 3/8" "1/3 1/2" "2/3 3/4")

ns="20 40 60 80 100 120 140 160 180 200"

for param in "${params[@]}"; do
  IFS=' ' read -r tw tn <<< "$param"
  echo "$param"
  for n in $ns; do
    ./lowerbounds/lowerbounds_main.py --tw "$tw" --tn "$tn" -n "$n" -W 1000 --parallel -v
  done
  echo
done
