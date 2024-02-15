#!/usr/bin/env bash
set -eu

declare -a params=("1/4 1/3" "1/3 3/8" "1/3 1/2" "2/3 3/4")

data_folder="./examples"

for dataset in $data_folder/*.dat; do
  echo "Processing $dataset"
  for param in "${params[@]}"; do
    IFS=' ' read -r tw tn <<< "$param"
    echo "wr --tw $tw --tn $tn $dataset --debug"
    ./main.py wr --tw "$tw" --tn "$tn" "$dataset" --debug
  done
  echo
done
