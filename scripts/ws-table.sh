#!/usr/bin/env bash
set -eu

# Generates the data for the WS part of Table 2 in the paper

data_folder="./examples"
datasets="aptos.dat tezos.dat filecoin.dat algorand.dat"

echo "========== WS: =========="
echo

echo "alpha = 1/4 beta = 1/3"
for dataset in $datasets; do
  ./main.py ws --alpha 1/4 --beta 1/3 "$data_folder/$dataset" --sum-only "$@"
done

echo "alpha = 1/3 beta = 1/2"
for dataset in $datasets; do
  ./main.py ws --alpha 1/3 --beta 1/2 "$data_folder/$dataset" --sum-only "$@"
done

echo "alpha = 2/3 beta = 3/4"
for dataset in $datasets; do
  ./main.py ws --alpha 2/3 --beta 3/4 "$data_folder/$dataset" --sum-only "$@"
done

echo "========================="
