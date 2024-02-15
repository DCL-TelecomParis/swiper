#!/usr/bin/env bash
set -eu

# Generates the data for Table 2 in the paper

data_folder="./examples"
datasets="aptos.dat tezos.dat filecoin.dat algorand.dat"

echo "alpha_w = 1/4 alpha_n = 1/3"
for dataset in $datasets; do
  ./main.py wr --tw 1/4 --tn 1/3 "$data_folder/$dataset" --sum-only
done
echo

echo "alpha_w = 1/3 alpha_n = 3/8"
for dataset in $datasets; do
  ./main.py wr --tw 1/3 --tn 3/8 "$data_folder/$dataset" --sum-only
done
echo

echo "alpha_w = 1/3 alpha_n = 1/2"
for dataset in $datasets; do
  ./main.py wr --tw 1/3 --tn 1/2 "$data_folder/$dataset" --sum-only
done
echo

echo "alpha_w = 2/3 alpha_n = 3/4"
for dataset in $datasets; do
  ./main.py wr --tw 2/3 --tn 3/4 "$data_folder/$dataset" --sum-only
done
