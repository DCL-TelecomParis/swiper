#!/usr/bin/env bash
set -eu
set -o pipefail

declare -a tests=(
  "wr --tw 1/4 --tn 1/3"
  "wr --tw 1/3 --tn 3/8"
  "wr --tw 0.3 --tn 0.4"
  "wr --tw 2/3 --tn 3/4"

  "wq --tw 0.3 --tn 0.2"
  "wq --tw 2/3 --tn 1/3"

  "ws --alpha 1/3 --beta 2/3"
  "ws --alpha 0.6 --beta 0.7"
  "ws --alpha 0.25 --beta 1/3"
)

declare -a extra_params=(
  "--linear"
  ""
)

data_folder="./examples"
temp_output="$(mktemp)"
last_output="./test-all-output.txt"

for dataset in $data_folder/*.dat; do
  echo "Processing $dataset"
  for tst in "${tests[@]}"; do
    for extra_param in "${extra_params[@]}"; do
      echo "./main.py $tst $dataset --debug --sum-only $extra_param"
      ./main.py $tst "$dataset" --debug --sum-only $extra_param "$@"
      code=$?
    done
  done
  echo
done | tee "$temp_output"
``
echo "The output is saved in $temp_output" >&2

if [ -f "$last_output" ]; then
  diff_file=$(mktemp)
  diff -u "$last_output" "$temp_output" > "$diff_file" || true
  if [ -s "$diff_file" ]; then
    echo "The output differs from $last_output:"
    diff -u --color=auto "$last_output" "$temp_output" || true
    echo "diff is saved in $diff_file"
    echo "If the new output is correct, run:"
    echo "cp '$temp_output' '$last_output'"
    exit 1
  else
    echo "The output is the same as in $last_output"
    exit 0
  fi
fi >&2  # Redirect the output to stderr
