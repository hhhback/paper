#!/usr/bin/env bash
set -e

data_path="$1"
output_base="$2"
train_start="$3"
train_end="$4"
model_name="$5"

if [ -z "$data_path" ] || [ -z "$output_base" ] || [ -z "$train_start" ] || [ -z "$train_end" ] || [ -z "$model_name" ]; then
    echo "Usage: $0 <data_path> <output_base> <train_start> <train_end> <model_name>"
    exit 1
fi

echo "Starting implicit extraction experiment"
echo "  data_path:   $data_path"
echo "  output_base: $output_base"
echo "  train_start: $train_start"
echo "  train_end:   $train_end"
echo "  model_name:  $model_name"

current_date="$train_start"
end_date="$train_end"

while [ "$current_date" != "$(date -d "$end_date + 1 day" +%Y-%m-%d)" ]; do
    echo "Processing date: $current_date"

    python async_run.py "$current_date" "$data_path" 1 \
        --output-path "${output_base}/implicit_data/${current_date}" \
        --datestr "$current_date"

    current_date="$(date -d "$current_date + 1 day" +%Y-%m-%d)"
done

echo "Implicit extraction experiment complete"
