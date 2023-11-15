#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="48"

# read ensemble (skip header)
root_dir=$(cd .. && pwd)
ensemble_path=$root_dir/ensembles/ensemble_v30.tsv
mapfile -s 1 -t ensemble < "$ensemble_path"

# setup directories
models_dir="$root_dir/models"  # models/fold{1,2,3}/model_name/checkpoint-123/pytorch_model.bin
embeddings_dir="/path/to/track_embeddings"  # directory with .npy files of track embeddings
predictions_dir="$root_dir/predictions"  # predictions/fold{1,2,3}/model_name__step123.npz
data_path="$root_dir/splits/test.csv"  # path to .csv with tracks and, optionaly, labels
blended_output_path="$root_dir/predictions/y_pred.csv"  # blended ensemble output

# iterate over models
for x in "${ensemble[@]}"; do
  a=( ${x} )
  name="${a[0]}"
  # iterate over splits
  for i in 1 2 3; do
    step="${a[$i]}"
    echo "model: $name; fold: $i; step: $step"
    output_dir="$predictions_dir/fold$i"
    mkdir -p "$output_dir"
    python "$root_dir/jobs/predict.py" \
      ++checkpoint_path="$models_dir/fold$i/$name/checkpoint-$step/pytorch_model.bin" \
      ++output_path="$output_dir/${name}__step$step.npz" \
      ++data_path="$data_path" \
      ++embeddings_dir=$embeddings_dir \
      ++output_format=npz
    echo "============================="
  done
done

# blend predictions
python "$root_dir/jobs/blend.py" \
  --data_path="$data_path" \
  --config_path="$ensemble_path" \
  --predictions_dir="$predictions_dir" \
  --output_path="$blended_output_path"