#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="48"

root_dir=$(cd .. && pwd)
models_dir=$root_dir/models
embeddings_dir=/path/to/track_embeddings
output_dir=$root_dir/models_v2  # to prevent overriding parent configs

# iterate over models
for model_dir in "${models_dir}"/fold1/*; do
  name=$(basename "$model_dir")
  # iterate over splits
  for fold in 1 2 3; do
    echo "model: $name; fold: $fold"
    model_dir_i="$models_dir/fold$fold/$name"  # parent
    output_dir_i="$output_dir/fold$fold/$name"  # child
    python "$root_dir/jobs/retrain.py" \
      train_data_path="$root_dir/splits/train_fold$fold.csv" \
      valid_data_path="$root_dir/splits/valid_fold$fold.csv" \
      embeddings_dir=$embeddings_dir \
      hydra.run.dir="$output_dir_i" \
      training_args.output_dir="$output_dir_i" \
      ++model_dir="$model_dir_i"
    echo "============================"
  done
done