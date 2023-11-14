#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="48"

models_dir=$(cd .. && pwd)/models
embeddings_dir=/path/to/track_embeddings
output_dir=$(cd .. && pwd)/models_v2  # to prevent overriding parent configs
python_path=/home/vitaly/anaconda3/bin/python
script_path=$(cd .. && pwd)/jobs/retrain.py

for model_dir in "${models_dir}"/fold1/*; do
  name=$(basename "$model_dir")
  for fold in 1 2 3; do
    echo "model: $name; fold: $fold"
    output_dir_i="${output_dir}/fold$fold/${name}"
    ${python_path} "${script_path}" \
      train_data_path="/$(cd .. && pwd)/splits/train_fold$fold.csv" \
      valid_data_path="/$(cd .. && pwd)/splits/valid_fold$fold.csv" \
      embeddings_dir=$embeddings_dir \
      hydra.run.dir="${output_dir_i}" \
      training_args.output_dir="${output_dir_i}" \
      ++model_dir="${model_dir}"
    echo "============================"
  done
done