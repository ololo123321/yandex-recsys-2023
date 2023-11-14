#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="48"

# read ensemble (skip header)
ensemble_path=/home/vitaly/PycharmProjects/yars/ensemble_best.tsv
mapfile -s 1 -t ensemble < $ensemble_path

# setup directories
models_dir="$(cd .. && pwd)/models"  # models/fold{1,2,3}/model_name/checkpoint-123/pytorch_model.bin
predictions_dir="$(cd .. && pwd)/predictions"  # predictions/fold{1,2,3}/model_name__step123.npz
data_path="/path/to/test.csv"  # path to .csv with tracks and, optionaly, labels
embeddings_dir="/path/to/track_embeddings"  # directory with .npy files of track embeddings
blended_output_path="/path/to/y_pred.csv"

# iterate over models
for x in "${ensemble[@]}"; do
  a=( ${x} )
  name="${a[0]}"
  echo "$name"
  # iterate over splits
  for i in 1 2 3; do
    step="${a[$i]}"
    output_dir="${predictions_dir}/fold${i}"
    mkdir -p ${output_dir}
    echo "fold: ${i}; step: ${step}"
    python "$(cd .. && pwd)/jobs/predict.py" \
      ++checkpoint_path="${models_dir}/fold${i}/${name}/checkpoint-${step}/pytorch_model.bin" \
      ++output_path="${output_dir}/${name}__step${step}.npz" \
      ++data_path=$data_path \
      ++embeddings_dir=$embeddings_dir \
      ++output_format=npz
    echo "============================="
  done
done

# blend predictions
python "$(cd .. && pwd)/jobs/blend.py" \
  --data_path=${data_path} \
  --config_path=${ensemble_path} \
  --predictions_dir=${predictions_dir} \
  --output_path=${blended_output_path}