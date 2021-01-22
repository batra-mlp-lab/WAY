#!/bin/bash
set -e

BASEDIR="/path/to/WAY/"

python -W ignore ${BASEDIR}lingUnet/run_lingunet.py \
    --evaluate \
    --sample_used 1 \
    --name base_test \
    --data_base_dir ${BASEDIR}data/ \
    --eval_ckpt ${BASEDIR}data/base.pt \
    --generate_predictions \
    --batch_size 32 \
    --distance_metric euclidean \
    --predictions_dir ${BASEDIR}data/logs/ \
