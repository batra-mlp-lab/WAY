#!/bin/bash
set -e

BASEDIR="/path/to/WAY/"

python -W ignore ${BASEDIR}lingUnet/run_lingunet.py \
    --train \
    --embedding_type glove \
    --num_lingunet_layers 3 \
    --num_rnn_layers 1 \
    --bidirectional \
    --sample_used 1 \
    --num_epoch 60 \
    --save \
    --log \
    --log_dir ${BASEDIR}data/logs/ \
    --data_base_dir ${BASEDIR}data/ \
    --summary_dir ${BASEDIR}logs/tensorboard/ \
    --visualization_dir ${BASEDIR}lingUnet/vis/ \
    --visualize \
    --data_aug \
    --res_connect \
    --distance_metric euclidean \
    --name base \
