#!/bin/bash
set -e

BASEDIR="/nethome/mhahn30/Repositories/LED/Modeling/lingUnet/"
SAVEDIR="/srv/share/mhahn30/LED/model_runs/"
DATADIR="/nethome/mhahn30/Repositories/LED/Modeling/data/"

python -W ignore ${BASEDIR}run_lingunet.py \
    --train \
    --embedding_type glove \
    --num_lingunet_layers 3 \
    --num_rnn_layers 1 \
    --bidirectional \
    --sample_used 1 \
    --num_epoch 60 \
    --save \
    --log \
    --log_dir ${SAVEDIR}logs/ \
    --data_base_dir ${DATADIR} \
    --summary_dir ${SAVEDIR}tensorboard/ \
    --visualization_dir ${BASEDIR}vis/ \
    --visualize \
    --data_aug \
    --res_connect \
    --distance_metric euclidean \
    --name base \
