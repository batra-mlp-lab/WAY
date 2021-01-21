#!/bin/bash
set -e

BASEDIR="/nethome/mhahn30/Repositories/LED/Modeling/lingUnet/"
SAVEDIR="/srv/share/mhahn30/LED/model_runs/logs/"
DATADIR="/nethome/mhahn30/Repositories/LED/Modeling/data/"

python -W ignore ${BASEDIR}run_lingunet.py \
    --evaluate \
    --sample_used 1 \
    --name base_test \
    --data_base_dir ${DATADIR} \
    --eval_ckpt ${SAVEDIR}base/base_acc0.60_unseenAcc0.45_epoch31.pt \
    --generate_predictions \
    --batch_size 32 \
    --distance_metric euclidean \
    --predictions_dir ${SAVEDIR}base_lingunet/ \
