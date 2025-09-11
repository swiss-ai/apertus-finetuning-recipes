#!/bin/bash
ACCEL_PROCS=$(( $SLURM_NNODES * $SLURM_GPUS_PER_NODE ))

MAIN_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MAIN_PORT=12802

accelerate launch --config_file configs/zero3_multinode.yaml \
           --num_machines=$SLURM_NNODES --num_processes=$ACCEL_PROCS \
           --machine_rank $SLURM_PROCID \
           --main_process_ip $MAIN_ADDR --main_process_port $MAIN_PORT \
  sft_train.py \
  --config configs/sft_full.yaml
