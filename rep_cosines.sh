#!/usr/bin/bash
set -e

source ~/miniforge3/bin/activate

python rep_cosines.py \
  --system_prompt \
  --savedir cache/rep_cosines \
  --dataset sni_sa \
  --evaldirs $STEERING_SCRATCH_DIR/exps/cb_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/sft_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/sft2cb_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/dpo_cls_diverse/reject_dsets_sni_s \

python rep_cosines.py \
  --system_prompt \
  --savedir cache/rep_cosines \
  --dataset sni_s \
  --evaldirs $STEERING_SCRATCH_DIR/exps/cb_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/sft_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/sft2cb_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/dpo_cls_diverse/reject_dsets_sni_s \

python rep_cosines.py \
  --system_prompt \
  --savedir cache/rep_cosines \
  --dataset sni_pe \
  --evaldirs $STEERING_SCRATCH_DIR/exps/cb_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/sft_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/sft2cb_cls_diverse/reject_dsets_sni_s $STEERING_SCRATCH_DIR/exps/dpo_cls_diverse/reject_dsets_sni_s \
