#!/bin/bash
# Script: infer_refytb.sh
# Description: Run inference on Refer-YouTube-VOS dataset

python3 infer_refytb.py \
    -c configs/refer_youtube_vos.yaml \
    -rm test \
    --version "infer_ytb_tiny" \
    -ng 1 \
    --backbone "video-swin-t" \
    -bpp "pretrained/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth" \
    -ckpt "checkpoints/finetune_ytb_tiny.pth.tar"