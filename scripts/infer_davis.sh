#!/bin/bash
# Script: infer_davis.sh
# Description: Run inference on DAVIS dataset and evaluate results

python3 infer_davis.py \
    -c configs/davis.yaml \
    -rm test \
    --version "davis_tiny_finetune_ytb" \
    -ng 1 \
    --backbone "video-swin-t" \
    -bpp "pretrained/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth" \
    -ckpt "checkpoints/finetune_ytb_tiny.pth.tar"


# After inference, evaluate the results using the DAVIS evaluation toolkit:
# python3 eval_davis.py --results_path "results/davis/davis_tiny_finetune_ytb/anno_0"
# python3 eval_davis.py --results_path "results/davis/davis_tiny_finetune_ytb/anno_1"
# python3 eval_davis.py --results_path "results/davis/davis_tiny_finetune_ytb/anno_2"
# python3 eval_davis.py --results_path "results/davis/davis_tiny_finetune_ytb/anno_3"