#!/bin/bash
# Script: demo_video.sh
# Description: Run demo inference on a video file

python3 demo_video.py \
    -c configs/a2d_sentences.yaml \
    -rm test \
    --backbone "video-swin-b" \
    -bpp "pretrained/pretrained_swin_transformer/swin_base_patch244_window877_kinetics400_22k.pth" \
    -ckpt "checkpoints/finetune_ytb_base.pth.tar" \
    --video_dir "visualize/infer_test.mp4"