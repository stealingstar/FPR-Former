MASTER_ADDR=localhost MASTER_PORT=29500 python ./infer_valid_submission.py \
    -c ./configs/finetune_mevis.yaml \
    --backbone "video-swin-t" \
    -bpp "./pretrained/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth" \
    -ckpt "./checkpoints/finetune_mevis.pth.tar" \
    --output_dir ./runs/mevis/submission_output \
    --device_ids 0 1 2 3
