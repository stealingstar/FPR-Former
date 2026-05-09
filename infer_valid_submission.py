import argparse
import os
import shutil
from os import path
import random
import sys

import numpy as np
import ruamel.yaml as YAML
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import misc as utils
from datasets import build_dataset
from models import build_model
from models.video_swin_transformer import compute_mask


def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        sample = sample.to(device)
    elif isinstance(sample, tuple) or isinstance(sample, list):
        sample = [to_device(s, device) for s in sample]
    elif isinstance(sample, dict):
        sample = {k: to_device(v, device) for k, v in sample.items()}
    return sample


def init_process_group_and_set_device(world_size, process_id, device_id, config):
    config.world_size = world_size
    config.rank = process_id
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    config.device = device
    if world_size > 1:
        config.distributed = True
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        utils.setup_for_distributed(config.rank == 0)
    else:
        config.distributed = False
    return device

@torch.no_grad()
def run_evaluation(config, model, postprocessor, data_loader, device):
    model.eval()
    is_main_process = config.rank == 0


    output_dir_path = config.output_dir
    if not output_dir_path:
        output_dir_path = "./submission_output"

    validation_output_dir = path.join(output_dir_path)
    epoch_validation_output_dir = path.join(validation_output_dir, f'mevis_submission_final')
    annotations_dir = path.join(epoch_validation_output_dir, 'Annotations')

    if is_main_process:
        if os.path.exists(epoch_validation_output_dir):
            print(f"Warning: Removing existing submission directory: {epoch_validation_output_dir}")
            shutil.rmtree(epoch_validation_output_dir)
        os.makedirs(annotations_dir, exist_ok=True)

    if config.distributed:
        dist.barrier()

    print(f"Process {config.rank}: Generating submission file for MeViS valid set...")


    for batch_dict in tqdm(data_loader, disable=not is_main_process):
        samples = batch_dict['samples'].to(device)
        targets = to_device(batch_dict['targets'], device)
        valid_indices = None
        text_queries = batch_dict['text_queries']

        outputs = model(samples, valid_indices, text_queries, targets)
        videos_metadata = batch_dict['videos_metadata']
        sample_shape_with_padding = samples.tensors.shape[-2:]
        preds_by_video = postprocessor(outputs, videos_metadata, sample_shape_with_padding)


        for p in preds_by_video:
            pred_dir_path = path.join(annotations_dir, p['video_id'], p['exp_id'])
            os.makedirs(pred_dir_path, exist_ok=True)
            for f_mask, f_idx in zip(p['pred_masks'], p['frame_indices']):
                pred_mask_path = path.join(pred_dir_path, f'{f_idx}.png')
                pred_mask = Image.fromarray((255 * f_mask.squeeze()).numpy())
                pred_mask.save(pred_mask_path)


    if config.distributed:
        print(f"Process {config.rank}: Finished predictions. Waiting for other processes...")
        dist.barrier()


    if is_main_process:
        print('Saving MeViS predictions complete.')
        print('Creating MeViS submission zip file...')
        zip_file_path = path.join(validation_output_dir, os.path.basename(config.checkpoint_path))

        shutil.make_archive(zip_file_path, 'zip', root_dir=annotations_dir)
        print(f'MeViS zip file created at: {zip_file_path}.zip')

        shutil.rmtree(epoch_validation_output_dir)

    if config.distributed:
        dist.barrier()

def main(process_id, config):
    device_id = config.device_ids[process_id]
    device = init_process_group_and_set_device(config.num_devices, process_id, config.device_ids[process_id], config)
    is_main_process = config.rank == 0


    model, _, postprocessor = build_model(config)
    model.to(device)
    model_without_ddp = model

    if config.distributed:
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
        model_without_ddp = model.module


    if config.checkpoint_path:
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        state_dict = checkpoint["model_state_dict"]
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0 and is_main_process:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0 and is_main_process:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if is_main_process:
            print(f"Checkpoint loaded from {config.checkpoint_path}")
    else:
        if is_main_process:
            print("ERROR: --checkpoint_path is required for evaluation.")
        return


    dataset_val_submission = build_dataset(image_set='valid', dataset_file='mevis', **vars(config))
    if config.distributed:
        sampler_val_submission = DistributedSampler(dataset_val_submission, num_replicas=config.world_size,
                                                    rank=config.rank, shuffle=False)
    else:
        sampler_val_submission = None

    data_loader_val_submission = DataLoader(dataset_val_submission, config.eval_batch_size,
                                            sampler=sampler_val_submission, drop_last=False,
                                            collate_fn=dataset_val_submission.collator,
                                            num_workers=24,
                                            pin_memory=True)


    run_evaluation(config, model, postprocessor, data_loader_val_submission, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MeViS VALID Submission Script')


    parser.add_argument('--config_path', '-c',
                        default='./configs/refer_youtube_vos.yaml',
                        help='path to configuration file')
    parser.add_argument('--checkpoint_path', '-ckpt', type=str, required=True,
                        help='The checkpoint path (.pth.tar file) to evaluate')
    parser.add_argument("--output_dir", type=str, default="./submission_output",
                        help="Path to save the final submission.zip")


    parser.add_argument("--device_ids", default=[0], type=int, nargs='+')
    parser.add_argument("--device", default="cuda")


    parser.add_argument('--running_mode', default='test')
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--backbone_pretrained_path", "-bpp", type=str)

    args = parser.parse_args()

    with open(args.config_path) as f:
        yaml = YAML.YAML(typ='rt')
        config = yaml.load(f)
    config = {k: v['value'] for k, v in config.items()}
    config = {**config, **vars(args)}
    config = argparse.Namespace(**config)


    config.num_devices = len(config.device_ids)
    print(f"Starting DDP evaluation with {config.num_devices} GPUs.")
    torch.multiprocessing.spawn(main, nprocs=config.num_devices, args=(config,))