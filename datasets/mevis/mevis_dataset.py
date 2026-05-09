###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
"""
MeViS data loader
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
import datasets.transform_video as T

import os
from PIL import Image
import json
import numpy as np
import random
import torchvision.transforms.functional as F

from pycocotools import mask as coco_mask
from misc import nested_tensor_from_videos_list


### --- ADDED: Copied from refer_youtube_vos_dataset.py --- ###
class MeViSTransforms:
    """
    Data augmentation transforms for MeViS dataset.
    Uses stronger augmentation strategy from official implementation to prevent overfitting.
    """
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 random_color, train_short_size, train_max_size, eval_short_size, eval_max_size, **kwargs):
        self.subset_type = subset_type
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if subset_type == 'train':
            # Official augmentation strategy with multiple scales and random crop
            scales = [288, 320, 352, 384, 416, 448, 480, 512]
            
            # Build transform pipeline matching official implementation
            transforms = []
            
            # Add horizontal flip
            if horizontal_flip_augmentations:
                transforms.append(T.RandomHorizontalFlip())
            
            # Add color augmentation
            if random_color:
                transforms.append(T.PhotometricDistort())
            
            # Add RandomSelect for diverse scale augmentation
            if resize_and_crop_augmentations:
                transforms.append(
                    T.RandomSelect(
                        # Branch 1: Simple multi-scale resize (50% probability)
                        T.Compose([
                            T.RandomResize(scales, max_size=train_max_size),
                            T.Check(),
                        ]),
                        # Branch 2: Random crop then resize (50% probability) - STRONGER augmentation
                        T.Compose([
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=train_max_size),
                            T.Check(),
                        ])
                    )
                )
            
            transforms.extend([T.ToTensor(), normalize])
            self.transforms = T.Compose(transforms)
            
        else:  # 'valid' or 'valid_u'
            transforms = []
            if resize_and_crop_augmentations:
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size))
            transforms.extend([T.ToTensor(), normalize])
            self.transforms = T.Compose(transforms)

    def __call__(self, source_frames, target, text_query):
        """
        Apply transforms to video frames and single unified target.
        
        Args:
            source_frames: List of PIL Images [T]
            target: Single unified target dict with keys like {'labels': [T], 'boxes': [T, 4], ...}
            text_query: Text query string
            
        Returns:
            source_frames: List of transformed PIL Images or tensors [T]
            target: Transformed target dict
            text_query: Possibly modified text query (e.g., left/right swapped after horizontal flip)
        """
        # transform_video expects (clip, target) where:
        # - clip: list of PIL Images [T]
        # - target: single dict with tensors
        
        # Apply transforms directly (transform_video handles clip + single target)
        source_frames, target = self.transforms(source_frames, target)
        
        # Stack frames if they're tensors
        if isinstance(source_frames[0], torch.Tensor):
            source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        
        if target and 'caption' in target:
            text_query = target['caption']
        
        return source_frames, target, text_query



### --- END ADDED --- ###


class Collator:
    """
    Collates samples with unified target format.
    Converts batch-major targets to time-major format expected by trainer.
    """
    def __init__(self, subset_type):
        self.subset_type = subset_type

    def __call__(self, batch):
        """
        Collate batch of samples with unified targets into trainer-expected format.
        
        Input:
            batch = [
                (imgs1, target1, exp1),  # Train
                (imgs2, target2, exp2),
                ...
            ]
            OR
            batch = [
                (imgs1, meta1, target1, exp1),  # Valid
                (imgs2, meta2, target2, exp2),
                ...
            ]
            
            where each target is a unified dict:
            {
                'labels': [T],
                'boxes': [T, 4],
                'masks': [T, H, W],
                ...
            }
        
        Output:
            {
                'samples': [B, T, C, H, W],
                'targets': [(dict_b1, dict_b2, ...), ...],  # List of T tuples
                'text_queries': [exp1, exp2, ...]
            }
        """
        if self.subset_type == 'train':
            # batch = [(imgs, target, exp), ...]
            samples = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            text_queries = [item[2] for item in batch]
        else:  # valid_u or valid
            # batch = [(imgs, meta, target, exp), ...]
            samples = [item[0] for item in batch]
            metas = [item[1] for item in batch]
            targets = [item[2] for item in batch]
            text_queries = [item[3] for item in batch]

        # Create NestedTensor from video list (required by model)
        samples = nested_tensor_from_videos_list(samples)  # NestedTensor with .tensors and .mask
        
        # Convert unified targets from batch-major to time-major
        # Input: list of B dicts, each with {'labels': [T], 'boxes': [T, 4], ...}
        # Output: list of T tuples, each with B dicts
        
        B = len(targets)
        # Use 'frames_idx' to get T since it exists in all target types (train, valid_u, valid)
        T = targets[0]['frames_idx'].shape[0] if 'frames_idx' in targets[0] else 1
        
        time_major_targets = []
        for t in range(T):
            frame_targets = []
            for b in range(B):
                frame_dict = {}
                
                for key, value in targets[b].items():
                    if key in ['caption', 'iscrowd', 'referred_instance_idx']:
                        # These don't have time dimension or are scalars
                        frame_dict[key] = value
                    elif key in ['orig_size', 'size']:
                        # These are [2] or [H, W] - no time dimension
                        frame_dict[key] = value
                    elif key in ['frames_idx']:
                        # [T] - extract t-th element as scalar
                        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] > t:
                            frame_dict[key] = value[t].unsqueeze(0)  # [1]
                        else:
                            frame_dict[key] = value.unsqueeze(0) if value.dim() == 0 else value
                    elif key in ['labels', 'valid', 'is_ref_inst_visible']:
                        # [T] - extract t-th element, keep as scalar for matcher
                        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] > t:
                            frame_dict[key] = value[t]  # scalar or [1] depending on original
                        else:
                            frame_dict[key] = value
                    elif key in ['boxes']:
                        # [T, 4] - extract t-th row
                        if isinstance(value, torch.Tensor) and value.dim() > 1 and value.shape[0] > t:
                            frame_dict[key] = value[t].unsqueeze(0)  # [1, 4]
                        else:
                            frame_dict[key] = value
                    elif key in ['masks', 'original_mask']:
                        # [T, H, W] - extract t-th slice
                        if isinstance(value, torch.Tensor) and value.dim() > 2 and value.shape[0] > t:
                            frame_dict[key] = value[t].unsqueeze(0)  # [1, H, W]
                        else:
                            frame_dict[key] = value
                    else:
                        # Unknown key, copy as is
                        frame_dict[key] = value
                
                frame_targets.append(frame_dict)
            
            time_major_targets.append(tuple(frame_targets))
        
        result = {
            'samples': samples,
            'targets': time_major_targets,  # List of T tuples
            'text_queries': text_queries
        }
        
        if self.subset_type != 'train':
            result['videos_metadata'] = metas  # Trainer expects 'videos_metadata'
        
        return result


class MeViSDataset(Dataset):
    """
    A dataset class for the MeViS dataset
    """

    def __init__(self, subset_type='train', dataset_path='./rvosdata/mevis', num_frames=5,
                 sampling_step=5,
                 **kwargs):
        if subset_type == 'test':
            subset_type = 'valid_u'
        self.subset_type = subset_type

        self.img_folder = os.path.join(dataset_path, subset_type)
        self.ann_file = os.path.join(dataset_path, subset_type, 'meta_expressions.json')

        self.train_num_frames = num_frames
        self.eval_num_frames = 5
        self.sampling_step = sampling_step
        self.prepare_metas()

        # **kwargs will pass 'horizontal_flip_augmentations', 'train_short_size', etc. from config
        self._transforms = MeViSTransforms(subset_type, **kwargs)

        self.collator = Collator(self.subset_type)

        if subset_type in ['train', 'valid_u']:
            self.mask_dict = json.load(open(os.path.join(dataset_path, subset_type, 'mask_dict.json')))
        else:
            self.mask_dict = None

        print('video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        # (This function remains unchanged)
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())
        self.metas = []
        for vid in self.videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():

                if self.subset_type == 'train':
                    # Logic: 'train' uses sparse sampling
                    for frame_id in range(0, vid_len, self.sampling_step):
                        meta = {}
                        meta['video_id'] = vid
                        meta['exp'] = exp_dict['exp']
                        meta['exp_id'] = exp_id
                        meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                        meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                        meta['frames'] = vid_frames
                        meta['frame_id'] = frame_id  # Starting frame of the clip
                        meta['category'] = 0
                        self.metas.append(meta)
                else:  # 'valid_u' or 'valid'
                    # Logic: 'valid_u' and 'valid' both use overlapping dense windows
                    stride = self.eval_num_frames // 2  # Window overlap by half
                    if stride == 0:
                        stride = 1
                    for frame_id in range(0, vid_len, stride):
                        meta = {}
                        meta['video_id'] = vid
                        meta['exp'] = exp_dict['exp']
                        meta['exp_id'] = exp_id
                        # 'valid_u' has obj_id, 'valid' does not
                        if self.subset_type == 'valid_u':
                            meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                            meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                        else:
                            meta['obj_id'] = []
                            meta['anno_id'] = []

                        meta['frames'] = vid_frames
                        meta['frame_id'] = frame_id  # This is the starting frame of the window
                        meta['category'] = 0
                        meta['is_chunk'] = True  # Add a flag
                        self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]
            video, exp, anno_id, category, frames, frame_id = \
                meta['video_id'], meta['exp'], meta['anno_id'], meta['category'], meta['frames'], meta['frame_id']
            exp = " ".join(exp.lower().split())
            vid_len = len(frames)

            meta_is_chunk = meta.get('is_chunk', False)

            if not meta_is_chunk:  # 'train' or 'valid_u'
                # Original logic: perform random clip sampling for 'train' and 'valid_u'
                num_frames = self.train_num_frames
                sample_indx = [frame_id]
                if self.train_num_frames != 1:
                    sample_id_before = random.randint(1, 3)
                    sample_id_after = random.randint(1, 3)
                    local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                    sample_indx.extend(local_indx)
                    if num_frames > 3:
                        all_inds = list(range(vid_len))
                        global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                        global_n = num_frames - len(sample_indx)
                        if len(global_inds) > global_n:
                            select_id = random.sample(range(len(global_inds)), global_n)
                            for s_id in select_id: sample_indx.append(global_inds[s_id])
                        elif vid_len >= global_n:
                            select_id = random.sample(range(vid_len), global_n)
                            for s_id in select_id: sample_indx.append(all_inds[s_id])
                        else:
                            select_id = np.random.choice(range(vid_len), global_n - vid_len).tolist() + list(
                                range(vid_len))
                            for s_id in select_id: sample_indx.append(all_inds[s_id])
                sample_indx.sort()
                if self.subset_type == "train" and np.random.rand() < 0.3:
                    sample_indx = sample_indx[::-1]

            else:  # self.subset_type == 'valid'
                # New logic: load a dense chunk for 'valid'
                num_frames = self.eval_num_frames
                start_frame = frame_id
                sample_indx = list(range(start_frame, min(start_frame + num_frames, vid_len)))

                # Ensure chunk is always full (pad with last frame)
                num_missing = num_frames - len(sample_indx)
                if num_missing > 0:
                    sample_indx.extend([sample_indx[-1]] * num_missing)

            # read frames and masks
            imgs = []  # List of PIL Images

            labels_list, boxes_list, masks_list, valid_list, frames_idx_list = [], [], [], [], []
            # Store original_mask separately (not in official but needed for your model)
            original_masks_list = []

            frame_has_valid_instance = False

            for j in range(num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                img = Image.open(img_path).convert('RGB')
                imgs.append(img)
                w, h = img.size

                if self.subset_type in ['train', 'valid_u']:
                    # Logic: 'train' and 'valid_u' need to load ground truth masks and boxes

                    mask = np.zeros(img.size[::-1], dtype=np.float32)
                    if self.mask_dict is not None:
                        for x in anno_id:
                            frm_anno = self.mask_dict[x][frame_indx]
                            if frm_anno is not None:
                                mask += coco_mask.decode(frm_anno)

                    label = torch.tensor(category)
                    is_frame_valid = False

                    if (mask > 0).any():
                        y1, y2, x1, x2 = self.bounding_box(mask)
                        box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                        is_frame_valid = True
                        frame_has_valid_instance = True  # Mark that clip has at least one valid frame
                    else:
                        box = torch.tensor([0, 0, 0, 0]).to(torch.float)

                    mask = torch.from_numpy(mask)

                    # Collect data for unified target
                    frames_idx_list.append(sample_indx[j])
                    labels_list.append(label)
                    boxes_list.append(box)
                    masks_list.append(mask)
                    original_masks_list.append(mask.clone())
                    valid_list.append(1 if is_frame_valid else 0)

                else:  # self.subset_type == 'valid'
                    # Logic: 'valid' split (no ground truth)
                    # For valid split, we still need to maintain structure
                    frames_idx_list.append(sample_indx[j])
                    # Add dummy data to maintain structure
                    labels_list.append(torch.tensor(0))
                    boxes_list.append(torch.tensor([0, 0, 0, 0]).float())
                    masks_list.append(torch.zeros(h, w))
                    original_masks_list.append(torch.zeros(h, w))
                    valid_list.append(0)

            if self.subset_type in ['train', 'valid_u']:
                # Build unified target with stacked tensors
                target = {
                    'frames_idx': torch.tensor(frames_idx_list),  # [T]
                    'labels': torch.stack(labels_list),  # [T]
                    'boxes': torch.stack(boxes_list),  # [T, 4]
                    'masks': torch.stack(masks_list),  # [T, H, W]
                    'original_mask': torch.stack(original_masks_list),  # [T, H, W]
                    'valid': torch.tensor(valid_list),  # [T]
                    'is_ref_inst_visible': torch.tensor(valid_list),  # [T]
                    'referred_instance_idx': torch.tensor(0),  # scalar
                    'caption': exp,  # Required by transform_video.RandomHorizontalFlip
                    'orig_size': torch.tensor([h, w]),  # [2]
                    'size': torch.tensor([h, w]),  # [2]
                    'iscrowd': torch.zeros(1)  # [1]
                }
            else:  # 'valid'
                # Minimal target for validation
                target = {
                    'frames_idx': torch.tensor(frames_idx_list),  # [T]
                    'orig_size': torch.tensor([h, w]),  # [2]
                    'size': torch.tensor([h, w]),  # [2]
                    'caption': exp  # Still need for transforms
                }

            # _transforms expects: List[PIL Images], target dict, str (text query)
            # Returns: stacked tensor [T, C, H, W], transformed target dict, possibly modified text
            imgs, target, exp = self._transforms(imgs, target, exp)

            # check if the clip has at least one valid instance
            if not meta_is_chunk:  # train or valid_u
                if frame_has_valid_instance:
                    instance_check = True
            else:  # valid
                instance_check = True  # always accept for validation

            if not instance_check:
                idx = random.randint(0, self.__len__() - 1)

        if self.subset_type == 'train':
            return imgs, target, exp
        else:  # 'valid_u' or 'valid'
            # For validation, add required metadata fields for postprocessor
            meta['resized_frame_size'] = imgs.shape[-2:]  # [H, W] after transforms
            meta['original_frame_size'] = target['orig_size'].cpu().numpy().tolist()  # [H, W] original
            meta['frame_indices'] = [frames[i] for i in sample_indx]  # frame names
            return imgs, meta, target, exp
