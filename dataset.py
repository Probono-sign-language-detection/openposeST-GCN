import json
import math
import os
import random
from glob import glob

import numpy as np

import cv2
import torch
import torch.nn as nn

from torch.utils.data import Dataset

def compute_difference(x):
    diff = []

    for i, xx in enumerate(x):
        temp = []
        for j, xxx in enumerate(x):
            if i != j:
                temp.append(xx - xxx)

        diff.append(temp)

    return diff

def read_pose_frames(file_path):
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    # try:
    content = json.load(open(file_path))["people"][0]
    # except :
    #     return "Frame을 찾을 수 없습니다"
    
    body_pose = content["pose_keypoints_2d"]
    left_hand_pose = content["hand_left_keypoints_2d"]
    right_hand_pose = content["hand_right_keypoints_2d"]

    body_pose.extend(left_hand_pose)
    body_pose.extend(right_hand_pose)

    x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
    y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]

    x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
    y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)

    x_diff = torch.FloatTensor(compute_difference(x)) / 2
    y_diff = torch.FloatTensor(compute_difference(y)) / 2

    zero_indices = (x_diff == 0).nonzero()

    orient = y_diff / x_diff
    orient[zero_indices] = 0

    xy = torch.stack([x, y]).transpose_(0, 1)

    ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

    xy = ft[:, :2]
    return xy

class ProbonoSignDataset(Dataset):
    def __init__(
        self, 
        pose_folder=r"C:\Users\JeongSeongYun\Desktop\openposetest\pyopen\output_json"

    ):
        self.pose_folder = pose_folder

    def __getitem__(self, idx):

        poses_across_time = self._load_poses()
        
        return poses_across_time


    def __len__(self):
        return len(glob(self.pose_folder+r"\*.json"))


    def _load_poses(self):
        """들어온 Frame 폴더에 접근하여
        read_pose_frames 함수를 이용해 pose정보를 추출합니다.
        추출한 pose정보를 torch.cat을 통해 Sequential하게 쌓아줍니다."""
        poses = []
        pose_folder = glob(self.pose_folder+r"\*.json")
        for pose_json in pose_folder:
            pose = read_pose_frames(pose_json)
            poses.append(pose)

        """
        여기가 좀 중요할 수 있습니다.
        pretrained된 모델은 한 video에서 100장의 frame sample로 추론을 하는데, 
        100장이 없는 경우가 있ㅇ을 것입니다. 그런 경우엔 padding을 통해 요구되는 num_samples수만큼 채워주빈다.
        나올 수 있는 질문:
            Q. 그럼 그냥 그 parameter를 조절하면 되는 거 아닌가요?
            A. 아쉽게도, 저희는 pretrinaed된 모델을 사용할 예정이기에 shape을 맞춰줄 필요가 있었습니다.
                때문에 그냥 pad를 통해 부족한 수를 채워줄 예정입니다.
        """
        # pad = None

        # if len(poses) < num_samples:
        #     num_padding = num_samples - len(frames_to_sample)
        #     last_pose = poses[-1]
        #     pad = last_pose.repeat(1, num_padding)

        # poses_across_time = torch.cat(poses, dim=1)
        # if pad is not None:
        #     poses_across_time = torch.cat([poses_across_time, pad], dim=1)

        # return poses_across_time
        pad =  None

        if len(poses) < 50: # 여기를 수정하면 될 거 같다. 지금은 50을 해야 맞는데.. 쩝.. 흠.. 여길 여차하면 그냥 모델 처음에 nn.linear로 떼우는 방법도 있긴 하다
            num_padding = 50 - len(glob(self.pose_folder+r"\*.json"))
            last_pose = poses[-1]
            pad = last_pose.repeat(1, num_padding)

        poses_across_time = torch.cat(poses, dim=1)
        if pad is not None:
            poses_across_time = torch.cat([poses_across_time, pad], dim=1)

        return poses_across_time