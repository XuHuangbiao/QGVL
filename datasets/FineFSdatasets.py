from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
import pandas as pd
import json


class FineFSDataset(Dataset):
    def __init__(self, video_feat_path, label_path, action_list, clip_num=26, action_type='Ball', score_type=None, train=True, args=None, seed=42):
        self.train = train
        self.video_path = video_feat_path
        self.erase_path = video_feat_path + '_erTrue'
        if score_type == "SP":
            score_idx = {'TES': 70, 'PCS': 50}
        else:
            score_idx = {'TES': 130, 'PCS': 100}
        self.score_range = score_idx[action_type]
        args.score_range = self.score_range
        print("score_range:", args.score_range)
        self.clip_num = clip_num
        print(score_type)
        if score_type == "SP":
            ranges = np.arange(0, 729)
            np.random.seed(seed)
            np.random.shuffle(ranges)
            if self.train:
                self.ran = ranges[:583]
            else:
                self.ran = ranges[583:]
        else:
            ranges = np.arange(729, 1167)
            np.random.seed(seed)
            np.random.shuffle(ranges)
            if self.train:
                self.ran = ranges[:350]
            else:
                self.ran = ranges[350:]
        self.labels = self.read_label(label_path, action_type, score_type)
        classes_all = pd.read_csv(action_list)
        self.action_list = classes_all['name'].values.tolist()

    @property
    def classes(self):
        return self.action_list

    def read_label(self, label_path, action_type, score_type):

        idx = {'TES': "total_element_score", 'PCS': "total_program_component_score(factored)"}
        labels = []
        score = []
        for i in self.ran:
            with open(os.path.join(label_path, str(i) + '.json')) as f:
                label = json.load(f)
            labels.append([str(i), float(label[idx[action_type]])])
            score.append(float(label[idx[action_type]]))
        print("max:", max(score))
        print("min:", min(score))
        return labels

    def __getitem__(self, idx):
        name = self.labels[idx][0]
        video_feat = torch.load(os.path.join(self.video_path, self.labels[idx][0] + '.pkl'))

        # temporal random crop or padding
        # video_feat = video_feat.mean(1)
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat

        # video_feat = torch.from_numpy(video_feat).float()
        return video_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range
