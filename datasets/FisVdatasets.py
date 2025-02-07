from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
import pandas as pd


class FisVDataset(Dataset):
    def __init__(self, video_feat_path, label_path, action_list, clip_num=124, action_type='all',
                 score_type='TES', train=True, args=None):
        # if action_type == 'all' and not train:
        #     raise SystemError
        self.train = train
        self.video_path = video_feat_path
        self.erase_path = video_feat_path + '_erTrue'

        self.clip_num = clip_num
        if score_type == 'TES':
            self.score_range = 45
        else:
            self.score_range = 40
        args.score_range = self.score_range
        print("score_range:", args.score_range)
        self.labels = self.read_label(label_path, score_type, action_type)
        classes_all = pd.read_csv(action_list)
        self.action_list = classes_all['name'].values.tolist()

    @property
    def classes(self):
        return self.action_list

    def read_label(self, label_path, score_type, action_type):
        fr = open(label_path, 'r')
        idx = {'TES': 1, 'PCS': 2}
        labels = []
        score = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            if action_type == 'all' or action_type == line[0].split('_')[0]:
                s = float(line[idx[score_type]])
                labels.append([line[0], s])
                score.append(s)
        print("max:",max(score))
        print("min:",min(score))
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, self.labels[idx][0] + '.npy'))

        # temporal random crop or padding
        # print(video_feat.shape)
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
        # else:
        #     if len(video_feat) > self.clip_num:
        #         st = (len(video_feat) - self.clip_num) // 2
        #         video_feat = video_feat[st:st + self.clip_num]
        #         # erase_feat = erase_feat[st:st + self.clip_num]
        #     elif len(video_feat) < self.clip_num:
        #         new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
        #         new_feat[:video_feat.shape[0]] = video_feat
        #         video_feat = new_feat
        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range
