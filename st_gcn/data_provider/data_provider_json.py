import json
import random
import os
import ast
import logging
import torch
import tqdm
import numpy as np

class DatasetJSON(torch.utils.data.Dataset):
    def __init__(self, path, split_file, partition, actions, temporal_win, num_keypoints, input_channels, sample_interval=1, random_sample=False, dataset_cfg=0, uniform_sample=False, slack_len=0):
        self.path = path
        self.split_file = split_file
        self.partition = partition
        self.dataset_cfg = dataset_cfg
        self.sample_interval = sample_interval
        self.temporal_win = temporal_win
        self.input_channels = input_channels
        self.num_keypoints = num_keypoints
        self.random_sample = random_sample
        self.uniform_sample = uniform_sample
        self.slack_len = slack_len
        self.active_classes = actions
        print(f"active_classes: {self.active_classes}")
        
        self.data = []
        self.labels = [] # id only
        self.file_path = []
        self.timeranges = []
        self.sample_intervals = []

        print(f"split_file: {split_file}")
        assert os.path.exists(split_file)
        with open(split_file) as json_file: 
            split_data = json.load(json_file)
            if partition == "val":
                if "val" in split_data:
                    file_list = split_data["val"] # other dataset, like DHG uses 'val' keyword
                else:
                    file_list = split_data["validate"] # hand emoji dataset uses 'validate' keyword
            elif partition == "train" or partition == "test":
                file_list = split_data[partition]
            else:
                print("split file does not contain either of the partitions: train, test, val, or validate! exit.")
                logging.info("split file does not contain either of the partitions: train, test, val, or validate! exit.")
                exit()

            print("Split partition ", partition, " files count: ", len(file_list))
            logging.info('Split partition %s, files count: %d' %(partition, len(file_list)))

            # iterate over each JSON keypoints file from the given partition list
            for file_bundle in tqdm.tqdm(file_list):
                file_name = file_bundle[0]
                label_name = file_bundle[2]

                if label_name not in self.active_classes: 
                    # treat inactive classes all as background class
                    print(f"label_name: {label_name}")
                    label_name = "Background"
                    label_id = self.active_classes.index(label_name)
                else: # reassign label id, use active class index
                    label_id = self.active_classes.index(label_name)
                
                file_name = os.path.basename(file_name)

                keypoint_file_path = path + "/" + file_name

                if not os.path.exists(keypoint_file_path):
                    print("keypoint file missing after fixing extensions %s" %keypoint_file_path)
                    logging.info("keypoint file missing %s" %keypoint_file_path)
                    continue

                keypoints = self.load_json(keypoint_file_path)
                
                if len(keypoints) == 0:
                    print("failed to load keypoints %s, skip" %(keypoint_file_path))
                    logging.info("failed to load keypoints %s, skip" %(keypoint_file_path))
                    continue

                num_rep = 1

                for _ in range(num_rep):
                    self.data.append(keypoints)
                    self.labels.append(label_id)
                    self.file_path.append(file_name)
                    self.sample_intervals.append(sample_interval)
                    self.timeranges.append([])
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoints = self.data[idx]
        label = self.labels[idx]
        file_path = self.file_path[idx]
        sample_interval = self.sample_intervals[idx]

        # loaded order [T, V*C], expected dimension order [C, V, T]
        data_numpy = np.asarray(keypoints)
        T, VC = data_numpy.shape
        assert VC == self.input_channels * self.num_keypoints
        data_numpy = data_numpy.reshape(T, self.num_keypoints, self.input_channels)

        # random sample a window out of a video
        if self.uniform_sample:
            data_numpy = self.uniform_sample_frame(data_numpy, self.temporal_win * sample_interval)
        else:
            data_numpy = self.consecutive_sample_frame(data_numpy, self.temporal_win * sample_interval, label)

        ## data augmentation for training 
        if self.partition == "train":
            data_numpy = self.random_move(data_numpy)
            data_numpy = self.random_scale(data_numpy)
            
        # T, V, C  -> T, C, V (preferred by coreml) 
        keypoints_output = torch.FloatTensor(data_numpy[::sample_interval, :, :]).permute(0, 2, 1).contiguous()

        return keypoints_output, label, file_path
    
    def load_json(self, path):  # load keypoints from json
        json_data = json.load(open(path))
        keypoints = []
        for poses_frame in json_data:
            try:
                hand_keypoints = np.array(poses_frame["keypoints"])
            except:
                print(f"failed path: {path}")
            else:
                # workaround the string the format in json
                hand_keypoints = np.asarray(ast.literal_eval(poses_frame["keypoints"]), dtype=np.float32)[0] # first person/hand
            
            assert hand_keypoints.shape == (3, self.num_keypoints)  # [C, V]
            hand_keypoints = np.transpose(hand_keypoints, (1, 0))
            keypoints.append(hand_keypoints.flatten().tolist())  # flattern to be consisent with existing format
        return keypoints

    def consecutive_sample_frame(self, data, size, label, auto_pad=True):
        # randomly sample a window of consecutive frames
        T, V, C= data.shape
        # image mode: window = 1
        if size == 1:
            begin = int(T / 2) # round down (truncate)
            sampled_data = data[begin:begin + size]
            return sampled_data
        # video mode: window > 1
        if T == size:
            return data
        elif T < size:
            if auto_pad:
                begin = random.randint(0, size - T)  # random begin
                data_paded = np.zeros((size, V, C))
                data_paded[begin:begin + T, :, :] = data
                return data_paded
            else:
                return data
        else:
            # compute the random begin point to sample a window consecutively
            begin = random.randint(0, T - size)
            # sample the window data
            new_data = data[begin:begin + size]
            return new_data

    def uniform_sample_frame(self, data, size, auto_pad=True):
        # randomly sample a window of non-consecutive frames, these frames are uniformly distributed across the entire video
        T, V, C = data.shape
        if T == size:
            return data
        elif T < size:
            if auto_pad:
                begin = random.randint(0, size - T)  # random begin
                data_paded = np.zeros((size, V, C))
                data_paded[begin:begin + T, :, :] = data
                return data_paded
            else:
                return data
        else:
            each_num = (T - 1) / (size - 1)
            idx_list = [0, T - 1]
            for i in range(size):
                index = round(each_num * i)
                if index not in idx_list and index < T:
                    idx_list.append(index)
            idx_list.sort()

            while len(idx_list) < size:
                idx = random.randint(0, T - 1)
                if idx not in idx_list:
                    idx_list.append(idx)
            idx_list.sort()
            selected_frames = [data[idx] for idx in idx_list]
            new_data = np.stack((selected_frames), axis=0)            
            return new_data

    def random_scale(self, data_numpy): # 20% (uniform)scale
        T, V, C = data_numpy.shape
        ratio = 0.2
        factor = np.random.uniform(1 - ratio, 1 + ratio)
        for t in range(T):
            for j_id in range(V):
                data_numpy[t][j_id][0:2] *= factor # only shift x, y, no changes to confidence
        data_numpy = np.array(data_numpy)
        return data_numpy

    # apply augmentation to keypoints position
    def random_move(self,
                    data_numpy,
                    angle_candidate=[-10., -5., 0., 5., 10.],
                    scale_candidate=[0.9, 1.0, 1.1],
                    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2]):
        # input:T, V, C
        T, V, C = data_numpy.shape
        node = [0, T]
        num_node = len(node)
        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                        [np.sin(a) * s, np.cos(a) * s]])

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[i_frame, :, 0:2] # keypoint position
            new_xy = np.dot(theta[:, :, i_frame], xy.transpose()) # apply transform to  position [x, y] of all keypoints i_frame
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[i_frame, :, 0:2] = new_xy.transpose()
        return data_numpy
