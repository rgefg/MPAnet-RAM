import os
import re
import os.path as osp
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from glob import glob
import json

'''
    Specific dataset classes for person re-identification dataset. 
'''

class SYSUDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()
            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)
        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]        #只保留行人ID在selected_ids中的图像
        #selectedid是训练 + 验证集的所有行人 ID，一起用来训练
        selected_ids = [int(i) for i in selected_ids]

        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]      #只保留摄像头编号为1 2 4 5的图像
        elif mode == 'query':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]          #只保留摄像头编号为 3 和 6的图像

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]     
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            print("id_map keys:", id_map.keys())
            print("example id:", int(img_paths[0].split('/')[-2]))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):                     #传入索引，传回数据
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

class CMGDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']
        self.root = root
        if mode == 'train':
            train_ids = np.load(os.path.join(root, 'npy', 'train_id.npy'))
            #print(train_ids)
            selected_ids = train_ids.tolist()
        else:
            test_ids = np.load(os.path.join(root, 'npy', 'test_id.npy'))
            selected_ids = test_ids.tolist()

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)
        #print(num_ids),320

        # 搜索所有图像路径
        img_paths = glob(os.path.join(root, 'camera*', '*', '*.jpeg'))
        #print(os.listdir(root))
        #print("Example img paths:")
        #for p in img_paths[:5]:
            #print(len(p))

        # 仅保留 ID 在 selected_ids 中的图像
        img_paths = [p for p in img_paths if int(os.path.basename(os.path.dirname(p))) in selected_ids]
        #print(f"img_path.shape:{len(img_paths)}")
        if mode == 'gallery':
            img_paths = [p for p in img_paths if int(os.path.basename(p).split('_')[0][-1]) in (1,2,3)]
        elif mode == 'query':
            img_paths = [p for p in img_paths if int(os.path.basename(p).split('_')[0][-1]) in (4,5,6)]

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(os.path.basename(p).split('_')[0][-1]) for p in img_paths]  # 提取camera id
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))  # map三位id到0~num_ids-1
            self.ids = [id_map[int(os.path.basename(os.path.dirname(p)))] for p in img_paths]
        else:
            self.ids = [int(os.path.basename(os.path.dirname(p))) for p in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        #if self.transform is not None:
            #img = self.transform(img)

        label = torch.tensor(self.ids[index], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[index], dtype=torch.long)
        index = torch.tensor(index, dtype=torch.long)

        file_name = os.path.basename(path).replace('.jpeg', '.json')
        pid_dir = os.path.basename(os.path.dirname(path))
        json_path = os.path.join(self.root, 'json', pid_dir, file_name)

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                bbox=json_data.get('bbox', [])
        else:
            bbox=[]  # 若无json文件，则为空
        img.transform = self.transform if self.transform is not None else lambda x: x
        return img, label, cam, path, index,bbox
    

class RegDBDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = '1'
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_thermal_'+num+'.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_thermal_'+num+'.txt','r'))

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in img_paths] 
        # the visible cams are 1 2 4 5 and thermal cams are 3 6 in sysu
        # to simplify the code, visible cam is 2 and thermal cam is 3 in regdb
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item


class MarketDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        self.transform = transform

        if mode == 'train':
            img_paths = glob(os.path.join(root, 'bounding_box_train/*.jpg'), recursive=True)
        elif mode == 'gallery':
            img_paths = glob(os.path.join(root, 'bounding_box_test/*.jpg'), recursive=True)
        elif mode == 'query':
            img_paths = glob(os.path.join(root, 'query/*.jpg'), recursive=True)
        
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        relabel = mode == 'train'
        self.img_paths = []
        self.cam_ids = []
        self.ids = []
        for fpath in img_paths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            self.img_paths.append(fpath)
            self.ids.append(all_pids[pid])
            self.cam_ids.append(cam - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
