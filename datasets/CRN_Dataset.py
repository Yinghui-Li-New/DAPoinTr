from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py
import random
from .build import DATASETS
from .io import IO
import torch.utils.data as data

#用于创建机器学习或深度学习任务的自定义类
@DATASETS.register_module()
class CRNShapeNet(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """
    def __init__(self, config):
        #self.args = args    #博涵数据集的配置选项，如数据集路径，类别选择和数据分割类型
        self.dataset_path = config.DATA_PATH #data/our_data
        self.class_choice = config.CLASS_CHOICE #chair
        self.split = config.SPLIT   #train

        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5') ##data/our_data/train_data.h5
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()] #gt的形状为[28974,2048,3]
        self.partial = data['incomplete_pcds'][()] #不完整点云形状一致
        self.labels = data['labels'][()]    #28974个label
        """
        
        np.savetxt(filename,self.gt[0])
        """
        np.random.seed(0)
        cat_ordered_list = ['plane','cabinet','car','chair','lamp','sofa','table','watercraft']

        cat_id = cat_ordered_list.index(self.class_choice.lower())  #选择类型，此处为初始定义的chair
        self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])   #筛选出所有cat_id为chair的索引                   

    def __getitem__(self, index):   #训练时使用
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx]) # fast alr
        label = self.labels[index]
        partial = torch.from_numpy(self.partial[full_idx])
        return gt, partial, full_idx

    def __len__(self):
        return len(self.index_list)

