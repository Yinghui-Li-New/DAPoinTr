from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py
from tqdm import tqdm
import pickle
from numpy.random import RandomState
import trimesh
import matplotlib.pyplot as plt

from tqdm import tqdm
import glob
from utils.io import read_ply_xyz, read_ply_from_file_list
from utils.pc_transform import swap_axis
from plyfile import PlyData

class CRNShapeNet(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """
    def __init__(self, args):
        self.args = args
        self.dataset_path = self.args.dataset_path
        self.class_choice = self.args.class_choice
        self.split = self.args.split

        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]
        
        np.random.seed(0)
        cat_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']

        cat_id = cat_ordered_list.index(self.class_choice.lower())
        self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])                      

    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx]) # fast alr
        label = self.labels[index]
        partial = torch.from_numpy(self.partial[full_idx])
        return gt, partial, full_idx

    def __len__(self):
        return len(self.index_list)
    
class KITTIDataset():
    def __init__(self, load_path):
        self.point_clouds = []
        file_list = glob.glob(load_path + '*.ply')
        total_num = len(file_list)
        for i in range(total_num):
            file_name = load_path + str(i) + '.ply'
            ply_file = PlyData.read(file_name)
            pc = np.array([ply_file['vertex']['x'], ply_file['vertex']['y'], ply_file['vertex']['z']])
            pc = np.transpose(pc,(1,0))
            self.point_clouds.append(pc)
        return

class RealWorldPointsDataset:
    def __init__(self, mesh_dir, batch_size=50, npoint=2048, shuffle=True, split='train', random_seed=None):
        '''
        part_point_cloud_dir: the directory contains the oringal ply point clouds
        batch_size:
        npoint: a fix number of points that will sample from the point clouds
        shuffle: whether to shuffle the order of point clouds
        normalize: whether to normalize the point clouds
        split: 
        extra_ply_point_clouds_list: a list contains some extra point cloud file names, 
                                     note that only use it in test time, 
                                     these point clouds will be inserted in front of the point cloud list,
                                     which means extra clouds get to be tested first
        random_seed: 
        '''
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mesh_dir = mesh_dir
        self.npoint = npoint
        self.split = split
        self.random_seed = random_seed

        # make a random generator
        self.rand_gen = RandomState(self.random_seed)
        #self.rand_gen = np.random

        # list of meshes
        self.meshes = self._read_all_meshes(self.mesh_dir) # a list of trimeshes
        self._preprocess_meshes_as_ShapeNetV2(self.meshes) # NOTE: using different processing...

        self.point_clouds = self._pre_sample_points(self.meshes)

        self.reset()

    def _shuffle_list(self, l):
        self.rand_gen.shuffle(l)
    
    def _preprocess_meshes_old(self, meshes):
        '''
        currently, just normalize all meshes, according to the height
        also, snap chairs to the ground
        '''

        max_height = -1

        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])

            if height > max_height:
                max_height = height
            
        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])
            scale_factor = height / max_height

            bbox_center = np.mean(bbox.vertices, axis=0)
            bbox_center[1] = height / 2.0 # assume that the object is alreay snapped to ground

            trans_v = -bbox_center 
            trans_v[1] += mesh.bounding_box.extents[1]/2.
            mesh.apply_translation(trans_v) # translate the bottom center bbox center to ori

            mesh.apply_scale(scale_factor) # do scaling

        return 0
    
    def _preprocess_meshes(self, meshes):
        '''
        assume the input mesh has already been snapped to the ground
        1. normalize to fit within a unit cube
        2. center the bottom center to the original
        '''

        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])
            extents = mesh.bounding_box.extents.copy()
            extents[1] = height

            scale_factor = 1.0 / np.amax(extents)

            bbox_center = np.mean(bbox.vertices, axis=0)

            trans_v = -bbox_center 
            trans_v[1] = 0 # assume already snap to the ground, so do not translate along y
            mesh.apply_translation(trans_v) # translate the center bbox bottom to ori

            mesh.apply_scale(scale_factor)
        
    def _preprocess_meshes_as_ShapeNetV2(self, meshes):
        '''
        the input meshes are pre-aligned facing -z and snapped onto the ground.
        then, make the diagonal length of the axis aligned bounding box around the shape is equal to 1
        center object bbox center to the original
        '''
        for mesh in meshes:
            
            pts_min = np.amin(mesh.vertices, axis=0)
            pts_min[1] = 0 # using the real height
            pts_max = np.amax(mesh.vertices, axis=0)
            diag_len = np.linalg.norm(pts_max - pts_min)

            scale_factor = 1.0 / diag_len

            bbox_center = (pts_max + pts_min) / 2.0

            trans_v = -bbox_center 
            mesh.apply_translation(trans_v) # translate the center of bbox to ori

            mesh.apply_scale(scale_factor)
        return
    
    def _read_all_meshes(self, mesh_dir):
        meshes_cache_filename = os.path.join(os.path.dirname(mesh_dir), 'meshes_cache_%s.pickle'%(self.split))
        
        if os.path.exists(meshes_cache_filename):
            #print('Loading cached pickle file: %s'%(meshes_cache_filename))
            p_f = open(meshes_cache_filename, 'rb')
            mesh_list = pickle.load(p_f)
            p_f.close()
        else:
            split_filename = os.path.join(os.path.dirname(mesh_dir), os.path.basename(mesh_dir)+'_%s_split.pickle'%(self.split))
            with open(split_filename, 'rb') as pf:
                mesh_name_list = pickle.load(pf)
            mesh_filenames = []
            for mesh_n in mesh_name_list:
                mesh_filenames.append(os.path.join(mesh_dir, mesh_n))
            mesh_filenames.sort() # NOTE: sort the file names here!

            print('Reading and caching...')
            mesh_list = []
            for mn in tqdm(mesh_filenames):
                m_fn = os.path.join(mesh_dir, mn)
                mesh = trimesh.load(m_fn)
            
                mesh_list.append(mesh)
            
            p_f = open(meshes_cache_filename, 'wb')
            pickle.dump(mesh_list, p_f)
            print('Cache to %s'%(meshes_cache_filename))
            p_f.close()

        return mesh_list

    def _pre_sample_points(self, meshes):
        presamples_cache_filename = os.path.join(os.path.dirname(self.mesh_dir), 'presamples_cache_%s.pickle'%(self.split))
        if os.path.exists(presamples_cache_filename):
            #print('Loading cached pickle file: %s'%(presamples_cache_filename))
            p_f = open(presamples_cache_filename, 'rb')
            points_list = pickle.load(p_f)
            p_f.close()
        else:
            print('Pre-sampling...')
            points_list = []
            for m in tqdm(meshes):
                samples, _ = trimesh.sample.sample_surface_even(m, self.npoint * 10)
                points_list.append(np.array(samples))

            p_f = open(presamples_cache_filename, 'wb')
            pickle.dump(points_list, p_f)
            p_f.close()

            print('Pre-sampling done and cached.')

        return points_list

    def reset(self):
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_list(self.meshes)

    def has_next_batch(self):
        num_batch = np.floor(len(self.meshes) / self.batch_size) + 1
        if self.batch_idx < num_batch:
            return True
        return False
    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = np.zeros((self.batch_size, self.npoint, 3))
        for i in range(start_idx, end_idx):

            if i >= len(self.point_clouds):
                i_tmp = i % len(self.point_clouds)
                pc_cur = self.point_clouds[i_tmp]
            else:
                pc_cur = self.point_clouds[i] # M x 3
                
            choice_cur = self.rand_gen.choice(pc_cur.shape[0], self.npoint, replace=True)
            idx_cur = i % self.batch_size
            data_batch[idx_cur] = pc_cur[choice_cur, :]

        self.batch_idx += 1
        return data_batch

    def get_npoint(self):
        return self.npoint

class PlyDataset(data.Dataset):
    """
    datasets that with Ply format
    without GT: MatterPort, ScanNet, KITTI
        Datasets provided by pcl2pcl
    with GT: PartNet, each subdir under args.dataset_path contains 
        the partial shape raw.ply and complete shape ply-2048.txt.
        Dataset provided by MPC

    """
    def __init__(self, args):
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path

        if self.dataset in ['MatterPort', 'ScanNet', 'KITTI']:
            input_pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            input_ls = read_ply_from_file_list(input_pathnames)
            # swap axis as pcl2pcl and ShapeInversion have different canonical pose
            input_ls_swapped = [swap_axis(itm, swap_mode='n210') for itm in input_ls]
            self.input_ls = input_ls_swapped
            self.stems = range(len(self.input_ls))
        elif self.dataset in ['PartNet']:
            pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            basenames = [os.path.basename(itm) for itm in pathnames]

            self.stems = [int(itm) for itm in basenames]

            input_ls = [read_ply_xyz(os.path.join(itm,'raw.ply')) for itm in pathnames]
            gt_ls = [np.loadtxt(os.path.join(itm,'ply-2048.txt'),delimiter=';').astype(np.float32) for itm in pathnames]
 
            # swap axis as multimodal and ShapeInversion have different canonical pose
            self.input_ls = [swap_axis(itm, swap_mode='210') for itm in input_ls]
            self.gt_ls = [swap_axis(itm, swap_mode='210') for itm in gt_ls]
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        if self.dataset in ['MatterPort','ScanNet','KITTI']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            return (input_pcd, stem)
        elif self.dataset  in ['PartNet']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            gt_pcd = self.gt_ls[index]
            return (gt_pcd, input_pcd, stem)
    
    def __len__(self):
        return len(self.input_ls)  

class RealDataset(data.Dataset):
    """
    datasets that with Ply format
    without GT: MatterPort, ScanNet, KITTI
        Datasets provided by pcl2pcl
    with GT: PartNet, each subdir under args.dataset_path contains 
        the partial shape raw.ply and complete shape ply-2048.txt.
        Dataset provided by MPC

    """
    def __init__(self, args):
        #self.dataset = args.dataset
        #self.dataset_path = args.dataset_path
        self.dataset = args.dataset#'ScanNet'
        self.random_seed = 0
        self.rand_gen = RandomState(self.random_seed)

        if self.dataset in ['MatterPort', 'ScanNet', 'KITTI']:
            if self.dataset == 'ScanNet':
                REALDATASET = RealWorldPointsDataset('./datasets/data/scannet_v2_'+args.class_choice+'s_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split=args.split, random_seed=0)
            elif self.dataset == 'MatterPort':
                if args.split in ['train', 'trainval']:
                    REALDATASET = RealWorldPointsDataset('./datasets/data/scannet_v2_'+args.class_choice+'s_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split=args.split, random_seed=0)
                else:
                    REALDATASET = RealWorldPointsDataset('./datasets/data/MatterPort_v1_'+args.class_choice+'_Yup_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split=args.split, random_seed=0)
            elif self.dataset == 'KITTI':
                if args.split in ['train']:
                    REALDATASET = KITTIDataset('./datasets/data/KITTI_frustum_data_for_pcl2pcl/point_cloud_train/')
                elif args.split in ['test', 'val']:
                    REALDATASET = KITTIDataset('./datasets/data/KITTI_frustum_data_for_pcl2pcl/point_cloud_val/')
            input_ls = REALDATASET.point_clouds 
            # swap axis as pcl2pcl and ShapeInversion have different canonical pose
            input_ls_swapped = [np.float32(swap_axis(itm, swap_mode='n210')) for itm in input_ls]
            self.input_ls = input_ls_swapped
            self.stems = range(len(self.input_ls))
        elif self.dataset in ['PartNet']:
            pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            basenames = [os.path.basename(itm) for itm in pathnames]

            self.stems = [int(itm) for itm in basenames]

            input_ls = [read_ply_xyz(os.path.join(itm,'raw.ply')) for itm in pathnames]
            gt_ls = [np.loadtxt(os.path.join(itm,'ply-2048.txt'),delimiter=';').astype(np.float32) for itm in pathnames]
 
            # swap axis as multimodal and ShapeInversion have different canonical pose
            self.input_ls = [swap_axis(itm, swap_mode='210') for itm in input_ls]
            self.gt_ls = [swap_axis(itm, swap_mode='210') for itm in gt_ls]
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        if self.dataset in ['MatterPort','ScanNet','KITTI']:
            stem = self.stems[index]
            choice = self.rand_gen.choice(self.input_ls[index].shape[0], 2048, replace=True)
            input_pcd = self.input_ls[index][choice,:]
            return (input_pcd, stem)
        elif self.dataset  in ['PartNet']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            gt_pcd = self.gt_ls[index]
            return (gt_pcd, input_pcd, stem)
    
    def __len__(self):
        return len(self.input_ls)  

class GeneratedDataset(data.Dataset):
    """
    datasets that with Ply format
    without GT: MatterPort, ScanNet, KITTI
        Datasets provided by pcl2pcl
    with GT: PartNet, each subdir under args.dataset_path contains 
        the partial shape raw.ply and complete shape ply-2048.txt.
        Dataset provided by MPC

    """
    def __init__(self, args):
        #self.dataset = args.dataset
        #self.dataset_path = args.dataset_path
        self.dataset = args.dataset#'ModelNet'
        self.category = args.save_inversion_path.split('/')[-3].split('_')[-1]
        if self.dataset == 'ModelNet':
            self.dataset_path = './datasets/ModelNet40_Completion/' + self.category + '/' + args.split
        elif self.dataset == '3D_FUTURE':
            self.dataset_path = './datasets/3D_FUTURE_Completion/' + self.category + '/' + args.split
        self.num_view = 5
        self.random_seed = 0
        self.rand_gen = RandomState(self.random_seed)

        if self.dataset in ['ModelNet', '3D_FUTURE']:
            complete_pathnames = sorted(glob.glob(self.dataset_path+'/*complete.npy'))
            partial_pathnames = sorted(glob.glob(self.dataset_path+'/*partial.npy'))
            self.input_ls = [np.load(itm).astype(np.float32) for itm in partial_pathnames]
            self.gt_ls = [np.load(itm).astype(np.float32) for itm in complete_pathnames]
            self.stems = range(len(self.input_ls))
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        if self.dataset in ['MatterPort','ScanNet','KITTI']:
            stem = self.stems[index]
            choice = self.rand_gen.choice(self.input_ls[index].shape[0], 2048, replace=True)
            input_pcd = self.input_ls[index][choice,:]
            return (input_pcd, stem)
        elif self.dataset  in ['PartNet', 'ModelNet', '3D_FUTURE']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            gt_pcd = self.gt_ls[index]
            return (gt_pcd, input_pcd, stem)
    
    def __len__(self):
        return len(self.input_ls)



class Dataloader(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.source_dataset_name = config.dataset_source
        self.target_dataset_name = config.dataset_target
        self.split = config.split

        # 初始化 args
        self.initialize_args()

        # 设置数据集
        self.source_dataset = self.setup_dataset(self.source_dataset_name, self.config.source_dataset_path)
        self.target_dataset = self.setup_dataset(self.target_dataset_name, self.config.target_dataset_path)

    def initialize_args(self):
        self.args = self.config  # 或者其他初始化 args 的方式
        self.args.split = self.split
        self.args.class_choice = getattr(self.config, 'class_choice', None)

    def adjust_split(self, dataset_name):
        if dataset_name in ['ScanNet', 'MatterPort']:
            return 'trainval'
        else:
            return self.split

    def setup_dataset(self, dataset_name, dataset_path):
        self.args.dataset = dataset_name
        self.args.dataset_path = dataset_path
        self.args.split = self.adjust_split(dataset_name)

        if dataset_name in ['MatterPort', 'ScanNet', 'KITTI', 'PartNet']:
            return PlyDataset(self.args)
        elif dataset_name in ['ModelNet', '3D_FUTURE']:
            return GeneratedDataset(self.config)
        elif dataset_name:  # 添加其他数据集名称的判断
            return CRNShapeNet(self.config)
        else:
            raise ValueError(f"未知的数据集名称: {dataset_name}")

    def __getitem__(self, index):
        source_data = self.source_dataset[index]
        target_data = self.target_dataset[index]
        return source_data, target_data

    def __len__(self):
        return min(len(self.source_dataset), len(self.target_dataset))


if __name__ == "__main__":
    # 创建一个配置对象，这里是示例，具体根据你的实际情况来定义
    config = {
        'dataset_source': 'MatterPort',  # 示例数据集名称
        'dataset_target': 'ScanNet',     # 示例数据集名称
        'split': 'train',                # 示例分割类型
        'source_dataset_path': '/path/to/dataset',  # 数据集路径
        'target_dataset_path': '/path/to/target/dataset',
        # 其他需要的配置项...
    }

    # 创建 Dataloader 实例
    dataloader = Dataloader(config)

    # 测试长度函数
    print(f"数据加载器长度: {len(dataloader)}")

    # 测试获取一些数据项
    for i in range(3):
        source_data, target_data = dataloader[i]
        print(f"样本 {i}:")
        print(f"源数据: {source_data}")
        print(f"目标数据: {target_data}")























