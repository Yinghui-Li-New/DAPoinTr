# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import torch

import chamfer
class DirectedHausdorffDistance(object):
    """
    Hausdorf distance
    """
    def __init__(self, reduce_mean=True):
        # super(DirectedHausdorff,self).__init__()
        self.reduce_mean = reduce_mean
    
    def __call__(self, point_cloud1, point_cloud2):
        """
        :param point_cloud1: (B, 3, N)  partial
        :param point_cloud2: (B, 3, M) output
        :return: directed hausdorff distance, A -> B
        """
        n_pts1 = point_cloud1.shape[2]
        n_pts2 = point_cloud2.shape[2]

        pc1 = point_cloud1.unsqueeze(3)
        pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
        pc2 = point_cloud2.unsqueeze(2)
        pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

        l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

        shortest_dist, _ = torch.min(l2_dist, dim=2)

        hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

        if self.reduce_mean:
            hausdorff_dist = torch.mean(hausdorff_dist)

        return hausdorff_dist
    
class UnidirectionalChamferDistance(torch.nn.Module):
    ''' Unidirectional Chamfer Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super(UnidirectionalChamferDistance, self).__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros1].unsqueeze(dim=0)  # assume xyz2 has the same mask as xyz1

        dist1, _ = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1)


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2

class ChamferDistanceL1_PM(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, _ = ChamferFunction.apply(xyz1, xyz2)
        dist1 = torch.sqrt(dist1)
        return torch.mean(dist1)

