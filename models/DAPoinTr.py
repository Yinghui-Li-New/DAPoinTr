import torch
from torch import nn
import os
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, UnidirectionalChamferDistance
from .Transformer import PCTransformer, SimpleRebuildFCLayer
from .build import MODELS
import torch.nn.functional as F
from utils.net_utils import grad_reverse
from .utils import MLP_Res, fps_subsample
from .SPD import SPD

class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=256, num_pc=128):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class Decoder(nn.Module):
    def __init__(self, dim_feat=256, num_pc=128, num_p0=256,
                 radius=1, bounding=True, up_factors=None):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors): #up_factors = 1,4, 8
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat, partial):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, num_pc, 3) [32,256,3]
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)  #[32,256,3]partial表示原始输入的不完整的点云/self.num_p0=512
        #if return_P0:
        #    arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()#[32,3,512]
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers        
        h = [hidden_dim] * (num_layers - 1) 
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
@MODELS.register_module()
class DAPoinTr(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        num_p0 = config.num_p0
        dim_feat = config.num_p0
        num_pc = config.num_pc
        radius = config.radius
        bounding = config.bounding
        up_factors = config.up_factors
        self.domain_enc = nn.Embedding(1, self.trans_dim*2) 
        self.domain_dec = nn.Embedding(1, self.trans_dim*2)

        self.base_model = PCTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [6, 8], drop_rate = 0., 
                                        num_query = self.num_query, knn_layer = self.knn_layer,
                                        domain_enc=self.domain_enc, domain_dec = self.domain_dec)
        
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, num_p0)
        self.reduce_map2 = nn.Linear(self.num_query, 1)
        self.build_loss_func()

        self.domain_pred_enc = MLP(self.trans_dim, self.trans_dim, output_dim=2, num_layers=3)
        self.domain_pred_dec = MLP(self.trans_dim, self.trans_dim, output_dim=2, num_layers=3)

        self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0, radius=radius, 
                               bounding=bounding, up_factors=up_factors)

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        self.inter_loss = ChamferDistanceL2()
        self.ucd_loss_func = UnidirectionalChamferDistance()

    def get_loss(self, ret, gt, epoch=0):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def get_ucd_loss(self,ret,partial,epoch=0):
        loss_coarse = self.ucd_loss_func(ret[0],partial)
        loss_fine = self.ucd_loss_func(ret[1],partial)
        return loss_coarse, loss_fine

    @torch.jit.unused
    def _set_aux_loss_domain(self, outputs_domains_enc, outputs_domains_dec):
        return [{'pred_domain_enc': a, 'pred_domain_dec': b}
                for a, b in zip(outputs_domains_enc[:-1], outputs_domains_dec[:-1])]
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        return [{'pred_coord': a}
                for a in outputs_coord]
    

    def forward(self, xyz, eta: float=1.0):
        q, coarse_point_cloud, inter_memory, inter_domain_enc, intermediate_q, intermediate_dec, dec_out= self.base_model(xyz) 
        B, M ,C = q.shape 
       
        intermediate_coor =[]
        for i in range(dec_out.size(0)):
            global_feature = self.increase_dim(dec_out[i].transpose(1,2)).transpose(1,2) 
            global_feature = torch.max(global_feature, dim=1)[0] 
            rebuild_feature = torch.cat([
                global_feature.unsqueeze(-2).expand(-1, M, -1),
                dec_out[i],
                coarse_point_cloud], dim=-1)  
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B, M, -1))
            rebuild_feature = self.reduce_map2(rebuild_feature.permute(0,2,1).contiguous())
            relative_xyz = self.decoder(rebuild_feature, xyz)
            rebuild_points = relative_xyz[-1]
            intermediate_coor.append(rebuild_points)
            
        intermediate_coor=torch.stack(intermediate_coor)
        outputs_domain_enc = []
        assert len(inter_domain_enc.shape) == 4 and len(inter_memory.shape) == 4
        assert inter_domain_enc.shape[0] == inter_memory.shape[0]
        for lvl in range(inter_domain_enc.shape[0]):
            domain_pred_enc = self.domain_pred_enc(grad_reverse(
                torch.cat([inter_memory[lvl],inter_domain_enc[lvl]],dim=1),eta=eta
            ))
            outputs_domain_enc.append(domain_pred_enc)
        outputs_domain_enc = torch.stack(outputs_domain_enc) 

        outputs_domain_dec = []
        assert len(intermediate_dec.shape)==4 and len(intermediate_q.shape)==4
        assert intermediate_q.shape[0]==intermediate_dec.shape[0]

        for lvl in range(intermediate_dec.shape[0]):
            domain_pred_dec = self.domain_pred_dec(grad_reverse(    
                torch.cat([intermediate_q[lvl], intermediate_dec[lvl]],dim=1),eta=eta
            ) ) 
            outputs_domain_dec.append(domain_pred_dec)
        outputs_domain_dec = torch.stack(outputs_domain_dec)   
        out = {'pred_domain_enc':outputs_domain_enc[-1],'pred_domain_dec':outputs_domain_dec[-1],\
               'pred_coord':intermediate_coor[-1]}

        if len(outputs_domain_enc)>1:
            out['aux_domain'] = self._set_aux_loss_domain(outputs_domain_enc,outputs_domain_dec)
        
        out['aux_outputs'] = self._set_aux_loss(intermediate_coor)

        return coarse_point_cloud, relative_xyz, out


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.dt_loss = nn.CrossEntropyLoss()
        self.dq_loss = nn.CrossEntropyLoss()
        self.cd_loss = ChamferDistanceL2()

    def loss_cmt(self, outputs, domain_label):

            assert 'aux_outputs' in outputs, "require auxiliary outputs for consistent matching"
            pred_coord_all = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                pred_coord_all.append(
                    aux_outputs['pred_coord']
                )  
            pred_coord_all = torch.stack(pred_coord_all)
            avg_coord = torch.mean(pred_coord_all, dim=(0)).detach()
            loss_cmt_geom = 0.
            for i, pred_coord_tmp in enumerate(pred_coord_all):
                chamfer_dist = self.cd_loss(pred_coord_tmp, avg_coord)
                loss_cmt_geom +=chamfer_dist

            losses = {
                'loss_cmt_geom': loss_cmt_geom / len(pred_coord_all)
            }

            return losses
    
    def loss_domains(self, outputs, domain_label):
        assert 'pred_domain_enc' in outputs and 'pred_domain_dec' in outputs
        domain_label_list = [domain_label for _ in range(outputs['pred_domain_enc'].shape[0])] #10
        domain_pred_enc = outputs['pred_domain_enc']       #[10,129,2]
        domain_pred_dec = outputs['pred_domain_dec']    #[10,129,2]
        
        B, len_enc, len_dec = domain_pred_enc.shape[0], domain_pred_enc.shape[1] - 1, domain_pred_dec.shape[1] - 1
        domain_pred_enc_token, domain_pred_enc_query = torch.split(domain_pred_enc, len_enc, dim=1) #[64,128,2]/ [64,1,2]
        domain_pred_dec_token, domain_pred_dec_query = torch.split(domain_pred_dec, len_dec, dim=1)

        domain_pred_enc_token = domain_pred_enc_token.flatten(0, 1) #[8192,2]
        domain_pred_enc_query = domain_pred_enc_query.squeeze(1) #[64,2]
        domain_pred_dec_token = domain_pred_dec_token.flatten(0, 1) #[8192,2]
        domain_pred_dec_query = domain_pred_dec_query.squeeze(1)    #[64,2]

        domain_label_query = torch.tensor(domain_label_list, dtype=torch.long, device=domain_pred_enc_query.device)
        domain_label_enc_token = domain_label_query[:, None].expand(B, len_enc).flatten(0, 1) #[8192]
        domain_label_dec_token = domain_label_query[:, None].expand(B, len_dec).flatten(0, 1)  #

        loss_domain_enc_token = self.dt_loss(domain_pred_enc_token, domain_label_enc_token)
        loss_domain_dec_token = self.dt_loss(domain_pred_dec_token, domain_label_dec_token)
        loss_domain_enc_query = self.dq_loss(domain_pred_enc_query, domain_label_query)
        loss_domain_dec_query = self.dq_loss(domain_pred_dec_query, domain_label_query)

        losses = {
            'loss_domain_enc_token': loss_domain_enc_token,
            'loss_domain_dec_token': loss_domain_dec_token,
            'loss_domain_enc_query': loss_domain_enc_query,
            'loss_domain_dec_query': loss_domain_dec_query,
        }
        if domain_label == 1:
            losses = {k + '_t': v for k, v in losses.items()}

        return losses
    
    def get_loss(self, loss, outputs, **kwargs):
        loss_map = {
            'domain': self.loss_domains,
            'cmt':self.loss_cmt
        }
        return loss_map[loss](outputs, **kwargs)

    def forward(self, outputs,  domain_label, cmt_loss):
        losses = {}

        losses = self.loss_domains(outputs, domain_label)
        if 'aux_domain' in outputs:
            for i, aux_domain in enumerate(outputs['aux_domain']):
                l_dict = self.loss_domains(aux_domain, domain_label)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        if cmt_loss:
            cmt_loss =self.loss_cmt(outputs,domain_label)
            losses.update(cmt_loss)

        return  losses
