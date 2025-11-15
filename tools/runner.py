import torch
import torch.nn as nn
import os
import numpy as np
from tools import builder
from utils import dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2,UnidirectionalChamferDistance, DirectedHausdorffDistance
from models.DAPoinTr import Criterion
import open3d as o3d
from utils.realtime_render import *
from torch.utils.data import Dataset, DataLoader
from utils import loss_util
from torch.utils.data import Dataset


class PseudoLabelDataset(Dataset):
    def __init__(self, pseudo_labels_data):

        self.pseudo_labels_data = pseudo_labels_data

    def __len__(self):
        return len(self.pseudo_labels_data)

    def __getitem__(self, idx):
        return self.pseudo_labels_data[idx]



def run_net(args, config, train_writer=None, val_writer=None):

    logger = get_logger(args.log_name)

    # build dataset CRN train and CRN test dataset
    train_sampler, train_dataloader = builder.virtual_dataset_builder(args,  config.dataset.train)  #CRN训练集 Chair 79
    config.dataset.test._base_.SPLIT= 'test'
    _, test_dataloader = builder.virtual_dataset_builder(args,config.dataset.test) #CRN测试集

    #new dataset build real data
    config.dataset.train._base_.SPLIT = 'train'
    _,real_train_dataloader = builder.real_dataset_builder(args, config.dataset.train) #3D FUTURE训练
    config.dataset.test._base_.SPLIT = 'test'
    real_test_sampler, real_test_dataloader = builder.real_dataset_builder(args,config.dataset.test) #3D FUTURE测试集

    # build model
    base_model = builder.model_builder(config.model)
    criterion = Criterion()

    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untcdrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    UCD_distance = UnidirectionalChamferDistance()
    UHD_distance = DirectedHausdorffDistance()
    completion_loss = loss_util.Completionloss(loss_func=config.consider_metric)


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)
    base_model.zero_grad()

    #weight dict
    weight_dict = {
        'loss_domain_enc_token': args.domain_enc_token_loss_coef * 0.25 ,
        'loss_domain_dec_token': args.domain_dec_token_loss_coef * 0.25 , 
        'loss_domain_enc_query': args.domain_enc_query_loss_coef * 0.025 , 
        'loss_domain_dec_query': args.domain_dec_query_loss_coef * 0.025, 
        'loss_domain_enc_token_t': args.domain_enc_token_loss_coef * 0.25 ,
        'loss_domain_dec_token_t': args.domain_dec_token_loss_coef * 0.25 ,
        'loss_domain_enc_query_t': args.domain_enc_query_loss_coef * 0.025 ,
        'loss_domain_dec_query_t': args.domain_dec_query_loss_coef * 0.025 ,
        'loss_cmt_geom':args.loss_cmt_geom * 1
    }


    bool_source_train = False
    bool_pseudo_label_train= True
    bool_model_train = True
    if bool_source_train:
        for epoch in range(start_epoch, config.pre_train_epoch):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            base_model.train()

            epoch_start_time = time.time()
            batch_start_time = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            avg_meter_loss = AverageMeter(['loss_partial', 'loss_pc', 'loss_p1', 'loss_p2', 'loss_p3'])
            num_iter = 0
            criterion.train()
            n_batches = len(real_train_dataloader) #
            train_dataloader_iter = iter(train_dataloader) #CRN
            real_train_dataloader_iter = iter(real_train_dataloader)    #3D FUTURE
            len_train_dataloader = len(train_dataloader)
            len_real_train_dataloader = len(real_train_dataloader)
            max_len = max(len_train_dataloader, len_real_train_dataloader)
            for idx in range(max_len):
                #load source data
                try:
                    source_data = next(train_dataloader_iter)
                except StopIteration:
                    train_dataloader_iter =iter(train_dataloader)
                    source_data = next(train_dataloader_iter)

                #load target data                
                try:
                    target_data = next(real_train_dataloader_iter)
                except StopIteration:
                    real_train_dataloader_iter = iter(real_train_dataloader) 
                    target_data = next(real_train_dataloader_iter)

                data_time.update(time.time() - batch_start_time)
                source_gt, source_partial, source_index = source_data
                source_dataset_name = config.dataset.train._base_.NAME 
                target_dataset_name = config.dataset.train.real_dataset

                if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE','ModelNet']:
                    target_gt, target_partial, _ = target_data
                if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
                    target_partial, _ = target_data


                if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE','ModelNet']:
                    source_gt = source_gt.squeeze(0).cuda() 
                    source_partial = source_partial.squeeze(0).cuda() 
                    if config.dataset.train._base_.CLASS_CHOICE == "lamp" or config.dataset.train._base_.CLASS_CHOICE == "table":
                        source_partial, _, _, _ = partial_render_batch(source_gt, source_partial) 

                    #train on source data
                    coarse_point, relative_out, out_source = base_model(source_partial) #[80,192,3]
                    loss_total, losses = completion_loss.get_loss(relative_out, source_partial, source_gt)
                    loss_dict = criterion(out_source, domain_label=0, cmt_loss=False)

                    #train on target data
                    target_partial = target_partial.squeeze(0).cuda()
                    _, _, out_target= base_model(target_partial)
                    loss_dict_t = criterion(out_target, domain_label=1,cmt_loss=False)
                    loss_dict.update(loss_dict_t)
                    domain_loss = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['KITTI','MatterPort','ScanNet']:

                    source_gt = source_gt.cuda() 
                    source_partial = source_partial.cuda() 

                    #train on source data
                    coarse_point, relative_out, out_source = base_model(source_partial) 
                    loss_total, losses = completion_loss.get_loss(relative_out, source_partial, source_gt)
                    loss_dict = criterion(out_source, domain_label=0, cmt_loss=True)

                    #train on target data
                    target_partial = target_partial.squeeze(0).float().cuda()
                    _, target_pred, out_target= base_model(target_partial)
                    loss_dict_t = criterion(out_target, domain_label=1,cmt_loss=True)
                    target_pred_points = target_pred[-1]
                    loss_dict.update(loss_dict_t)
                    domain_loss = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict)        
                       
                else:
                    raise NotImplementedError(f'Train phase do not support {source_dataset_name}')


                num_iter += 1
                _loss =   loss_total + 0.01 * domain_loss 
                _loss.backward()

                if num_iter == config.step_per_update:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                    num_iter = 0
                    optimizer.step()
                    base_model.zero_grad()

                if args.distributed:
                    coarse_loss = dist_utils.reduce_tensor(coarse_loss, args)
                    rebuild_loss = dist_utils.reduce_tensor(rebuild_loss, args)
                else:
                    avg_meter_loss.update(losses)

                if args.distributed:
                    torch.cuda.synchronize()

                n_itr = epoch * n_batches + idx
                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                if idx % 50 == 0:

                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                                (epoch, config.pre_train_epoch, idx + 1, max_len, batch_time.val(), data_time.val(),
                                ['%.6f' % l for l in losses] , optimizer.param_groups[0]['lr']), logger = logger) 


                if config.scheduler.type == 'GradualWarmup':
                    if n_itr < config.scheduler.kwargs_2.total_epoch:
                        scheduler.step()

            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step()
            else:
                scheduler.step()
            epoch_end_time = time.time()

            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                (epoch,  epoch_end_time - epoch_start_time, ['%.6f' % l for l in losses]), logger = logger)

            if epoch % args.val_freq == 0:
                # Validate the current model
                metrics = validate(base_model, real_test_dataloader, epoch, 
                                    ChamferDisL1, ChamferDisL2, UCD_distance, UHD_distance, val_writer, args, config, logger=logger)

                # Save ckeckpoints
                if  metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt_source_best', args, logger = logger)


    if bool_pseudo_label_train:
        best_model_path = os.path.join(args.experiment_path, 'ckpt_source_best.pth')
        if os.path.isfile(best_model_path):
            print("Loading best model...")
            checkpoint = torch.load(best_model_path, map_location='cpu')  
            base_model.load_state_dict(checkpoint['base_model'])  
            print("Best model loaded.")
        else:
            print("Best model not found. Check the path.")
        
        with torch.no_grad():
            cd_threshold = 0  
            pseudo_labels = []
            loss_cmt_geom_values = []
            for epoch in range(10):#10/8

                if args.distributed:
                    train_sampler.set_epoch(epoch)
                base_model.eval()
                criterion.eval()
                real_train_dataloader_iter = iter(real_train_dataloader) 

                for idx in range(len(real_train_dataloader)):
                    try:
                        target_data = next(real_train_dataloader_iter)
                    except StopIteration:
                        real_train_dataloader_iter = iter(real_train_dataloader)
                        target_data = next(real_train_dataloader_iter)

                    source_dataset_name = config.dataset.train._base_.NAME                  
                    target_dataset_name = config.dataset.train.real_dataset

                    if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE','ModelNet']:
                        _, target_partial, _ = target_data  # [10,2048,3]
                    if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
                        target_partial,_ = target_data

                    if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE','ModelNet']:
                        
                        target_partial = target_partial.squeeze(0).cuda()
                        _, relative_xyz, out_target = base_model(target_partial)
                        relative_xyz = relative_xyz[-1]
                        loss_dict = criterion(out_target, domain_label=1,cmt_loss=True)
                        if epoch < 8:
                            loss_cmt_geom_values.append(loss_dict['loss_cmt_geom'].item()) 
                        elif epoch ==8:
                            cd_threshold = torch.mean(torch.tensor(loss_cmt_geom_values)).item()
                        elif epoch > 8:
                            if loss_dict['loss_cmt_geom'].item() < cd_threshold:
                                pseudo_labels.append([relative_xyz, target_partial, idx])
                    elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
                        target_partial = target_partial.squeeze(0).cuda()
                        _, relative_xyz, out_target = base_model(target_partial)
                        relative_xyz = relative_xyz[-1]
                        loss_dict = criterion(out_target, domain_label=1,cmt_loss=True)
                        if epoch < 1:
                            loss_cmt_geom_values.append(loss_dict['loss_cmt_geom'].item())
                        elif epoch ==1:
                            cd_threshold = torch.mean(torch.tensor(loss_cmt_geom_values)).item()
                        elif epoch > 1:
                            if loss_dict['loss_cmt_geom'].item() < cd_threshold:
                                # Build up pseudo dataset
                                pseudo_labels.append([relative_xyz, target_partial, idx])

    if bool_model_train:
        base_model.zero_grad()
        for epoch in range(start_epoch, config.max_epoch+1): 
            if args.distributed:
                train_sampler.set_epoch(epoch)
            
            base_model.train()
            pseudo_labels_data = PseudoLabelDataset(pseudo_labels)
            pseudo_labels_dataloader = DataLoader(pseudo_labels_data, batch_size=1, shuffle=True)
            epoch_start_time = time.time()
            batch_start_time = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            avg_meter_loss = AverageMeter(['loss_partial', 'loss_pc', 'loss_p1', 'loss_p2', 'loss_p3','domain_loss'])
            num_iter = 0

            base_model.train()  # set model to training mode
            criterion.train()
            n_batches = len(pseudo_labels_dataloader) #

            
            pseudo_train_dataloader_iter = iter(pseudo_labels_dataloader)   

            len_pseudo_train_dataloader = len(pseudo_labels_dataloader)

            for idx in range(len_pseudo_train_dataloader):
                try:
                    pseudo_target_data = next(pseudo_train_dataloader_iter)
                except StopIteration:
                    real_train_dataloader_iter = iter(pseudo_labels_dataloader) 
                    pseudo_target_data = next(pseudo_train_dataloader_iter)

                data_time.update(time.time() - batch_start_time)
                pseudo_target_gt, pseudo_target_partial, _ = pseudo_target_data
                source_dataset_name = config.dataset.train._base_.NAME 
                target_dataset_name = config.dataset.train.real_dataset

                if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE','ModelNet']:
                    #train on pseudo label data
                    pseudo_target_partial = pseudo_target_partial.squeeze(0).cuda()
                    pseudo_target_gt = pseudo_target_gt.squeeze(0).cuda()
                    pseudo_coarse_point, relative_xyz, out_pseudo= base_model(pseudo_target_partial) #[80,192,3]
                    pseudo_loss_total, pseudo_losses = completion_loss.get_loss(relative_xyz, pseudo_target_partial, pseudo_target_gt)

                elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
                    pseudo_target_partial = pseudo_target_partial.squeeze(0).cuda()
                    pseudo_target_gt = pseudo_target_gt.squeeze(0).cuda()
                    pseudo_coarse_point, relative_xyz, out_pseudo= base_model(pseudo_target_partial) #[80,192,3]
                    pseudo_loss_total, pseudo_losses = completion_loss.get_loss(relative_xyz, pseudo_target_partial, pseudo_target_gt)
                else:
                    raise NotImplementedError(f'Train phase do not support {source_dataset_name}')

                num_iter += 1

                _loss = pseudo_loss_total 
                _loss.backward()
                # forward
                if num_iter == config.step_per_update:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                    num_iter = 0
                    optimizer.step()
                    base_model.zero_grad()

                if args.distributed:
                    coarse_loss = dist_utils.reduce_tensor(coarse_loss, args)
                    rebuild_loss = dist_utils.reduce_tensor(rebuild_loss, args)
                else:
                    avg_meter_loss.update(pseudo_losses)

                if args.distributed:
                    torch.cuda.synchronize()
                n_itr = epoch * n_batches + idx

                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/partial_matching', pseudo_losses[0], n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_pc', pseudo_losses[1], n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_p1', pseudo_losses[2], n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_p2', pseudo_losses[3], n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_p3', pseudo_losses[4], n_itr)

    
                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()
                if idx % 100 == 0:
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                                (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                                ['%.4f' % l for l in pseudo_losses], optimizer.param_groups[0]['lr']), logger = logger)

                if config.scheduler.type == 'GradualWarmup':
                    if n_itr < config.scheduler.kwargs_2.total_epoch:
                        scheduler.step()

            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step()
            else:
                scheduler.step()
            epoch_end_time = time.time()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Epoch/partial_matching', avg_meter_loss.avg(0), epoch)
                train_writer.add_scalar('Loss/Epoch/cd_pc', avg_meter_loss.avg(1), epoch)
                train_writer.add_scalar('Loss/Epoch/cd_p1', avg_meter_loss.avg(2), epoch)
                train_writer.add_scalar('Loss/Epoch/cd_p2', avg_meter_loss.avg(3), epoch)
                train_writer.add_scalar('Loss/Epoch/cd_p3', avg_meter_loss.avg(4), epoch)

            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in pseudo_losses]), logger = logger)

            if epoch % args.val_freq == 0:
                # Validate the current model
                metrics = validate(base_model, real_test_dataloader, epoch, 
                                    ChamferDisL1, ChamferDisL2, UCD_distance, UHD_distance,val_writer, args, config, logger=logger)

                # Save ckeckpoints
                if  metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
            

            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
            if (config.max_epoch - epoch) < 2:
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model,real_test_dataloader, epoch, ChamferDisL1, ChamferDisL2, UCD_distance, UHD_distance, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    test_losses = AverageMeter(['Coarse_Loss_L1','Coarse_Loss_L2','Rebuild_Loss_L1','Rebuild_Loss_L2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(real_test_dataloader) # bs is 1

    interval =  n_samples // 10
    source_dataset_name = config.dataset.train._base_.NAME 
    target_dataset_name = config.dataset.train.real_dataset
    with torch.no_grad():
        if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE', 'ModelNet']:
            for idx, data in enumerate(real_test_dataloader):

                target_gt,target_partial, _ = data  
                if args.use_gpu:
                    target_partial = target_partial.cuda()
                    target_gt = target_gt.cuda()

                # Calculate losses for target
                coarse_points, relative_xyz, _= base_model(target_partial)
                rebuild_dense_points = relative_xyz[-1]

                coarse_loss_l1 =  ChamferDisL1(coarse_points, target_gt)
                coarse_loss_l2 =  ChamferDisL2(coarse_points, target_gt)
                rebuild_loss_l1 =  ChamferDisL1(rebuild_dense_points, target_gt)
                rebuild_loss_l2 =  ChamferDisL2(rebuild_dense_points, target_gt)

                test_losses.update([coarse_loss_l1.item() * 10000, coarse_loss_l2.item() * 10000, 
                                    rebuild_loss_l1.item() * 10000, rebuild_loss_l2.item() * 10000])

                _metrics = Metrics.get(rebuild_dense_points, target_gt)
                if args.distributed:
                    _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                test_metrics.update(_metrics)

                if (idx + 1) % interval == 0:
                    print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                        (idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()],
                        ['%.4f' % m for m in _metrics]), logger=logger)
        elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
            for idx, data in enumerate(real_test_dataloader):

                target_partial, _ = data   #[1,2048,3],[1,2048,3]
                if args.use_gpu:
                    target_partial = target_partial.float().cuda()

                # Calculate losses for target
                coarse_points, relative_xyz, _= base_model(target_partial)
                rebuild_dense_points = relative_xyz[-1]
                uhd_rebuild_loss =  UHD_distance(target_partial.permute([0,2,1]),rebuild_dense_points.permute([0,2,1]))
                ucd_rebuild_loss =  UCD_distance(target_partial, rebuild_dense_points)

                test_losses.update([ucd_rebuild_loss.item()*10000, uhd_rebuild_loss.item()*100])

                _metrics = Metrics.get(target_partial,rebuild_dense_points)
                if args.distributed:
                    _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                test_metrics.update(_metrics)

                if (idx + 1) % interval == 0:
                    print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                        (idx + 1, n_samples, ['%.6f' % l for l in test_losses.val()],
                        ['%.6f' % m for m in _metrics]), logger=logger)

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.6f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
    
    # Print testing results
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = '\t\t'
    
    for metric in test_metrics.items:
        msg += metric + '\t'
    print_log(msg, logger=logger)

    msg=''
    msg += 'Overall\t'
    for value in test_metrics.avg():
        msg += '%.6f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse_L1', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Sparse_L2', test_losses.avg(1), epoch)
        #val_writer.add_scalar('Loss/Epoch/Dense_L1', test_losses.avg(2), epoch)
        #val_writer.add_scalar('Loss/Epoch/Dense_L2', test_losses.avg(3), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE', 'ModelNet']:
        return Metrics(config.consider_metric_2, test_metrics.avg())
    elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
        return Metrics(config.consider_metric_3, test_metrics.avg())





def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    config.dataset.test._base_.SPLIT = 'test'
    _, test_dataloader = builder.real_dataset_builder(args,config.dataset.test)
    category_metrics = dict()
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    UnidirectionalCD = UnidirectionalChamferDistance()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, UnidirectionalCD, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, UnidirectionlCD ,args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx,data in enumerate(test_dataloader):
            
            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test.real_dataset
            if dataset_name in ['3D_FUTURE','ModelNet']:
                partial = data[1].cuda()
                gt = data[0].cuda()

                coarse_points, relative_xyz, _= base_model(partial)

                dense_points = relative_xyz[-1]
                if idx < 500:
                    visualize_point_cloud_batch(dense_points, f"pred_point_{idx}")
                    visualize_point_cloud_batch(gt, f"gt_{idx}")
                    visualize_point_cloud_batch(partial, f"partial_{idx}")
                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                test_metrics.update(_metrics)
            elif dataset_name in ['KITTI','ScanNet','MatterPort']:
                partial = data[0].cuda()
                coarse_points, relative_xyz, _= base_model(partial)
                dense_points= relative_xyz[-1]
                if idx < 200:
                    visualize_point_cloud_batch(dense_points, f"pred_point_{idx}")
                    visualize_point_cloud_batch(partial, f"partial_{idx}")
                ucd_loss = UnidirectionlCD(partial, dense_points)

                _metrics = Metrics.get(partial, dense_points, require_emd=True)
                test_metrics.update(_metrics)


            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 100 == 0:
                print_log('Test[%d/%d]  Losses = %s Metrics = %s' %
                            (idx + 1, n_samples,  ['%.4f' % l for l in test_losses.val()], 
                            ['%.6f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.6f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    print_log('============================ TEST RESULTS ============================',logger=logger)

    msg =''
    for metric in test_metrics.items:
        msg += metric + '\t'
    print_log(msg, logger=logger)
    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.6f \t' % value
    print_log(msg, logger=logger)
    return 


def visualize_point_cloud_batch(batch_points, name):

    output_dir = "point_virtualization"
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(batch_points, torch.Tensor):
        batch_points = batch_points.cpu().double().numpy()
    elif isinstance(batch_points, np.ndarray):
        batch_points = batch_points.astype(np.float64).copy()

    if batch_points.ndim == 3 and batch_points.shape[2] == 3:

        for i, points in enumerate(batch_points):
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("Input dimension error。")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # generate directionary and store files
            filename = os.path.join(output_dir, f"point_cloud_{i}_{name}.ply")
            o3d.io.write_point_cloud(filename, pcd)