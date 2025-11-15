import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--virtualdataset', type=str, default='CRN', help='Virtual dataset name')
    parser.add_argument('--realdataset', type=str, default='3D_FUTURE', help='Real dataset name')
    parser.add_argument('--class_choice', type=str, default='chair', help='Class choice for training')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--save_inversion_path', type=str, default='./logs/3D_FUTURE_chair', help='Path to save inversion')
    parser.add_argument('--dataset_path', type=str, default='./data/3D_Future_Completion/', help='Path to dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)   
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    #loss coefficientrs for DA

    parser.add_argument('--domain_loss_coef', default=None, type=float,
                        help='same coef for token & query loss in both encoder & decoder')
    parser.add_argument('--cd_loss', default=1, type=float)
    parser.add_argument('--ucd_loss', default=1, type=float)
    parser.add_argument('--domain_enc_token_loss_coef', default=1, type=float)
    parser.add_argument('--domain_dec_token_loss_coef', default=1, type=float)
    parser.add_argument('--domain_enc_query_loss_coef', default=1, type=float)
    parser.add_argument('--domain_dec_query_loss_coef', default=1, type=float)
    parser.add_argument('--loss_cmt_geom', default=1, type=float)

    # model args
    parser.add_argument('--cd_threshold', type =float, default=0, help = 'threshold for CD loss')
    parser.add_argument('--cd_threshold_increament', type =float, default=0, help = 'increase_threshold for CD loss')

    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path, exist_ok=True)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

