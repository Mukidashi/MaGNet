import argparse
import os
import sys
import numpy as np
import cv2

import torch

import utils.utils as utils
from utils.losses import MagnetLoss
from models.MAGNET import MAGNET

from road_video_data import RoadVidData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_argparse():
    
    # Arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    # directory
    parser.add_argument('--out_dir', required=True, type=str)
    # parser.add_argument('--visible_gpus', required=True, type=str)
    parser.add_argument('--pose_dir', required=True, type=str)
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--calib_yaml', required=True, type=str)

    # output
    parser.add_argument('--output_dim', default=2, type=int)
    parser.add_argument('--output_type', default='G', type=str)
    parser.add_argument('--downsample_ratio', default=4, type=int)

    # DNET architecture
    parser.add_argument('--DNET_architecture', type=str, default='DenseDepth_BN', help='{DenseDepth_BN, DenseDepth_GN}')
    parser.add_argument("--DNET_fix_encoder_weights", type=str, default='None', help='None or AdaBins_fix')
    parser.add_argument("--DNET_ckpt", required=True, type=str)

    # FNET architecture
    parser.add_argument('--FNET_architecture', type=str, default='PSM-Net')
    parser.add_argument('--FNET_feature_dim', type=int, default=64)
    parser.add_argument("--FNET_ckpt", required=True, type=str)

    # Multi-view matching hyper-parameters
    parser.add_argument('--MAGNET_sampling_range', type=int, default=3)
    parser.add_argument('--MAGNET_num_samples', type=int, default=5)
    parser.add_argument('--MAGNET_mvs_weighting', type=str, default='CW5')
    parser.add_argument('--MAGNET_num_train_iter', type=int, default=3)
    parser.add_argument('--MAGNET_num_test_iter', type=int, default=3)
    parser.add_argument('--MAGNET_window_radius', type=int, default=10)
    parser.add_argument('--MAGNET_num_source_views', type=int, default=4)

    # dataset
    # parser.add_argument("--dataset_name", required=True, type=str, help="{kitti, scannet}")
    # parser.add_argument("--dataset_path", required=True, type=str, help="path to the dataset")
    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--dpv_height', type=int, help='input height', default=120)
    parser.add_argument('--dpv_width', type=int, help='input width', default=160)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

    # dataset - crop
    parser.add_argument('--do_kb_crop', default=True, help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--eigen_crop', default=False, help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', default=False, help='if set, crops according to Garg  ECCV16', action='store_true')

    # dataset - augmentation
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")

    # todo - best iter
    parser.add_argument("--MAGNET_ckpt", default='', type=str)

    return parser



if __name__ == "__main__":

    parser = set_argparse()

    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda:0')

    model = MAGNET(args).to(device)
    model = utils.load_checkpoint(args.MAGNET_ckpt, model)
    model.eval()

    dataset = RoadVidData(args)

    with torch.no_grad():
        for idx in range(dataset.__len__()):
            datas = dataset.get_data(idx)
            if datas is None:
                continue

            bname = os.path.basename(dataset.imgPathList[idx+1])
            bname, ext = os.path.splitext(bname)
            print("## Processing {0}".format(bname))
            
            ref_img = datas[0] 
            nghbr_imgs = datas[1]
            nghbr_poses = datas[2]
            is_valid = datas[3]
            cam_intrins = datas[4]

            pred_list = model(ref_img, nghbr_imgs, nghbr_poses, is_valid, cam_intrins, mode='test')

            print(len(pred_list))
            pred_dmap, pred_stdev = torch.split(pred_list[-1],1,dim=1)

            pred_dmap = pred_dmap.detach().cpu().permute(0,2,3,1).numpy().squeeze()
            pred_stdev = pred_stdev.detach().cpu().permute(0,2,3,1).numpy().squeeze()

            print(pred_dmap.shape, np.min(pred_dmap), np.max(pred_dmap))
            print(pred_stdev.shape, np.min(pred_stdev), np.max(pred_stdev))
            
            depImg = (np.clip(pred_dmap,0,65)*1000.0).astype(np.uint16)
            stdImg = np.clip(pred_stdev,0,255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out_dir, "{0}_dep.png".format(bname)), depImg)
            cv2.imwrite(os.path.join(args.out_dir, "{0}_stdev.png".format(bname)), stdImg)

