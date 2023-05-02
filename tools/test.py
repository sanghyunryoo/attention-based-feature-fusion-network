import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from utils2 import *
import open3d


import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


from utils2 import *



def render_lidar_with_boxes(objects, calib):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)
    gt = []
    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)
        # # Open3d boxes
        boxes3d_pts = open3d.utility.Vector3dVector(boxes3d_pts.T)       
        box = open3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        tan = box.R[0][1] / box.R[0][0]
        yaw = np.arctan(tan)

        gt.append([box.center[0], box.center[1], box.center[2], box.extent[0], box.extent[1], box.extent[2], yaw])
    return gt
   

def parse_config():
    parser3D = argparse.ArgumentParser(description='arg parser')
    # parser2D = argparse.ArgumentParser()

    parser3D.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser3D.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser3D.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser3D.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser3D.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser3D.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser3D.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser3D.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser3D.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser3D.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser3D.add_argument('--start_epoch', type=int, default=0, help='')
    parser3D.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser3D.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser3D.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser3D.add_argument('--save_to_file', action='store_true', default=False, help='')






    # Yolo
    parser3D.add_argument('--weights', nargs='+', type=str, default=ROOT / '/home/cocel/OpenPCDet/yolov5/runs/train/results5/weights/best.pt', help='model path or triton URL')
    parser3D.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser3D.add_argument('--data', type=str, default=ROOT / 'data/data.yaml', help='(optional) dataset.yaml path')
    parser3D.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser3D.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser3D.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser3D.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser3D.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser3D.add_argument('--view-img', action='store_true', help='show results')
    parser3D.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser3D.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser3D.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser3D.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser3D.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser3D.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser3D.add_argument('--augment', action='store_true', help='augmented inference')
    parser3D.add_argument('--visualize', action='store_true', help='visualize features')
    parser3D.add_argument('--update', action='store_true', help='update all models')
    parser3D.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser3D.add_argument('--name', default='exp', help='save results to project/name')
    parser3D.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser3D.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser3D.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser3D.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser3D.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser3D.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser3D.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')



    args = parser3D.parse_args()

    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand


    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    
    # GT Label Call-----------------------------------------------------------------------------------------------------
    # label_dir = '/home/yolo/OpenPCDet/data/kitti/training/label_2'

    # label_list = os.listdir(label_dir)
    # label_list.sort()
    # label_list = label_list[2000:3000]
    # gt = []
    # for i in range(len(label_list)):
    #     name = label_dir + '/' + label_list[i]
    #     gt.append(np.loadtxt(name, delimiter=' ', usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]))
    #         # gt.append(np.genfromtxt(name, delimiter=',', 
    #         #     dtype=None, names=('Car', 'Pedestrian', 'Truck', 'Cyclist', 'Misc', 'Van', 'Tram', 'Person_sitting', 'DontCare')))



    # Load calibration
    # calib_dir = '/home/yolo/OpenPCDet/data/kitti/training/calib/'
    # label_dir = '/home/yolo/OpenPCDet/data/kitti/training/label_2/'


    # calib_list = os.listdir(calib_dir)
    # calib_list.sort()
    # calib_list = calib_list[3711:]

    # label_list = os.listdir(label_dir)5
    # label_list.sort()
    # label_list = label_list[3711:]

        

    calib_dir = '/home/cocel/OpenPCDet/data/kitti/training/calib/'
    label_dir = '/home/cocel/OpenPCDet/data/kitti/training/label_2/'
    val_list_path = '/home/cocel/OpenPCDet/data/kitti/ImageSets/val.txt'

    calib_list_a = os.listdir(calib_dir)
    calib_list_a.sort()

    label_list_a = os.listdir(label_dir)
    label_list_a.sort()

    calib_list =[]
    label_list =[]
    with open(val_list_path,'r') as val_file:
        while True:
            line = val_file.readline()   
            if not line:
                break
            val_idx = int(line.strip())
            calib_list.append(calib_list_a[val_idx])
            label_list.append(label_list_a[val_idx])




    calibs = []
    labels = []
    gt_boxes = []
    for i in range(len(label_list)):
        calib_name = calib_dir + '/' + calib_list[i]
        label_name = label_dir + '/' + label_list[i]

        calib = read_calib_file(calib_name)
        label = load_label(label_name)

        gt_boxes.append(render_lidar_with_boxes(label, calib))
 

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()