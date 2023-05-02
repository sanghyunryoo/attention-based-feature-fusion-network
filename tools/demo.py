name = '000241'
threshold = 13
import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from utils2 import *
import math
import matplotlib.pyplot as plt

def rotationMatrixToEulerAngles(R) :
 
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

# def render_lidar_with_boxes(objects, calib):
#     # Projection matrix
#     proj_cam2_2_velo = project_cam2_to_velo(calib)
#     gt = []
#     # Draw objects on lidar
#     for obj in objects:        
#         if obj.type == 'Car':
#             # Project boxes from camera to lidar coordinate
#             boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)
#             # # Open3d boxes
#             boxes3d_pts = open3d.utility.Vector3dVector(boxes3d_pts.T)       
#             box = open3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)    
#             print(box)  
#             angle = np.arccos((np.trace(box.R)-1)/2)
#             tan = box.R[1][0] / box.R[0][0]
#             yaw = np.arctan(tan)
            
#             gt.append([box.center[0], box.center[1]/10, box.center[2], box.extent[0], box.extent[1], box.extent[2], angle])        
#         else:
#             continue

#     return gt

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/voxel_rcnn_3classes.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/cocel/OpenPCDet/data/kitti/testing/velodyne/' + name + '.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/home/cocel/OpenPCDet/output/kitti_models/voxel_rcnn_3classes/default/ckpt_folder/original/checkpoint_epoch_79.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--name', type=str, default='000000',
                        help='specify the point cloud data file name')   
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def th_delete(tensor, indices):
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

import sys
import os

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

import platform
from utils2 import *
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 cutout, letterbox, mixup, random_perspective)

def run(
        frame_id,
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/train/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride

        cfg_file=None,  
        batch_size=None, 
        workers=4,  
        extra_tag='default',  
        ckpt=None,
        launcher='none', 
        tcp_port=18888, 
        local_rank=0,  
        set_cfgs=None, 
        max_waiting_mins=30,  
        start_epoch=0,  
        eval_tag='default',  
        eval_all=False, 
        ckpt_dir=None,  
        save_to_file=False, 
):
    source = str('/home/cocel/OpenPCDet/data/kitti/testing/image_2/' + str(frame_id) + '.png')
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend('/home/cocel/OpenPCDet/yolov5/yolov5s.pt', device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                bbox = []
                for *xyxy, conf, cls in reversed(det):
                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())
                    # if cls == 0:
                    bbox.append([x1, y1, x2, y2])
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                return bbox
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

from geometry_utils import *

def render_lidar_on_image(pts_velo, calib, img_width, img_height, bbox):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    depth_list = []
    avg_depth = []

    for j, detection in enumerate(bbox):
        depth_list_per_detection = []
        x1 = detection[0]
        x2 = detection[2]
        y1 = detection[1]
        y2 = detection[3]         
        w = x2 - x1
        h = y2 - y1
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        for i in range(imgfov_pc_pixel.shape[1]):
            # for j, detection in enumerate(bbox):
            if x_center - w/4 < int(np.round(imgfov_pc_pixel[0, i])) <x_center + w/4 and y_center - h/4 < int(np.round(imgfov_pc_pixel[1, i])) < y_center + h/4 :
                depth = imgfov_pc_cam2[2, i]
                depth_list_per_detection.append(depth)
        try:
            avg_depth.append(sum(depth_list_per_detection)/len(depth_list_per_detection))
        except:
            pass
    # print(avg_depth)


    pedestrian_depth = np.array(avg_depth)
    
    # Get intrinsic parameters
    K = intrinsic_from_fov(img_height, img_width, 90)  # +- 45 degrees
    K_inv = np.linalg.inv(K)


    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    i = 0
    # Loop through each pixel in the image

    cam_coord = []
    for i, detection in enumerate(bbox):

        try:
            x1 = detection[0]
            x2 = detection[2]
            y1 = detection[1]
            y2 = detection[3]        
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)     

            # Apply equation in fig 3
            x = (x_center - u0) * avg_depth[i] / fx
            y = (y_center - v0) * avg_depth[i] / fy
            z = avg_depth[i]
            cam_coord.append([x, y, z])
        except:
            pass


    lidar = project_camera_to_lidar(np.array(cam_coord).T, proj_cam2_2_velo)

    return lidar.T

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)


            pc_velo = load_velo_scan('/home/cocel/OpenPCDet/data/kitti/testing/velodyne/' + name + '.bin')[:, :3]
            calib = read_calib_file('/home/cocel/OpenPCDet/data/kitti/testing/calib/' + name + '.txt')
            images_load = load_image('/home/cocel/OpenPCDet/data/kitti/testing/image_2/' + name + '.png')
            images = name
            bbox = (run(images))
            lidar = torch.from_numpy(render_lidar_on_image(pc_velo, calib, images_load.shape[1], images_load.shape[0], bbox)).cuda()

            idx = []
            real_car_idx = []
            real_ped_idx = []
            real_cycle_idx = []
            
            for pedestrain in lidar:
                for i in range(len(pred_dicts[0]['pred_labels'])):

                    if pred_dicts[0]['pred_labels'][i] == 1:
                        distance = torch.sum((pred_dicts[0]['pred_boxes'][i][:3] - pedestrain) ** 2) ** 0.5
                        idx.append(i)
                        if distance < threshold:
                            real_car_idx.append(i)
                        else:
                            pass
                    elif pred_dicts[0]['pred_labels'][i] == 2:
                        distance = torch.sum((pred_dicts[0]['pred_boxes'][i][:3] - pedestrain) ** 2) ** 0.5
                        idx.append(i)
                        if distance < threshold:
                            real_ped_idx.append(i)
                        else:
                            pass
                    elif pred_dicts[0]['pred_labels'][i] == 3:
                        distance = torch.sum((pred_dicts[0]['pred_boxes'][i][:3] - pedestrain) ** 2) ** 0.5
                        idx.append(i)
                        if distance < threshold:
                            real_cycle_idx.append(i)
                        else:
                            pass                            
            idx = list(set(idx))
            real_car_idx = list(set(real_car_idx))
            real_ped_idx = list(set(real_ped_idx))
            real_cycle_idx = list(set(real_cycle_idx))




            before_late_fusion = pred_dicts[0]['pred_boxes']
            for i in range(len(real_car_idx)):
                idx.remove(real_car_idx[i])
            for i in range(len(real_ped_idx)):
                idx.remove(real_ped_idx[i])
            for i in range(len(real_cycle_idx)):
                idx.remove(real_cycle_idx[i])
            pred_dicts[0]['pred_labels'] = th_delete(pred_dicts[0]['pred_labels'], idx)
            pred_dicts[0]['pred_boxes'] = th_delete(pred_dicts[0]['pred_boxes'], idx)



            # GT
            # labels = load_label('/home/cocel/OpenPCDet/data/kitti/testing/label_2/' + name + '.txt')
            # gt = render_lidar_with_boxes(labels, calib)
            # gt = np.array(gt)
            # gt = torch.from_numpy(gt)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes = None, ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
