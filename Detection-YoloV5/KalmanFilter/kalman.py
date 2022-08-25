import os
import sys
import argparse
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#from tracker import EuclideanDistTracker
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements,colorstr,increment_path, non_max_suppression, print_args, scale_coords,strip_optimizer,xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from KalmanFiltresi import KalmanFilter

kf = KalmanFilter()

@torch.no_grad()
def run( weights=ROOT / '/home/cal-display1/Documents/hdd2/Rabia/Yolov5/yolov5/runs/train/exp4/weights/best.pt', 
        source=ROOT / '/home/cal-display1/Downloads/Drone/drone222.mp4',  #file, 0 for webcam
        data=ROOT / '/home/cal-display1/Documents/hdd2/Rabia/Yolov5/yolov5/custom.yaml', #dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        iou_thres=0.45,  # NMS IOU threshold
        conf_thres=0.25,  # confidence threshold
        max_det=100,  # maximum detections per image
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
        project=ROOT / 'Results',  
        name='detect',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness 
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False ):
    source = str(source)
    
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (not is_file)
    if is_file:
        source = check_file(source)  # download

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    s3 = time_sync()
    device = select_device(device) #to get CUDA
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data) #.yaml ve pt.file yüklenme kısmı
    stride, names, pt= model.stride, model.names, model.pt

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader for webcam
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        batch_size = len(dataset)  
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        batch_size = 1  
    vid_path, vid_writer = [None] * batch_size, [None] * batch_size
    s4= time_sync()
    
    #Run inference
    model.warmup(imgsz=(1 if pt else batch_size, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  
            
        #inference time       
        s5= time_sync()
        pred_ = model(im, augment=augment, visualize=visualize)
        s6 = time_sync()
        pred = non_max_suppression(pred_,conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
          
         
        # Process predictions
        if len(pred):
            for i, det in enumerate(pred):  # per image
                seen += 1
                           
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                #s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                                
                if len(pred[0])>0:
              
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                      
                        # Write results
                        for *xyxy, cls in reversed(det):
                            if save_txt:  # Write to file                        
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                                line = (cls, *xywh) if save_conf else (cls, *xywh)
                                                            
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None #If we cannot write 'drone' on frame and annotator have label !!
                                #Kalman Filter
                                annotator.box_label(xyxy, color=colors(c, True)) 
                                                                                              
                                p1,p2 = annotator.p1, annotator.p2                     
                                p1, p2 =list(p1), list(p2) 
                                p1.extend(p2)  #combination  for lists                    
                                point1 = int(((p1[0])+(p1[2]))/2)
                                point2 = int(((p1[1])+(p1[3]))/2)                 
                                center = np.array([point1,point2])   
                                predicted = kf.predict(point1,point2)                         
                                center[0], center[1] = int(center[0]), int(center[1])
                                
                                for f in range(7):
                                    predicted = kf.predict(point1, point2)
                                    cv2.circle(im0, (point1,point2), 10, (0,255,0),4)
                                    cv2.circle(im0, (predicted[0],predicted[1]), 15, (0,0,255),4)
                                    if f ==0:
                                        print("origin    : ", point1,",", point2)
                                    else:
                                        print("predicted : ", predicted[0],",",predicted[1])
                                        
                else:                      
                    cv2.circle(im0, (predicted[0],predicted[1]), 35, (255,0,0),4)
                    print("Not Interrupted",predicted[0],predicted[1])                          
                                                                                                                                                                   
                s7 = time_sync()
                im0 = annotator.result()

                if view_img:
                    cv2.imshow(str(p),im0)
                    cv2.waitKey(1)  
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
                            else: 
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

                                
    #LOGGER.info(f'{s}Done. ({t3 - s5:.3f}s)')
    LOGGER.info(f'{s}Pre-Process time: ({s5 - s4:.3f}s)')
    LOGGER.info(f'{s}Inference time: ({s6 - s5:.3f}s)')
    LOGGER.info(f'{s}Post-Process time: ({s7 - s6:.3f}s)')       
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '/home/cal-display1/Documents/hdd2/Rabia/Yolov5/yolov5/runs/train/exp4/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '/home/cal-display1/Downloads/Drone/drone222.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / '/home/cal-display1/Documents/hdd2/Rabia/Yolov5/yolov5/custom.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    
