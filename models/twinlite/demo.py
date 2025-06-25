import torch
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from pathlib import Path
import cv2
import time
import numpy as np
import os
import shutil
from tqdm import tqdm

from model.model import TwinLiteNetPlus
from utils import val, netParams
from loss import TotalLoss
from demoDataset import LoadImages, LoadStreams

def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False, palette=None):
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)

    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] == 1] = [255, 0, 0]
    color_seg = color_area[..., ::-1]

    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return img

def detect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type != "cpu"

    model = TwinLiteNetPlus(args).to(device)
    if half:
        model.half()

    if args.source.isnumeric():
        cudnn.benchmark = True
        dataset = LoadStreams(args.source, img_size=args.img_size)
    else:
        dataset = LoadImages(args.source, img_size=args.img_size)

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)
    _ = model(img.half() if half else img)

    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    vid_path, vid_writer = None, None

    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = img.to(device).half() / 255.0 if half else img.to(device).float() / 255.0

        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_seg_out, ll_seg_out = model(img)

        save_path = str(args.save_dir + '/' + Path(path).stem + "_out.avi")

        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        img_vis = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _)

        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_vis)

        elif dataset.mode == 'video':
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = vid_cap.get(cv2.CAP_PROP_FPS) or 25.0
                h_vis, w_vis, _ = img_vis.shape
                vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w_vis, h_vis))

            if img_vis.dtype != np.uint8:
                img_vis = (img_vis * 255).astype(np.uint8)
            h_vis, w_vis, _ = img_vis.shape
            img_vis = cv2.resize(img_vis, (w_vis, h_vis))
            vid_writer.write(img_vis)

        else:
            cv2.imshow('image', img_vis)
            cv2.waitKey(1)

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    print(f"\n✅ Gotowe! Wideo zapisane w: {save_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='pretrained/large.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos/video.mp4', help='source file/folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--config', type=str, choices=["nano", "small", "medium", "large"], help='Model configuration')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt)
