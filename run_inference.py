from argparse import ArgumentParser
import os
import os.path as osp
import time

import cv2
import gdown
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from face_parsing.model import BiSeNet


weight_dic = {'seg.pth': 'https://drive.google.com/file/d/1lIKvQaFKHT5zC7uS4p17O9ZpfwmwlS62/view?usp=sharing'}

"""
'1: face skin, 2: nose, 3: eyeglasses, 4: eyes, 5: eyebrows,'
'6: ears, 7: teeth, 8: top lip, 9: bottom lip, 10: hair, '
'11: hat, 12: earrings'
"""


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--imgs_dir', type=str)
    parser.add_argument('--save_dir', type=str)

    return parser.parse_args()


def download_weight(weight_path):
    if osp.isfile(weight_path):
        return
    gdown.download(weight_dic[osp.basename(weight_path)],
                   output=weight_path, fuzzy=True)


def load_image(fp, num_retries=10):
    retry_n = 0
    while retry_n < num_retries:
        img = cv2.imread(fp)
        if img is not None:
            break
        time.sleep(2)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def array2tensor(img, device):
    img = torch.from_numpy(img) / 255.0
    img = img.permute(2, 0, 1)[None]
    img = img.to(device=device, dtype=torch.float32)
    return img


class Inferencer(nn.Module):
    def __init__(self, segm_ckpt_fp, device, target_size=1024):
        super().__init__()

        self.segm_ckpt_fp = segm_ckpt_fp
        self.device = device
        self.target_size = target_size

        self.segm_model = None
        self._load_segmentation_model()

    def _load_segmentation_model(self):
        self.segm_model = BiSeNet(n_classes=16)

        if not osp.exists(self.segm_ckpt_fp):
            os.makedirs(osp.dirname(self.segm_ckpt_fp), exist_ok=True)
            download_weight(self.segm_ckpt_fp)
        self.segm_model.load_state_dict(torch.load(self.segm_ckpt_fp, map_location='cpu'))
        for param in self.segm_model.parameters():
            param.requires_grad = False
        self.segm_model.eval()
        self.segm_model.to(device=self.device)

    @torch.no_grad()
    def predict_segmentation(self, img):
        img = img.clamp(0.0, 1.0)
        face_seg_logits = self.segm_model(img)
        face_seg = torch.argmax(face_seg_logits, dim=1, keepdim=False).long()
        return face_seg

    def single_image_inference(self, img_path):
        img = load_image(img_path)
        img = array2tensor(img, self.device)
        segm = self.predict_segmentation(img)
        segm = segm[0].cpu().numpy().astype(np.uint8)
        h, w = segm.shape
        if h != self.target_size or w != self.target_size:
            segm = cv2.resize(segm, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        return segm

    def inference_on_dir(self, imgs_dp, save_dp):
        os.makedirs(save_dp, exist_ok=True)
        for name in tqdm(os.listdir(imgs_dp)):
            if name.startswith('.'):
                continue
            img_fp = osp.join(imgs_dp, name)
            segm = self.single_image_inference(img_fp)
            new_name = osp.splitext(name)[0] + '.png'
            cv2.imwrite(osp.join(save_dp, new_name), segm)


if __name__ == '__main__':
    args = parse_args()
    segm_generator = Inferencer('./pretrained_models/seg.pth', device='cuda:0', target_size=512)
    segm_generator.inference_on_dir(args.imgs_dir, args.save_dir)
