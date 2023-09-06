import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

from run_inference import load_image


IMAGES_DIR = '/mnt/hdd/datasets/beauty_test_set'
SEGM_DIR = '/mnt/hdd/datasets/beauty_test_set-bisenet_masks'
SAVE_DIR = 'debug'
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_ID = 2


def main():
    for name in tqdm(os.listdir(IMAGES_DIR)):
        if name.startswith('.'):
            continue
        mask_name = osp.splitext(name)[0] + '.png'
        mask_fp = osp.join(SEGM_DIR, mask_name)

        img = load_image(osp.join(IMAGES_DIR, name))
        mask = cv2.imread(mask_fp)

        h, w = img.shape[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        bin_mask = (mask == CLASS_ID)

        img_masked = np.where(bin_mask, np.ones_like(img) * 255, img)

        save_fp = osp.join(SAVE_DIR, name)
        cv2.imwrite(save_fp, cv2.cvtColor(img_masked, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()

