from __future__ import division, print_function

import os
import argparse
import numpy as np
import tensorflow as tf

from WCT.utils import preserve_colors_np
from WCT.utils import get_files, get_img, get_img_crop, save_img, resize_to, center_crop
import scipy
import time
from WCT.wct import WCT

parser = argparse.ArgumentParser()

# parser.add_argument('--vgg-path', type=str, help='Path to vgg_normalised.t7', default='models/vgg_normalised.t7')
# parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
# parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')
# parser.add_argument('--mask-path', type=str, dest='mask_path', help='mask image or folder of images', default=None)
# parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')

# parser.add_argument('--device', type=str, help='Device to perform compute on, e.g. /gpu:0', default='/gpu:0')
# parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512",
#                     default=0)
# parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
# parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=0)
# parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=0)
# parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)
# parser.add_argument('-r', '--random', type=int, help="Choose # of random subset of images from style folder", default=0)
# parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
# parser.add_argum
# args = parser.parse_args()

import cv2
import numpy as np


def getMask(mask_path):
    # opencv loads the image in BGR, convert it to RGB
    img = cv2.cvtColor(cv2.imread(mask_path),
                       cv2.COLOR_BGR2RGB)
    lower_white = np.array([220, 220, 220], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img, lower_white, upper_white)  # could also use threshold
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
    3, 3)))  # "erase" the small white points in the resulting mask
    mask = cv2.bitwise_not(mask)  # invert mask

    # load background (could be an image too)
    bk = np.full(img.shape, 255, dtype=np.uint8)  # white bk

    # get masked foreground
    fg_masked = cv2.bitwise_and(img, img, mask=mask)

    # get masked background, mask must be inverted
    mask = cv2.bitwise_not(mask)
    bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

    # combine masked foreground and masked background
    final = cv2.bitwise_or(fg_masked, bk_masked)
    mask = cv2.bitwise_not(mask)  # revert mask to original
    cv2.imwrite("./tmp.png", cv2.bitwise_not(bk_masked))
    return


def removeBG(mask_path, stylized_img_path, original_img_path, target_path="outputs/"):
    h, w, _ = cv2.imread(mask_path).shape
    img = np.clip(cv2.resize(stylized_img_path, (w, h)), 0, 255).astype(np.uint8)
    getMask(mask_path)
    mask = cv2.resize(cv2.imread(mask_path, 0), (w, h))
    img_1 = cv2.resize(cv2.imread(original_img_path), (w, h))

    invermask = cv2.bitwise_not(mask)

    res = cv2.bitwise_and(img, img, mask=mask)
    res2 = cv2.bitwise_and(img_1, img_1, mask=invermask)
    cv2.imwrite('WCT/outputs/mask1_0resize.jpg', res + res2)
    return


def main():
    print('in main')
    start = time.time()

    checkpoints=["WCT/models/relu5_1", "WCT/models/relu4_1", "WCT/models/relu3_1", "WCT/models/relu2_1","WCT/models/relu1_1"]
    relutargets=["relu5_1", "relu4_1", "relu3_1", "relu2_1" ,"relu1_1"]
    style_size=512
    vgg='WCT/models/vgg_normalised.t7'
    alpha =0.6
    style_path="WCT/styles"
    content="WCT/masks"
    out="WCT/outputs"
    masks="WCT/masks/mask1.jpg"
    cropsize=0
    # Load the WCT model
    concat=False
    adain=False
    
    swap5=False
    ss_alpha=0.6
    ss_patch_size=3
    ss_stride=1
    ## Style swap args
    device="/gpu:0"
    keepcolors=False
    content_size=0
    passes=1
    
    wct_model = WCT(checkpoints=checkpoints,
                    relu_targets=relutargets,
                    vgg_path=vgg,
                    device=device,
                    ss_patch_size=ss_patch_size,
                    ss_stride=ss_stride)

    # Get content & style full paths
    if os.path.isdir(content):
        content_files = get_files(content)
    else:  # Single image file
        content_files = [content]
    if os.path.isdir(style_path):
        style_files = get_files(style_path)
        if 0 > 0:
            style_files = np.random.choice(style_files, random)
    else:  # Single image file
        style_files = [style_path]

    os.makedirs(out, exist_ok=True)

    count = 0

    ### Apply each style to each content image
    for content_fullpath in content_files:
        content_prefix, content_ext = os.path.splitext(content_fullpath)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext

        content_img = get_img(content_fullpath)
        if content_size > 0:
            content_img = resize_to(content_img, content_size)

        for style_fullpath in style_files:
            style_prefix, _ = os.path.splitext(style_fullpath)
            style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

            # style_img = get_img_crop(style_fullpath, resize=args.style_size, crop=args.crop_size)
            # style_img = resize_to(get_img(style_fullpath), content_img.shape[0])
            print("for loop")
            try:

                style_img = get_img(style_fullpath)
                st = style_img
                if style_size > 0:
                    print("style")
                    style_img = resize_to(style_img, style_size)
                    print(style_img)
                    cv2.imshow(style_img.shape)
                    cv2.waitKey(0)
                if crop_size > 0:
                    print("crop")
                    style_img = center_crop(style_img, crop_size)

                if keep_colors:
                    print("color")
                    style_img = preserve_colors_np(style_img, content_img)
            except:
                print("except")
                # continue
            # if args.noise:  # Generate textures from noise instead of images
            #     frame_resize = np.random.randint(0, 256, frame_resize.shape, np.uint8)
            #     frame_resize = gaussian_filter(frame_resize, sigma=0.5)

            # Run the frame through the style network
            print("help me out")
            print(content_img.shape)
            print(style_img.shape)
            stylized_rgb = wct_model.predict(content_img, style_img, alpha, swap5, ss_alpha, adain)

            if passes > 1:
                print("help me out 2")
                for _ in range(passes - 1):
                    stylized_rgb = wct_model.predict(stylized_rgb, style_img, alpha, swap5, ss_alpha,
                                                     adain)

            # Stitch the style + stylized output together, but only if there's one style image
            if concat:
                # Resize style img to same height as frame
                style_img_resized = scipy.misc.imresize(style_img, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                # margin = np.ones((style_img_resized.shape[0], 10, 3)) * 255
                stylized_rgb = np.hstack([style_img_resized, stylized_rgb])

            # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
            out_f = os.path.join("WCT/outputs", '{}_{}{}'.format(content_prefix, style_prefix, content_ext))
            # out_f = f'{content_prefix}_{style_prefix}.{content_ext}'

            removeBG(masks, stylized_rgb, masks, out_f)
            print(stylized_rgb.shape)
            save_img(out_f, stylized_rgb)
            count += 1
            print("{}: Wrote stylized output image to {}".format(count, out_f))

    print("Finished stylizing {} outputs in {}s".format(count, time.time() - start))


if __name__ == '__main__':
    main()

# !python WCT/stylize.py --checkpoints  WCT/models/relu5_1 WCT/models/relu4_1 WCT/models/relu3_1 WCT/models/relu2_1 WCT/models/relu1_1 --relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1 --style-size 512 --alpha 0.6 --style-path "WCT/styles/"  --content-path "WCT/masks/"  --out-path "WCT/outputs/" --mask-path "WCT/masks/mask1.jpg"