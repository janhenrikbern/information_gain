from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_stitcher import segment, stitch
import glob, os


def run(ground_truth_path, mat, gt_out_dir, permutations_out_dir):
    im = cv2.imread(ground_truth_path)
    data = segment(im)
    stitch(
        data["X"],
        permutations_out_dir,
        name=mat,
        targets=9,
        use_combinations=True,
        img_cnt=5,
    )
    if not os.path.exists(gt_out_dir + "/" + mat):
        os.mkdir(gt_out_dir + "/" + mat)
    for k, out in enumerate(data["y"]):
        print(out)
        cv2.imwrite(
            os.path.join(gt_out_dir + "/" + mat, f"{mat}-target-{k}.png"),
            out,
        )


if __name__ == "__main__":
    for img in glob.glob("/content/base/*.png"):
        mat = os.path.basename(img)
        run(img, mat=mat[:-4])
