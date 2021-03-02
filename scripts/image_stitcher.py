import os
import cv2
import numpy as np
import argparse
from itertools import combinations
import time

DEBUG = False
IMG_CNT = 1

def get_combinations(n, img_cnt, unfiltered=False):
    combos = combinations(range(n), img_cnt)
    out = []
    last = set()
    for c in combos:
        cur = set(c)
        if len(last.intersection(cur)) <= round(img_cnt/2.):
            out.append(c)
            last = cur
    return list(combos) if unfiltered else out


def stitch(imgs, out_dir, name=None, targets=4, use_combinations=False, img_cnt=IMG_CNT):
    if len(imgs) < img_cnt:
        return

    combos = [range(img_cnt)]
    if use_combinations:
        combos = get_combinations(len(imgs), img_cnt)
        if DEBUG:
            print(combos)

    h, w, c = imgs[0].shape
    cnt = name if name else int(round(time.time() * 1000))
    for k, combo in enumerate(combos):
        out = np.full((h, w * (img_cnt + targets), c), (0, 0, 0), dtype=np.uint8)

        for i, idx in enumerate(combo):
            out[0:h, i * w : (i + 1) * w] = imgs[idx]

        cv2.imwrite(
            os.path.join(out_dir, f"{cnt}_{k+1}.png"),
            out,
        )

def segment(img, truth_cnt=4, img_size=256):
    out = []
    h, w, c = img.shape
    cnt = w // img_size
    for i in range(cnt):
        offset = i * img_size
        out.append(img[:, offset : offset + img_size, :])

    return {"X": out[:-truth_cnt], "y": out[-truth_cnt:]}



def main(in_path, out_path, n_targets, use_combos, n_picks):
    queue = [in_path]
    cnt = 0
    while queue:
        in_pwd = queue.pop(0)
        imgs = []
        paths = list(os.listdir(in_pwd))
        paths.sort()
        sample = in_pwd.split("/")[-1]
        picks_cnt = 0
        for f in paths:
            if f.endswith("DS_Store"):
                continue
            f_path = os.path.join(in_pwd, f)
            if os.path.isdir(f_path):
                queue.append(f_path)
            elif os.path.isfile(f_path) and (picks_cnt < n_picks or a.parse_folder):
                picks_cnt += 1
                imgs.append(cv2.imread(f_path))
        cnt += 1
        if a.parse_folder:
            for ix, img in enumerate(imgs):
                stitch(
                    [img], out_path, 
                    name=f"{sample}_{ix}", 
                    targets=n_targets, 
                    use_combinations=use_combos,
                    img_cnt=n_picks
                )
        else:
            stitch(
                imgs, out_path, 
                name=f"{sample}_{cnt}", 
                targets=n_targets, 
                use_combinations=use_combos,
                img_cnt=n_picks
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True, help="path to folder containing images"
    )
    parser.add_argument("--output_dir", required=True, help="Output directory of raw images")
    parser.add_argument(
        "--n_targets", default=4, help="Number of target blanks", type=int
    )
    parser.add_argument(
        "--use_combos",
        default=False,
        help="Generate a stitched version of all possible combos",
        type=bool,
    )
    parser.add_argument(
        "--n_picks",
        default=IMG_CNT,
        help="Generate a stitched version of all possible combos",
        type=int,
    )
    parser.add_argument(
        "--parse_folder",
        default=IMG_CNT,
        help="Generate a stitched version of all possible combos",
        type=bool,
    )
    a = parser.parse_args()
    print("Running image stitcher")
    main(a.input_dir, a.output_dir, a.n_targets, a.use_combos, a.n_picks)
