from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from image_stitcher import segment, stitch

def run(ground_truth_path, gt_out_dir="/content/gt/", permutations_out_dir="/content/permutations/"):
    im = Image.open(ground_truth_path)
    im = np.array(im)
    data = segment(im)

    stitch(data["X"], permutations_out_dir, name=None, targets=4, use_combinations=True, img_cnt=5)    
    for k, out in enumerate(data["y"]):
        cv2.imwrite(
                os.path.join(gt_out_dir, f"target-{k}.png"),
                out,
            )


if __name__ == "__main__":
    for img in glob.glob("/content/gt/*.png"):
        run(img)