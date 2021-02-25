from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from image_stitcher import segment, stitch

path = "test.png"

im = Image.open(path)
im = np.array(im)
data = segment(im)

stitch(data["X"], "test/", name=None, targets=4, use_combinations=True, img_cnt=5)
stitch(data["Y"], "truth/", name=None, targets=0, use_combinations=False, img_cnt=4)