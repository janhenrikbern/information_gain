import numpy as np
import matplotlib.pyplot as plt
from pytorch_svbrdf_renderer import Renderer, Config
import torch


if __name__ == "__main__":
    N_VIEWS = 3
    IS_COLLOCATED = True
    UPDATE_METHOD = 0
    OFFSET_DIST = torch.Tensor([0.1, 0.1, 0])

    epochs = 3
    lr = 0.1
