"""
CODE from https://github.com/mworchel/svbrdf-estimation

ADD CREDITS / REFERENCE BEFORE CODE RELEASE!!!
"""


import math
import numpy as np
from PIL import Image
import random
import torch
import matplotlib.pyplot as plt


def plot_imgs(name, imgs, n_rows, n_cols, row_major=True, permute=False):
        fig = plt.figure(figsize=(n_cols, n_rows))
        grid = fig.add_gridspec(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):

                if row_major:
                    img = imgs[i, j]
                else:
                    img = imgs[j, i]
                if permute:
                    img = img.permute(1,2,0)
                
                fig.add_subplot(grid[i, j])
                plt.axis("off")
                plt.imshow(img)

        plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")



def enable_deterministic_random_engine(seed=313):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def crop_square(tensor, anchor, size):
    num_dimensions = len(tensor.shape)
    if num_dimensions == 3:
        return tensor[:, anchor[0] : anchor[0] + size, anchor[1] : anchor[1] + size]
    elif num_dimensions == 4:
        if len(anchor.shape) == 1:  # Only one anchor for all images
            return tensor[
                :, :, anchor[0] : anchor[0] + size, anchor[1] : anchor[1] + size
            ]
        elif (
            len(anchor.shape) == 2
        ):  # One anchor for each image (handle cropping individually)
            images = torch.split(tensor, 1, dim=0)
            for i in range(len(images)):
                images[i] = crop_square(images[i], anchor[i], size)
            return torch.cat(images, dim=0)
    else:
        raise Exception("Cannot crop tensor of dimension {:d}".format(num_dimensions))


def gamma_decode(images):
    return torch.pow(images, 2.2)


def gamma_encode(images):
    return torch.pow(images, 1.0 / 2.2)


def pack_svbrdf(normals, diffuse, roughness, specular):
    # We concat on the feature dimension. Here negative in order to handle batches intrinsically-
    return torch.cat([normals, diffuse, roughness, specular], dim=-3)


def unpack_svbrdf(svbrdf, is_encoded=False):
    svbrdf_parts = svbrdf.split(1, dim=-3)

    normals = None
    diffuse = None
    roughness = None
    specular = None
    if not is_encoded:
        normals = torch.cat(svbrdf_parts[0:3], dim=-3)
        diffuse = torch.cat(svbrdf_parts[3:6], dim=-3)
        roughness = torch.cat(svbrdf_parts[6:9], dim=-3)
        specular = torch.cat(svbrdf_parts[9:12], dim=-3)
    else:
        normals = torch.cat(svbrdf_parts[0:2], dim=-3)
        diffuse = torch.cat(svbrdf_parts[2:5], dim=-3)
        roughness = torch.cat(svbrdf_parts[5:6], dim=-3)
        specular = torch.cat(svbrdf_parts[6:9], dim=-3)

    return normals, diffuse, roughness, specular


# We don't really need the encoding...maybe only for testing
# Assumes SVBRDF channels are in range [-1, 1]
def encode_svbrdf(svbrdf):
    raise NotImplementedError(
        "This function does not currently work. The normal encoding is bugged (normal vector is not converted to [x, y, 1] before slicing)"
    )

    normals, diffuse, roughness, specular = unpack_svbrdf(svbrdf, False)

    roughness = roughness.split(1, dim=-3)[
        0
    ]  # Only retain one channel (roughness if grayscale anyway)
    normals = torch.cat(
        normals.split(1, dim=-3)[:2]
    )  # Only retain x and y coordinates of the normal

    return pack_svbrdf(normals, diffuse, roughness, specular)


# Assumes SVBRDF channels are in range [-1, 1]
def decode_svbrdf(svbrdf):
    normals_xy, diffuse, roughness, specular = unpack_svbrdf(svbrdf, True)

    # Repeat roughness channel three times
    # The weird syntax is due to uniform handling of batches of SVBRDFs and single SVBRDFs
    roughness_repetition = [1] * len(diffuse.shape)
    roughness_repetition[-3] = 3
    roughness = roughness.repeat(roughness_repetition)

    normals_x, normals_y = torch.split(normals_xy.mul(3.0), 1, dim=-3)
    normals_z = torch.ones_like(normals_x)
    normals = torch.cat([normals_x, normals_y, normals_z], dim=-3)
    norm = torch.sqrt(torch.sum(torch.pow(normals, 2.0), dim=-3, keepdim=True))
    normals = torch.div(normals, norm)

    return pack_svbrdf(normals, diffuse, roughness, specular)


# Transforms range [-1, 1] to [0, 1]
# Corresponds to helpers.deprocess() in the reference code
def encode_as_unit_interval(tensor):
    return (tensor + 1.) / 2.


# Transforms range [0, 1] to [-1, 1]
# Corresponds to helpers.preprocess() in the reference code
def decode_from_unit_interval(tensor):
    return tensor * 2. - 1.


def generate_normalized_random_direction(count, min_eps=0.001, max_eps=0.05):
    r1 = torch.Tensor(count, 1).uniform_(0.0 + min_eps, 1.0 - max_eps)
    r2 = torch.Tensor(count, 1).uniform_(0.0, 1.0)

    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2

    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - r ** 2)

    return torch.cat([x, y, z], axis=-1)

def generate_random_scenes(count):
    # Randomly distribute both, view and light positions
    view_positions  = generate_normalized_random_direction(count, 0.001, 0.1) # shape = [count, 3]
    light_positions = generate_normalized_random_direction(count, 0.001, 0.1)
    light_intensity = torch.ones_like(light_positions) * 20.
    return view_positions, light_positions, light_intensity


def generate_specular_scenes(count):
    # Only randomly distribute view positions and place lights in a perfect mirror configuration
    view_positions  = generate_normalized_random_direction(count, 0.001, 0.1) # shape = [count, 3]
    light_positions = view_positions * torch.Tensor([-1.0, -1.0, 1.0]).unsqueeze(0)

    # Reference: "parameters chosen empirically to have a nice distance from a -1;1 surface.""
    distance_view  = torch.exp(torch.Tensor(count, 1).normal_(mean=0.5, std=0.75)) 
    distance_light = torch.exp(torch.Tensor(count, 1).normal_(mean=0.5, std=0.75))

    # Reference: "Shift position to have highlight elsewhere than in the center."
    # NOTE: This code only creates guaranteed specular highlights in the orthographic rendering, not in the perspective one.
    #       This is because the camera is -looking- at the center of the patch.
    shift = torch.cat([torch.Tensor(count, 2).uniform_(-1.0, 1.0), torch.zeros((count, 1)) + 0.0001], dim=-1)

    view_positions  = view_positions  * distance_view  + shift
    light_positions = light_positions * distance_light + shift
    light_intensity = torch.ones_like(light_positions) * 50.

    return view_positions, light_positions, light_intensity
