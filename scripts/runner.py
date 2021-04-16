from git import Repo
import wget
import os
import shutil
import glob
import generate_permutations
import setup_deschaintre
import zipfile
from pytorch_ssim import ssim
import numpy as np
import matplotlib.pyplot as plt

# Global variables that can be changed to organize directory structure.
# Important to have test directory in permutations for placing the view combinations
GLOBAL_DIR = "/Users/abhinav/Research/infogain"
GT_DIR = GLOBAL_DIR + "/gt"
PERMUTATIONS_DIR = GLOBAL_DIR + "/permutations"
PERMUTATIONS_TEST_DIR = GLOBAL_DIR + "/permutations/test"
BASE_DIR = GLOBAL_DIR + "/base"
GLOBAL_IMG_NUMBER = 3


def clone_repo(url, dir):
    """
    Simple helper to clone repos easily
    """
    Repo.clone_from(url, dir)


def imports(dir):
    """
    Imports necessary repos (pytorch-ssim) and SVBRDF acquisition (deschaintre)
    """
    if not os.path.exists(dir + "/multi-image-deepNet-SVBRDF-acquisition"):
        clone_repo(
            "https://github.com/valentin-deschaintre/multi-image-deepNet-SVBRDF-acquisition.git",
            dir + "/multi-image-deepNet-SVBRDF-acquisition",
        )
    if not os.path.exists(dir + "/pytorch-ssim"):
        clone_repo(
            "https://github.com/Po-Hsun-Su/pytorch-ssim.git", dir + "/pytorch-ssim"
        )
    # wget.download("https://repo-sam.inria.fr/fungraph/multi_image_materials/supplemental_multi_images/checkpointTrained.zip", bar=bar_thermometer)


def make_dirs():
    """
    Helper function to make directories of associated ground truth values,
    permutations of view combinations, and a base directory to store test images
    """
    if not os.path.exists(GT_DIR):
        os.mkdir(GT_DIR)
    if not os.path.exists(PERMUTATIONS_DIR):
        os.mkdir(PERMUTATIONS_DIR)
    if not os.path.exists(PERMUTATIONS_TEST_DIR):
        os.mkdir(PERMUTATIONS_TEST_DIR)
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)


def assemble_targets():
    """
    Correctly organizes the ground truth SVBRDF maps for each material
      Parameters:
        none

      Returns:
        target_paths: array of all the target SVBRDF ground truth values for each material
    """

    target_endings = ["target-0.png", "target-1.png", "target-2.png"]
    target_paths = []
    rootdir_glob = GT_DIR + "/**/*"
    target_paths = [
        f
        for f in glob.iglob(rootdir_glob, recursive=True)
        if f[-5:] == "0.png" or f[-5:] == "1.png" or f[-5:] == "2.png"
    ]

    return target_paths


def load_data(target_paths):
    """
    Loads SVBRDF predictions into output and associated naming conventions for
    tracking of the different view combinations
      Parameters:
        target_paths: array of all the target ground truth SVBRDF maps for each material
        (used to help with naming conventions for view combinations)

      Returns:
        out: output array of SVBRDF predictions for each material's view combinations
        code: output array of associated names for the SVBRDF maps of each material's view combinations
    """

    out = []
    code = []
    num_targets = 3 * GLOBAL_IMG_NUMBER
    for j in range(num_targets):
        y_true = load_tensor(target_paths[j])
        y_name = os.path.basename(target_paths[j])
        print(y_name)
        target_num = y_name[-5]
        mat_name = y_name[:-13]
        comp_type = []
        comp_type_2 = []
        for i in range(5):
            p = f"{GLOBAL_DIR}/OutputDirectory/{i}/images"
            tmp = []
            tmp2 = []
            filenames = [
                f
                for f in glob.glob(os.path.join(p, f"*outputs-{target_num}-.png"))
                if (mat_name in f)
            ]
            filenames.sort()

            for idx, filename in enumerate(filenames):
                y_hat = load_tensor(filename)
                # print(os.path.basename(filename)[:-5])
                tmp.append(
                    {
                        "SSD": ((y_true - y_hat) ** 2).sum().cpu().detach().numpy(),
                        "SSIM": ssim(
                            y_true.view(1, *y_true.shape), y_hat.view(1, *y_hat.shape)
                        )
                        .cpu()
                        .detach()
                        .numpy(),
                    }
                )
                tmp2.append((os.path.basename(filename)[:-5]))

            comp_type.append(tmp)
            comp_type_2.append(tmp2)

        out.append(comp_type)
        code.append(comp_type_2)

    arr = np.array(out)
    print("out", arr.shape)
    print("code", code)
    return out, code


def load_tensor(path):
    """
    Helper function to load tensor for ground truth and predicted SVBRDF maps
      Parameters:
        path: filepath of SVBRDF maps for each material ( ground truth and predicted)


      Returns:
        none, but saves SSIM graphs and view combination images along with predicted
        SVBRDF maps for each view combination
    """

    im = Image.open(path)
    return transforms.functional.to_tensor(im).cuda()


def graph(code, target_paths, out):
    """
    Uses pytorch SSIM library to graph SSIM score for each view combination to compare.
      Parameters:
        code: array of names for each view combination for each material
        target_paths: array of expected ground truth SVBRDF maps
        out: outputted SVBRDF maps of each of the view combinations for comparison with targets

      Returns:
        saves graphs of SSIM for each view combination
    """
    count = 0
    for j in out:
        lbl = target_paths[count]
        map_types = ["Normal Map", "Diffuse Albedo", "Roughness Map"]
        if lbl[-5] == "0":
            map_type = 0
        print(lbl[-5])
        if lbl[-5] == "1":
            map_type = 1
        elif lbl[-5] == "2":
            map_type = 2
        print(map_type)
        print(lbl)
        plt.figure()
        name = code[count][0][0][:-20]
        for i, y in enumerate(j):
            plt.scatter(
                [i + 1 for i in range(len(y))],
                [im["SSIM"] for im in y],
                label=f"k={i+1}",
            )
        plt.title(name + " " + map_types[map_type])
        plt.xlabel("Combination")
        plt.ylabel("SSIM")
        count += 1
    plt.legend()


def run():
    """
    Uses other scripts (generate_permutations and setup_deschaintre) to create
    view combinations and runs full grid search and graphs SSIM scores for every
    view combination.
      Parameters:


      Returns:
        none, but saves SSIM graphs and view combination images along with predicted
        SVBRDF maps for each view combination
    """
    imports(GLOBAL_DIR)

    make_dirs()

    for file in glob.glob(
        f"{GLOBAL_DIR}/multi-image-deepNet-SVBRDF-acquisition/testImagesExamples/*.png"
    ):
        shutil.copy(file, BASE_DIR)

    for img in glob.glob(BASE_DIR + "/*.png"):
        print(img)
        mat = os.path.basename(img)
        generate_permutations.run(
            img,
            mat=mat[:-4],
            gt_out_dir=GT_DIR,
            permutations_out_dir=PERMUTATIONS_TEST_DIR,
        )

    setup_deschaintre.upgrade_tf(
        GLOBAL_DIR + "/multi-image-deepNet-SVBRDF-acquisition/"
    )
    if not os.path.exists(
        GLOBAL_DIR + "/multi-image-deepNet-SVBRDF-acquisition/checkpointTrained/"
    ):
        os.mkdir(
            GLOBAL_DIR + "/multi-image-deepNet-SVBRDF-acquisition/checkpointTrained/"
        )
        wget.download(
            "https://repo-sam.inria.fr/fungraph/multi_image_materials/supplemental_multi_images/checkpointTrained.zip",
            out=GLOBAL_DIR
            + "/multi-image-deepNet-SVBRDF-acquisition/checkpointTrained/",
        )
        with zipfile.ZipFile(
            GLOBAL_DIR
            + "/multi-image-deepNet-SVBRDF-acquisition/checkpointTrained/checkpointTrained.zip",
            "r",
        ) as zip_ref:
            zip_ref.extractall(
                GLOBAL_DIR
                + "/multi-image-deepNet-SVBRDF-acquisition/checkpointTrained/"
            )

    # Upgrade Tensorflow
    os.system(
        f"exec tf_upgrade_v2 --intree {GLOBAL_DIR}/multi-image-deepNet-SVBRDF-acquisition --inplace --reportfile {GLOBAL_DIR}/multi-image-deepNet-SVBRDF-acquisition/report.txt"
    )

    outputDir = GLOBAL_DIR + "/OutputDirectory/"
    checkpoint = (
        GLOBAL_DIR + "/multi-image-deepNet-SVBRDF-acquisition/checkpointTrained/"
    )

    os.system(
        f"exec python {GLOBAL_DIR}/multi-image-deepNet-SVBRDF-acquisition/pixes2Material.py --mode test --output_dir {outputDir} --input_dir {PERMUTATIONS_DIR} --batch_size 1 --input_size 256 --nbTargets 4 --useLog --includeDiffuse --which_direction AtoB --inputMode folder --maxImages 5 --nbInputs 10 --feedMethod files --useCoordConv --checkpoint {checkpoint} --fixImageNb"
    )

    target_paths = assemble_targets()
    out, code = load_data(target_paths)
    graph(code, target_paths, out)


if __name__ == "__main__":
    run()
