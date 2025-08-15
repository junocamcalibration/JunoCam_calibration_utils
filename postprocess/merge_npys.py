import argparse
import os
import glob
import tqdm

import numpy as np

# Merge RGB, UV, and Methane npys to prepare them for converting to GeoTiff

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "-bgr_files",
        "--bgr_files",
        help="Path to folder containing the BGR npy files",
    )
    parser.add_argument(
        "-uvm_files",
        "--uvm_files",
        help="Path to folder containing the UVM npy files",
    )
    parser.add_argument(
        "-save_dir", "--save_dir", help="Root directory to save the npy files"
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    job_BGR_dir = args.bgr_files
    job_UVM_dir = args.uvm_files

    BGR_files = sorted(glob.glob(os.path.join(job_BGR_dir, "*.npy")))
    UVM_files = sorted(glob.glob(os.path.join(job_UVM_dir, "*.npy")))

    if len(BGR_files) != len(UVM_files):
        raise ValueError(
            f"Number of BGR files ({len(BGR_files)}) is not the same as UVM files ({len(UVM_files)})"
        )

    for i, (BGR_file, UVM_file) in enumerate(tqdm.tqdm(zip(BGR_files, UVM_files), total=len(BGR_files), desc='Merging npys')):
        BGR_fname = os.path.basename(BGR_file)
        UVM_fname = os.path.basename(UVM_file)
        bgr = np.load(BGR_file)  # H x W x 3, B, G, R
        uvm = np.load(UVM_file)  # H x W x 2, UV, M
        # merge them and change the order of the wavelength axis
        merged_npy = np.concatenate((bgr, uvm), axis=2)[
            :, :, [3, 0, 1, 2, 4]
        ]  # UV, B, G, R, M
        np.save(os.path.join(save_dir, BGR_fname), merged_npy)

