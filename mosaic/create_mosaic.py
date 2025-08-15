'''
This is the main driver for generating mosaics from a given set of image tiles
'''

import argparse
import os
import pathlib

import cv2
import numpy as np
import tqdm
from astropy.io import ascii
from utils import mosaic

"""
# Example cmd using images:

python create_mosaic.py \
--input mosaic_example/junocam_calibration_C27_PJ27_npy_SB_ZonePair_fixBug/A_metadata.csv \
--path_to_images mosaic_example/junocam_calibration_C27_PJ27_npy_SB_ZonePair_fixBug/fake_images/ \
--subsample_rate 2 \
--output mosaic_example/junocam_calibration_C27_PJ27_npy_SB_ZonePair_fixBug/mosaic \
--input_type img \
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="input csv file containing the image filenames and metadata",
        required=True,
    )
    parser.add_argument(
        "--path_to_images",
        type=str,
        default='mosaic_example/testA/',
        help="",
        required=False,
    )
    parser.add_argument(
        "--subsample_rate", type=int, default=1, help="Subsample images", required=False
    )
    parser.add_argument(
        "--output_resolution",
        type=float,
        default=10,
        help="output resolution [pixels per degree]",
        required=False,
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=8,
        help="distance to the edge from the center of the image in Mm",
        required=False,
    )
    parser.add_argument(
        "--extents",
        type=float,
        nargs=4,
        default=[-180, 180, -90, 90],
        help="the extent of the mosaic (lon_min, lon_max, lat_min, lat_max) [degrees]",
        required=False,
    )
    parser.add_argument(
        "--output",
        type=str,
        default='mosaic',
        help="output file name (without extension)",
        required=False,
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default='npy',
        choices=['npy', 'img', "UVM"],
        help="UVM assumes either UV or Methane images as 1 channel inputs",
        required=False,
    )
    args = parser.parse_args()

    metadata = ascii.read(args.input, format='csv')

    # latitude = np.array(metadata['latitude'])
    # longitude = np.array(metadata['longitude'])
    # meta_filenames = np.array(metadata['filepath_'+args.input_type])

    # Read list of imgs from given path
    filelist = os.listdir(args.path_to_images)
    filelist.sort()
    filelist = [os.path.join(args.path_to_images, x) for x in filelist]
    img_filenames = np.asarray(filelist)
    img_filenames = img_filenames[:: args.subsample_rate]  # subsample images

    # The filepath in the metadata points to the original segments, not the dataset, so we need to match the filenames to get the lat/lon
    filetype = 'img' if args.input_type in ['img', 'UVM'] else args.input_type
    latitude, longitude = [], []
    meta_filepaths = np.array(metadata['filepath_' + args.input_type])
    meta_file_ids = [os.path.splitext(os.path.basename(x))[0] for x in meta_filepaths]
    for idx in tqdm.tqdm(range(len(img_filenames)), desc="Getting file info"):
        file_id, _ = os.path.splitext(os.path.basename(img_filenames[idx]))
        info = metadata[meta_file_ids.index(file_id)]
        latitude.append(info['latitude'])
        longitude.append(info['longitude'])

    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)

    # print(latitude)

    # assume that the image properties are the same for all images
    if args.input_type == "img":
        img_height, img_width, nfilters = cv2.imread(img_filenames[0]).shape
    elif args.input_type == "UVM":
        img_height, img_width = cv2.imread(img_filenames[0], cv2.IMREAD_GRAYSCALE).shape
        nfilters = 1
    else:
        img_height, img_width, nfilters = np.load(img_filenames[0]).shape

    lonmin, lonmax, latmin, latmax = args.extents

    lon_grid = np.linspace(
        lonmin,
        lonmax,
        int((lonmax - lonmin) * args.output_resolution) + 1,
        endpoint=True,
    )
    lat_grid = np.linspace(
        latmin,
        latmax,
        int((latmax - latmin) * args.output_resolution) + 1,
        endpoint=True,
    )

    # print(lat_grid.min(), lat_grid.max())

    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    xx_grid = np.linspace(-args.max_dist, args.max_dist, img_width) * 1e6
    yy_grid = np.linspace(-args.max_dist, args.max_dist, img_height) * 1e6
    XX, YY = np.meshgrid(xx_grid, yy_grid)
    image_coordinates = np.stack([XX, YY], axis=2)

    mos = mosaic(
        img_filenames,
        longitude,
        latitude,
        LON,
        LAT,
        image_coordinates,
        nfilters,
        args.input_type,
    )

    np.save(f'{args.output}.npy', mos)

    # print(mosaic.shape)
    if args.input_type == 'npy':
        mos *= 255

    if nfilters == 2:
        # Case when we are loading UVM as npy files
        cv2.imwrite(f'{args.output}_UV.png', mos[0, :, :])
        cv2.imwrite(f'{args.output}_M.png', mos[1, :, :])
    elif nfilters == 3:
        # Every other case nfilters should be 3 or 1
        cv2.imwrite(f'{args.output}.png', mos[::-1, :, :])
    else:
        print(f"Found {nfilters=}. Cannot create an image output")
