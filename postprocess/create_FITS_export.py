import argparse
import json
import logging

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_KEYS = ['junopj', 'modelep', 'ckpt', 'HSTCYC']

COMMENTS = {
    "junopj": "Juno perijove number",
    "modelep": "model epoch",
    "ckpt": "model checkpoint",
    "hstcyc": "HST cycle used for training",
}

FILTER_NAMES = ['F275W', 'F395N', 'F502N', 'F631N', 'FQ889N']


def get_comments(key: str) -> str | None:
    return COMMENTS[key.lower()] if key.lower() in COMMENTS else None


def format_dict(input: dict) -> dict:
    return {
        key.upper(): {"value": value, "comment": get_comments(key)}
        for key, value in input.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "-path",
        "--path",
        help="Path to .npy file containing the mosaic",
        required=False,
    )
    parser.add_argument(
        "-bgr_path",
        "--bgr_path",
        help="Path to .npy file containing the BGR mosaic",
        required=False,
    )
    parser.add_argument(
        "-uv_path",
        "--uv_path",
        help="Path to .npy file containing the uv mosaic",
        required=False,
    )
    parser.add_argument(
        "-methane_path",
        "--methane_path",
        help="Path to .npy file containing the methane mosaic",
        required=False,
    )
    parser.add_argument("-metadata", "--metadata", help="Path to metadata .json file")
    parser.add_argument(
        "-output", "--output", help="Output FITS filename (no extension)"
    )
    parser.add_argument(
        "-overwrite", "--overwrite", action="store_true", help="overwrite existing FITS export"
    )
    args = parser.parse_args()

    if args.path:
        data = np.load(args.path)
    elif args.bgr_path and args.uv_path and args.methane_path:
        bgr = np.load(args.bgr_path)
        uv = np.load(args.uv_path)
        methane = np.load(args.methane_path)
        data = np.concatenate((bgr, uv, methane), axis=2)[:, :, [3, 0, 1, 2, 4]]
    else:
        raise ValueError(
            "Either pass in --path with the filepath to a single 5-channel npy or --bgr_path, --uv_path and --methane_path to BGR/UV/M mosaics"
        )

    height, width, nfilters = data.shape
    if nfilters != 5:
        raise ValueError(f"Data must be 5-channel mosaic. Found shape: {data.shape}")

    if (width - 1) / (height - 1) != 2:
        raise ValueError(f"Data must be full-globe mosaic. Found shape: {data.shape}")

    map_resolution = 360 / width

    with open(args.metadata, 'r') as indata:
        metadata = format_dict(json.load(indata))

    for key in REQUIRED_KEYS:
        if key.upper() not in metadata:
            logger.warning(f"{key.upper()} keyword is not in the metadata list!")

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [width / 2, height / 2]
    wcs.wcs.cdelt = [-map_resolution, map_resolution]
    wcs.wcs.crval = [180, 0]
    wcs.wcs.ctype = ['JULN-CAR', 'JULT-CAR']

    header = fits.Header({})
    for key, value in metadata.items():
        header.set(key, value=value["value"], comment=value["comment"])
    hdu = fits.PrimaryHDU(header=header)
    hdus = [hdu]
    for n in range(nfilters):
        imghdu = fits.ImageHDU(data=data[:,:,n],name=FILTER_NAMES[n],header=fits.Header(wcs.to_header()))
        hdus.append(imghdu)
    hdulist = fits.HDUList(hdus=hdus)


    hdulist.writeto(f"{args.output}.fits", overwrite=args.overwrite)
