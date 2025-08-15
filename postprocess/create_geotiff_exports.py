import argparse
import os

import numpy as np
import rasterio
import tqdm
from astropy.io import ascii
from pyproj.crs import coordinate_operation, coordinate_system, crs, datum
from rasterio.transform import from_bounds

# define the ellipsoid and datum for Jupiter
jupiter = datum.CustomEllipsoid(
    'Jupiter', semi_major_axis=71492e3, semi_minor_axis=66854e3
)
primem = datum.CustomPrimeMeridian(longitude=0, name='Jupiter Prime Meridian')
jupiter_datum = datum.CustomDatum('Jupiter', ellipsoid=jupiter, prime_meridian=primem)

# this is the base cylindrical projection for Jupiter
jupiter_crs = crs.GeographicCRS('Jupiter', datum=jupiter_datum)


def save_geotiff(
    output_fname: str,
    segment: np.ndarray,
    longitude: float,
    latitude: float,
    max_dist: float = 8,
) -> None:
    """
    Convert the input image segment and save it as a GeoTIFF

    :param output_fname: output filename for the GeoTIFF
    :param segment: the segment data (shape: [ny, nx, nchannels])
    :param longitude: the center longitude of the segment [degree]
    :param latitude: the center latitude of the segment [degree]
    :param max_dist: the distance from the center to the edge of the segment [Mm]
    """

    # ensure the the input file has a channel dimension
    segment = np.atleast_3d(segment)

    # the input file is in LAEA projection, so we will build the projection using the latitude/longitude
    # value of the file
    laea = coordinate_operation.LambertAzimuthalEqualAreaConversion(
        latitude_natural_origin=latitude, longitude_natural_origin=longitude
    )
    jupiter_laea = crs.ProjectedCRS(
        laea, 'Jupiter LAEA', coordinate_system.Cartesian2DCS(), jupiter_crs
    )

    # get the transformation from the pixel index to distance
    transform = from_bounds(
        -max_dist * 1e6,
        -max_dist * 1e6,
        max_dist * 1e6,
        max_dist * 1e6,
        file_data.shape[1],
        file_data.shape[0],
    )

    # save the output file
    with rasterio.open(
        output_fname,
        'w',
        driver='GTiff',
        height=segment.shape[0],
        width=segment.shape[1],
        count=segment.shape[2],
        dtype=segment.dtype,
        crs=jupiter_laea,
        transform=transform,
    ) as dset:
        for i in range(segment.shape[2]):
            dset.write(segment[:, :, i].astype(np.double), i + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--input_metadata",
        help="Path to the metadata file containing the metadata for each segment",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--path-to-images",
        help="Path to folder containing the images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--path-to-output",
        help="Path to output folder to store GeoTIFFs",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--max_dist",
        help="Distance from edge of segment from the center [in Mm]",
        type=float,
        default=8,
    )
    args = parser.parse_args()

    if not os.path.exists(args.path_to_output):
        os.makedirs(args.path_to_output)

    table = ascii.read(args.input_metadata)
    files = []

    for ind in range(len(table)):
        file_name = os.path.basename(table['filepath_npy'][ind])
        filename = os.path.join(args.path_to_images, f"{file_name}.npy")
        if os.path.isfile(filename):
            files.append((ind, filename))

    for ind, file in tqdm.tqdm(files):
        latitude = table['latitude'][ind].item()
        longitude = table['longitude'][ind].item()
        # print(type(latitude.item()), type(longitude.item()))
        file_data = np.load(file)
        # print(file_data.min(), file_data.max())

        fname, _ = os.path.splitext(os.path.basename(file))

        save_geotiff(
            os.path.join(args.path_to_output, f"{fname}.tif"),
            file_data,
            longitude,
            latitude,
        )
