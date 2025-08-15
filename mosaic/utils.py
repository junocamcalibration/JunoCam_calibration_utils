import cartopy.crs as ccrs
import cv2
import numpy as np
import tqdm
from pyproj import crs
from pyproj.transformer import Transformer
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

tree = None

# set expected types to fix the NotImplementedError
ccrs._CRS._expected_types = ("Projected CRS", "Derived Projected CRS")

# define the ellipsoid and datum for Jupiter
jupiter = crs.datum.CustomEllipsoid(
    'Jupiter', semi_major_axis=71492e3, semi_minor_axis=66854e3
)
primem = crs.datum.CustomPrimeMeridian(longitude=0, name='Jupiter Prime Meridian')
jupiter_datum = crs.datum.CustomDatum(
    'Jupiter', ellipsoid=jupiter, prime_meridian=primem
)

# this is the base cylindrical projection for Jupiter
jupiter_crs = crs.GeographicCRS('Jupiter', datum=jupiter_datum)


def find_nearest_indices_kdtree(x2):
    """
    Find the indices in the tree which correspond to the locations in x2

    :param x2: query coordinates

    :returns: indices in tree which are closest to the values in x2
    """
    global tree
    _, indices = tree.query(x2, k=1)  # Query nearest
    return indices


def mosaic(
    image_filenames: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    image_coordinates: np.ndarray,
    n_filters: int,
    input_type,
) -> np.ndarray:
    """
    Construct a global mosaic given an image set of images with coordinates and an output lat/lon grid. Each pixel in the output image is averaged over the different tiles.

    :param image_filenames: input filenames for the npy files (shape: [ntiles])
    :param longitudes: array of longitudes corresponding to each tile (shape: [ntiles], units: degrees)
    :param latitudes: array of latitudes corresponding to each tile (shape: [ntiles], units: degrees)
    :param lon_grid: grid of longitudes corresponding to each pixel in the output image (shape: [output_height, output_width], units: degrees)
    :param lat_grid: grid of latitudes corresponding to each pixel in the output image (shape: [output_height, output_width], units: degrees)
    :param image_coordinates: grid of pixel distances in the LAEA coordinate frame (shape: [height, width, 2], units: m)

    :returns: image in the new (lon_grid, lat_grid) cylindrical projection (shape: [output_height, output_width, n_filters])
    """
    global tree
    n_images = len(image_filenames)
    # n_filters = np.load(image_filenames[0] + ".npy").shape[0]

    output = np.zeros((*lat_grid.shape, n_filters))
    count = np.zeros_like(output)

    # construct the K-d tree since the images coordinates will not change
    XX = image_coordinates[:, :, 0]
    YY = image_coordinates[:, :, 1]
    xx_flat = XX.flatten()
    yy_flat = YY.flatten()
    tree = cKDTree(np.vstack((xx_flat, yy_flat)).T)  # Build KD-tree

    with tqdm.tqdm(range(n_images)) as pbar:
        for n in pbar:
            latitude = latitudes[n]
            longitude = longitudes[n]
            pbar.set_postfix({'lon': longitude, 'lat': latitude})

            laea = crs.coordinate_operation.LambertAzimuthalEqualAreaConversion(
                latitude_natural_origin=latitude, longitude_natural_origin=longitude
            )
            jupiter_laea = crs.ProjectedCRS(
                laea, 'Jupiter LAEA', crs.coordinate_system.Cartesian2DCS(), jupiter_crs
            )
            # get the coordinate transformation from Cylindrical -> LAEA

            if input_type == "img":
                image_data = cv2.imread(image_filenames[n])[::-1, :]
            elif input_type == "UVM":
                image_data = cv2.imread(image_filenames[n], cv2.IMREAD_GRAYSCALE)
                image_data = np.expand_dims(image_data, axis=2)
                image_data = image_data[::-1, :]
            else:
                image_data = np.load(image_filenames[n])[::-1]

            output_i = project(
                image_data, image_coordinates, jupiter_laea, lon_grid, lat_grid
            )

            for filt in range(n_filters):
                mask = output_i[:, :, filt] > 0.0
                output[:, :, filt][mask] += output_i[:, :, filt][mask]
                count[:, :, filt][mask] += 1

    return output / (count + 1e-10)


def project(
    image_data: np.ndarray,
    coordinates: np.ndarray,
    projection: crs.coordinate_operation.CoordinateOperation,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    n_neighbor: int = 10,
    max_dist: float = 25,
) -> np.ndarray:
    """
    Construct a mosaic given an image set of images with coordinates and an output lat/lon grid

    :param image_data: input image tile (shape: [height, width, nchannels])
    :param image_coordinates: grid of pixel distances in the projection coordinate frame (shape: [height, width, 2], units: m)
    :param projection: pyproj projection of the input image
    :param lon_grid: grid of longitudes corresponding to each pixel in the output image (shape: [output_height, output_width], units: degrees)
    :param lat_grid: grid of latitudes corresponding to each pixel in the output image (shape: [output_height, output_width], units: degrees)
    :param n_neighbor: the number of nearest neighbors to use for interpolation (see create_image_from_grid)
    :param max_dist: the maximum distance of nearest neighbors to use for interpolation (see create_image_from_grid)

    :returns: image in the new (lon_grid, lat_grid) cylindrical projection (shape: [output_height, output_width, n_filters])
    """
    transformer = Transformer.from_crs(
        projection, jupiter_crs
    )  # to check the bounds of the image in lat/lon space
    inv_transformer = Transformer.from_crs(
        jupiter_crs, projection
    )  # inverse of the above

    XX = coordinates[:, :, 0]
    YY = coordinates[:, :, 1]

    x_max = np.abs(XX).max()
    y_max = np.abs(YY).max()

    lonmin, latmin, lonmax, latmax = transformer.transform_bounds(
        XX.min(), YY.min(), XX.max(), YY.max()
    )

    # mask the locations outside this latitude range
    mask_bounds = (lat_grid.flatten() >= latmin) & (lat_grid.flatten() <= latmax)

    # find the distance from the center of the image (i.e. the coordinate in LAEA frame)
    # we will use this to find the nearest pixel in the image that this lat/lon point
    # corresponds to
    xx_t = np.zeros(lon_grid.size)
    yy_t = np.zeros(lon_grid.size)
    xx_t[mask_bounds], yy_t[mask_bounds] = inv_transformer.transform(
        lon_grid.flatten()[mask_bounds], lat_grid.flatten()[mask_bounds]
    )

    xx_t[~mask_bounds] = 1e25
    yy_t[~mask_bounds] = 1e25

    pix = np.nan * np.zeros((lon_grid.size, 2))
    mask_extent = (np.abs(xx_t) < x_max) & (np.abs(yy_t) < y_max)

    pix[mask_extent] = np.stack(
        np.unravel_index(
            find_nearest_indices_kdtree(
                np.vstack([xx_t[mask_extent], yy_t[mask_extent]]).T
            ),
            (XX.shape[0], XX.shape[1]),
        ),
        axis=1,
    )[:, ::-1]

    inds = np.where(np.isfinite(pix[:, 0] * pix[:, 1]))[0]
    pix_masked = pix[inds]
    pixel_inds = np.arange(lon_grid.size)[inds]
    coordsx = np.arange(image_data.shape[1])
    coordsy = np.arange(image_data.shape[0])

    coords = np.stack(np.meshgrid(coordsx, coordsy), axis=2).reshape((-1, 2))

    return create_image_from_grid(
        coords,
        image_data.reshape((-1, image_data.shape[-1])),
        pixel_inds,
        pix_masked,
        lat_grid.shape,
        n_neighbor=n_neighbor,
        max_dist=max_dist,
    )


def create_image_from_grid(
    coords: np.ndarray,
    imgvals: np.ndarray,
    inds: np.ndarray,
    pix: np.ndarray,
    img_shape: tuple[int],
    n_neighbor: int = 5,
    max_dist: float = 25.0,
) -> np.ndarray:
    '''
        Reproject an irregular spaced image onto a regular grid from a list of coordinate
        locations and corresponding image values. This uses an inverse lookup-table defined
        by `pix`, where pix gives the coordinates in the original image where the corresponding
        pixel coordinate on the new image should be. The coordinate on the new image is given by
        the `inds`.

    :param coords: the pixel coordinates in the original image
    :param imgvals: the image values corresponding to coords
    :param inds: the coordinate on the new image where we need to interpolate
    :param pix: the coordinate in the original image corresponding to each pixel in inds
    :param img_shape: the shape of the new image
    :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
    :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

    :return: the interpolated new image of shape `img_shape` where every pixel at `inds` has corresponding values interpolated from `imgvals`
    '''
    ncoords, _ = coords.shape
    nchannels = imgvals.shape[-1]

    newvals = np.zeros((pix.shape[0], nchannels))
    mask = np.isfinite(coords[:, 0] * coords[:, 1])
    neighbors = NearestNeighbors().fit(coords[mask])
    dist, indi = neighbors.kneighbors(pix, n_neighbor)
    weight = 1.0 / (dist + 1.0e-16)
    weight = weight / np.sum(weight, axis=1, keepdims=True)
    weight[dist > max_dist] = 0.0

    for n in range(nchannels):
        newvals[:, n] = np.sum(
            np.take(imgvals[:, n][mask], indi, axis=0) * weight, axis=1
        )

    IMG = np.zeros((*img_shape, nchannels))
    # loop through each point observed by JunoCam and assign the pixel value
    for k, ind in enumerate(inds):
        if len(img_shape) == 2:
            j, i = np.unravel_index(ind, img_shape)

            # do the weighted average for each filter
            for n in range(nchannels):
                IMG[j, i, n] = newvals[k, n]
        else:
            for n in range(nchannels):
                IMG[ind, n] = newvals[k, n]

    IMG[~np.isfinite(IMG)] = 0.0

    return IMG
