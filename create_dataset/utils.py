import numpy as np
import glob
import re
from pyproj import crs
import pprint
from params import Params
import cartopy.crs as ccrs
from scipy.spatial import cKDTree
from pyproj.transformer import Transformer
from skimage.transform import rescale


# set expected types to fix the NotImplementedError
ccrs._CRS._expected_types = ("Projected CRS", "Derived Projected CRS")


def parse_readme(readme):
    '''Load the README file and parse the I/F scale

    :param readme: path to the README file

    :return: the corresponding I/F scales for all the filters
    '''
    pattern = r'FQ?([0-9]+N?W?)\s+(\S+)\s+(\.[0-9]+)'
    out = {'rot1': {}, 'rot2': {}}
    with open(readme, 'r') as data:
        lines = data.getlines()

        rotation = None
        for line in lines:
            if "Rotation 1" in line:
                rotation = 1
            elif "Rotation 2" in line:
                rotation = 2

            matches = re.findall(pattern, line)

            if len(matches) > 0:
                filt, minn, scale = matches[0]
                out[f'rot{rotation}'][filt] = scale
    return scale


def find_nearest_indices_kdtree(x2, tree):
    """
    Find the indices in the tree which correspond to the locations in x2

    :param x2: query coordinates

    :returns: indices in tree which are closest to the values in x2
    """
    _, indices = tree.query(x2, k=1)  # Query nearest
    return indices


def project(image_data: np.ndarray, projection: crs.coordinate_operation.CoordinateOperation,
            x_grid: np.ndarray, y_grid: np.ndarray, params: Params, dataset: str, n_neighbor: int = 5, max_dist: float = 25) -> np.ndarray:
    """
    Construct a mosaic given an image set of images with coordinates and an output lat/lon grid

    :param image_data: input image tile (shape: [height, width, nchannels])
    :param image_coordinates: grid of lon/lat in the image (shape: [height, width, 2], units: m)
    :param projection: pyproj projection of the input image
    :param x_grid: grid of distances along x-axis corresponding to each pixel in the output image (shape: [output_height, output_width], units: m)
    :param y_grid: grid of distances along y-axis corresponding to each pixel in the output image (shape: [output_height, output_width], units: m)
    :param params: parameters object containing Jupiter's radius and other constants
    :param n_neighbor: the number of nearest neighbors to use for interpolation (see create_image_from_grid)
    :param max_dist: the maximum distance of nearest neighbors to use for interpolation (see create_image_from_grid)

    :returns: image in the new (lon_grid, lat_grid) cylindrical projection (shape: [output_height, output_width, n_filters])
    """
    inv_transformer = Transformer.from_crs(projection, params.jupiter_crs)  # inverse of the above

    LON = params.dataset_params[dataset]['LON']
    LAT = params.dataset_params[dataset]['LAT']
    tree = params.dataset_params[dataset]['lonlat_tree']

    # get the corresponding lat/lon grid for the image
    lon_t, lat_t = inv_transformer.transform(x_grid.flatten(), y_grid.flatten())

    pix = np.nan * np.zeros((x_grid.size, 2))
    mask = (np.abs(lon_t) <= np.abs(LON).max()) & (lat_t <= LAT.max()) & (lat_t >= LAT.min())
    pix[mask] = np.stack(np.unravel_index(find_nearest_indices_kdtree(np.vstack([lon_t[mask], lat_t[mask]]).T, tree), (LON.shape[0], LON.shape[1])), axis=1)[:, ::-1]

    inds = np.where(np.isfinite(pix[:, 0] * pix[:, 1]))[0]
    pix_masked = pix[inds]
    pixel_inds = np.asarray(range(pix.shape[0]))[inds]
    coordsx = np.arange(image_data.shape[1])
    coordsy = np.arange(image_data.shape[0])

    coords = np.stack(np.meshgrid(coordsx, coordsy), axis=2).reshape((-1, 2))

    return create_image_from_grid(coords, image_data.reshape((-1, image_data.shape[-1])), pixel_inds, pix_masked, params.dataset_params[dataset]['pix_tree'], x_grid.shape, n_neighbor=n_neighbor, max_dist=max_dist)


def create_image_from_grid(coords: np.ndarray, imgvals: np.ndarray, inds: np.ndarray, pix: np.ndarray, tree: cKDTree,
                           img_shape: tuple[int], n_neighbor: int = 5, max_dist: float = 25.) -> np.ndarray:
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
    :param tree: the KDTree of the original image coordinates (in pixel space)
    :param img_shape: the shape of the new image
    :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
    :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

    :return: the interpolated new image of shape `img_shape` where every pixel at `inds` has corresponding values interpolated from `imgvals`
    '''
    nchannels = imgvals.shape[-1]

    newvals = np.zeros((pix.shape[0], nchannels))
    mask = np.isfinite(coords[:, 0] * coords[:, 1])
    dist, indi = tree.query(pix, k=n_neighbor)
    weight = 1. / (dist + 1.e-16)
    weight = weight / np.sum(weight, axis=1, keepdims=True)
    weight[dist > max_dist] = 0.

    for n in range(nchannels):
        newvals[:, n] = np.sum(np.take(imgvals[:, n][mask], indi, axis=0) * weight, axis=1)

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

    IMG[~np.isfinite(IMG)] = 0.

    return IMG


def get_frame_at_lonlat(lon: float, lat: float, img: np.ndarray, params: Params, dataset: str, img_size: tuple[int], max_dist: float = 8):
    '''Get the frame at a given longitude/latitude coordinate

    :param lon: central longitude [degrees]
    :param lat: central latitude [degrees]
    :param img: the full multi-channel/multi-rotation image
    :param params: the configuration parameters
    :param dataset: the dataset name (HST or JVH)
    :param img_size: the size of the image to be returned (height, width)
    :param max_dist: the maximum distance from the center to include in the frame (in 1000s of km)

    :return: the image cube centered at lon/lat for all channels
    '''

    x_grid = np.linspace(-max_dist * 1e6, max_dist * 1e6, img_size[1])
    y_grid = np.linspace(-max_dist * 1e6, max_dist * 1e6, img_size[0])

    XX, YY = np.meshgrid(x_grid, y_grid)

    laea = crs.coordinate_operation.LambertAzimuthalEqualAreaConversion(latitude_natural_origin=lat, longitude_natural_origin=lon)
    jupiter_laea = crs.ProjectedCRS(laea, 'Jupiter LAEA', crs.coordinate_system.Cartesian2DCS(), params.jupiter_crs)

    # reproject the HST data so that the rotations are concatenated to the filter axis
    if dataset == 'HST':
        n_rots = img.shape[3]
        img = img.reshape((img.shape[0], img.shape[1], img.shape[2] * img.shape[3]))

    frames = project(img, jupiter_laea, XX, YY, params, dataset)

    # check to make sure there are no blank regions in the frame
    if np.sum(frames[:, :, 0] < 1.e-10) > 5:
        print("Frame is blank!")
        return None

    if dataset == 'HST':
        frames = frames.reshape((frames.shape[0], frames.shape[1], -1, n_rots))

    return frames[::-1]


def get_i_f_scale(cycle):
    # This function reads in the readme file for all the data in the data/ folder and 
    # generates the I/F scaling for each cycle/rotation/filter pair
    cycle_data = {}

    readme = sorted(glob.glob(f'data/HST_fits/cycle{cycle}/*_readme.txt'))[0]
    with open(readme, 'r') as infile:
        lines = infile.readlines()

        rot_lines = []
        for i, line in enumerate(lines):
            if re.match(r'^Rotation [12] \(', line):
                print(line)
                rot_lines.append(i)

        if len(rot_lines) > 1:
            if rot_lines[1] - rot_lines[0] < 5:
                rot_lines[0] = rot_lines[1]

        filter_data = []

        for rot, start in enumerate(rot_lines):
            try:
                end = rot_lines[rot + 1]
                if end == start:
                    raise IndexError("using the same scales for both rotations")
            except IndexError:
                end = len(lines)

            line_rot = lines[start:end]

            filt_rot = {}
            for line in line_rot:
                matches = re.findall(r'(F[Q]?[0-9]+[WN])\s+([0-9\.n/a]+)\s+([0-9\.]+)', line)
                if len(matches) > 0:
                    filt_rot[matches[0][0]] = float(matches[0][-1])

            filter_data.append(filt_rot)
    pprint.pprint(filter_data)
    cycle_data[str(cycle)] = filter_data
    return cycle_data


def color_correction(data):
    scaled = rescale(data, (0.25, 0.25, 1))
    gray = scaled.mean(axis=-1)
    y, x = np.unravel_index(np.argmax(gray), (gray.shape[0], gray.shape[1]))
    val = 1 / scaled[y, x, :]

    data[:, :, 2] = val[2] * data[:, :, 2]
    data[:, :, 1] = val[1] * data[:, :, 1]
    data[:, :, 0] = val[0] * data[:, :, 0]

    data = data / (1.1 * data.max())

    return data
