'''
This is the main driver for generating crops from different Hubble OPAL data.
To run this code, you need to download the OPAL data from https://archive.stsci.edu/hlsp/opal
or run the get_links.sh.
The JVH_PJimgs can be downloaded here: https://github.com/ramanakumars/JuDE/tree/main/backend/PJimgs

This code expects the data to be in an adjacent data/ folder with the following folder structure:
data/
    get_links.sh
    HST_fits/
        cycleXX/
            hlsp_***.fits
            hlsp_***_readme.txt
            ...
    JVH_PJimgs/
        PJXX
            globe_mosaic_highres.png
        ...
'''

import numpy as np
import netCDF4 as nc
import utils as ut
import re
import glob
import tqdm
import healpy as hp
from astropy.io import fits
import gc
import os
import argparse
import cv2
from params import Params
from sklearn.neighbors import NearestNeighbors


def collect_HST_maps(cycle, filts, pattern, fits_resolution, rot_list):
    # Get all the HST FITS images for this cycle
    files = sorted(glob.glob(f'data/HST_fits/cycle{cycle}/*.fits'))
    file_data = []

    # Load the scale parameters to convert to I/F
    scales = ut.get_i_f_scale(cycle)

    # go through the files and get the corresponding info from the filename
    for file in files:
        match = re.findall(pattern, file.split('/')[-1])
        if len(match) > 0:
            year, rot, filt = match[0]
            if filt in filts:
                file_data.append({'year': year, 'rotation': rot, 'filter': filt,
                                  'path': file, 'filename': file.split('/')[-1]})

    # fill in the image from the FITS files
    scales_cycle = []
    img = np.zeros((fits_resolution[0], fits_resolution[1], len(filts), len(rot_list)))

    for r, rot in enumerate(rot_list):
        for f, filt in enumerate(filts):
            try:
                scale = scales[str(cycle)][r]
            except IndexError:
                print(f"Skipping {filt}")
                break
            file = list(filter(
                lambda f: (f['rotation'] == rot) & (
                    f['filter'] == filt), file_data
            ))[0]['path']
            # print(file)

            hdu = fits.open(file)
            imgi = hdu[0].data
            
            img[:, :, f, r] = imgi
        img_scale = np.percentile(img.flatten(), 99.9)
        scales_cycle.append({filt: scale[filt.upper()] * img_scale for filt in filts})

    # normalize the image
    img = img / np.percentile(img.flatten(), 99.9)

    return file_data, img, scales_cycle


def collect_JVH_maps(perijove):
    jvh_map = cv2.imread("data/JVH_PJimgs/PJ" + str(perijove) + "/globe_mosaic_highres.png")
    jvh_map = (jvh_map / 255.0).astype(np.float32)
    return jvh_map


def create_database(nc_fname, img_size, filts, rot_dim):
    with nc.Dataset(nc_fname, 'w') as dset:
        dset.createDimension('segment', None)  # number of segments
        dset.createDimension('y', img_size[0])
        dset.createDimension('x', img_size[1])
        dset.createDimension('filter', len(filts))
        dset.createDimension('rot', rot_dim)

        dset.filters = filts

        dset.createVariable('img', 'u2', ('segment', 'y', 'x', 'filter', 'rot'))
        dset.createVariable('lon', 'float32', ('segment'))
        dset.createVariable('lat', 'float32', ('segment'))
    return dset


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def convert_to_centric(lonlats):
    new_lonlats = np.zeros((lonlats.shape[0], lonlats.shape[1]), dtype=np.float32)
    # There is a small misalignment between latitudes of HST and JVH
    with open('data/kmdeg_0.25.txt', 'r') as file:
        lines = [line.rstrip() for line in file]
    lines = lines[2:]  # remove title and empty line

    lats_correspondence = []
    for i in range(len(lines)):
        cols = [x for x in lines[i].split(' ') if x != '']
        # 0 is graphic, 1 is centric
        lats_correspondence.append(cols[:2])
    lats_correspondence = np.asarray(lats_correspondence).astype(np.float32)

    # For every sampled lat in lonlats find the closest graphic lat
    # Update the lat with the corresponding centric lat

    # Need to keep the sign of the original lat because lats_correspondence is only positive
    lats_signs = np.sign(lonlats[:, 1])

    sampled_lats = np.abs(lonlats[:, 1])
    graphic_lats = lats_correspondence[:, 0]

    dist, idx = nearest_neighbor(sampled_lats.reshape(-1, 1), graphic_lats.reshape(-1, 1))
    new_lats = lats_correspondence[idx, 1]
    new_lats = np.multiply(new_lats, lats_signs)
    new_lonlats[:, 0] = lonlats[:, 0]
    new_lonlats[:, 1] = new_lats
    return new_lonlats


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--cycle", type=int, default=0, help="HST OPAL Cycle to process (22-30). If 0, skip HST", required=False)
    parser.add_argument("--nside", type=int, default=32, help="Map resolution needed by healpy, must be power of 2", required=False)
    parser.add_argument("--max_lat", type=int, default=72, help="Max latitude for sampling segments", required=False)
    parser.add_argument("--IMG_H", type=int, default=256, help="for now keep same as cut_size...", required=False)
    parser.add_argument("--IMG_W", type=int, default=256, help="for now keep same as cut_size...", required=False)
    parser.add_argument("--max_dist", type=int, default=8, help="The maximum distance in Mm from the center to the edge of the frame in each axis", required=False)
    parser.add_argument("--perijove", type=int, default=0, help="JVH Perijove map to process (13-36), If 0, skip JVH", required=False)
    parser.add_argument('--zones', nargs='+', default=["GRS"], help='Which zone to use. The corresponding lats are defined in params.py')
    args = parser.parse_args()

    params = Params(args)

    # Sample segment locations (across all lon and lat)
    npix = hp.nside2npix(params.nside)
    lons, lats = hp.pix2ang(params.nside, np.arange(npix), lonlat=True)  # Angular coordinates corresponding to pixel indices
    lons -= 180

    for zone in params.zones:
        lat_range = params.zones_lats[zone]
        print("Creating databases for", zone, "at lat range", lat_range)

        # Sample coordinates within the zone range
        lat_inds = np.where((lats <= lat_range[1]) & (lats >= lat_range[0]))[0]
        lonlat_flat = np.stack([lons[lat_inds], lats[lat_inds]], axis=1)

        # For JVH we need to convert the sampled lats to centric (essentially a small shift)
        lonlat_flat_jvh = convert_to_centric(lonlat_flat)

        # Create and populate NC database for HST for this specific zone
        if params.cycle != 0:
            print("Collecting HST from cycle", params.cycle, "for zone", zone)
            
            params.set_dataset_params('HST')

            # Special case where rot B of cycle 27 is empty
            if params.cycle == 27:
                rot_list = ['a']
            else:
                rot_list = ['a', 'b']

            fits_resolution = (params.dataset_params['HST']['lat_resolution'], params.dataset_params['HST']['lon_resolution'])

            _, hst_map, scales_cycle = collect_HST_maps(params.cycle, params.filts, params.fits_pattern,
                                                                fits_resolution=fits_resolution, rot_list=rot_list)

            hst_nc_fname = params.hst_nc_dir + 'cycle' + str(params.cycle) + '_rot_' + zone + '.nc'
            if not os.path.isdir(params.hst_nc_dir):
                os.makedirs(params.hst_nc_dir)
            create_database(hst_nc_fname, img_size=(params.IMG_H, params.IMG_W), filts=params.filts, rot_dim=len(rot_list))

            with tqdm.tqdm(range(len(lonlat_flat)), desc=f'Cycle {params.cycle}', ascii=True, dynamic_ncols=True) as pbar:
                for n in pbar:
                    lonlati = lonlat_flat[n]
                    lon, lat = lonlati

                    pbar.set_postfix_str(f"nremain: {len(pbar) - n:6d} lon: {lonlati[0]:.1f} lat: {lonlati[1]:.1f}")

                    # get the hst frame at this lat/lon for all filters
                    hst_frames = ut.get_frame_at_lonlat(lon, lat, hst_map, params, 'HST', img_size=(params.IMG_H, params.IMG_W), max_dist=params.max_dist)

                    if hst_frames is None:  # there are missing pixels
                        print("Returned HST frame is invalid! Skipping ...")
                    else:
                        # save the frame and corresponding details out to the NC file
                        with nc.Dataset(hst_nc_fname, 'r+') as dset:
                            start = dset.dimensions['segment'].size
                            for rot in range(len(rot_list)):
                                for f, filt in enumerate(params.filts):
                                    hst_frames[:, :, f, rot] = hst_frames[:, :, f, rot] * scales_cycle[rot][filt]
                            # Converting to 16-bit unsigned int to save disk space. The loss of information is minimal (after 4 decimal points)
                            dset.variables['img'][start] = np.asarray(hst_frames * 65535, dtype=np.ushort)
                            dset.variables['lon'][start] = lonlati[0]
                            dset.variables['lat'][start] = lonlati[1]
                        gc.collect()

        # Create and populate NC database for JVH for this specific zone
        if params.perijove != 0:
            print("collecting JVH map from PJ", params.perijove, "for zone", zone)
            
            params.set_dataset_params('JVH')
            jvh_map = collect_JVH_maps(params.perijove)

            jvh_nc_fname = params.jvh_nc_dir + 'PJ' + str(params.perijove) + '_' + zone + '.nc'
            if not os.path.isdir(params.jvh_nc_dir):
                os.makedirs(params.jvh_nc_dir)
            create_database(jvh_nc_fname, img_size=(params.IMG_H, params.IMG_W), filts=['R', 'G', 'B'], rot_dim=1)

            with tqdm.tqdm(range(len(lonlat_flat_jvh)), desc=f'PJ {params.perijove}', ascii=True, dynamic_ncols=True) as pbar:
                for n in pbar:
                    lonlati = lonlat_flat_jvh[n]
                    lon, lat = lonlati

                    lat_ind = np.argmin((params.dataset_params['JVH']['lat_range'] - lat)**2.)
                    lon_ind = np.argmin((params.dataset_params['JVH']['lon_range'] - lon)**2.)

                    # ignore tiles where there is no data in the center
                    # this should speed up calculation a bit
                    if jvh_map[lat_ind, lon_ind].sum() < 1e-10:
                        continue

                    pbar.set_postfix_str(f"nremain: {len(pbar) - n:6d} lon: {lonlati[0]:.1f} lat: {lonlati[1]:.1f}")

                    # get the JVH frame at this lat/lon
                    jvh_frames = ut.get_frame_at_lonlat(lon, lat, jvh_map, params, 'JVH', img_size=(params.IMG_H, params.IMG_W), max_dist=params.max_dist)

                    if jvh_frames is None:
                        print("Returned JVH frame is invalid. Probably blank. Skipping ...")
                    else:
                        with nc.Dataset(jvh_nc_fname, 'r+') as dset:
                            start = dset.dimensions['segment'].size
                            dset.variables['img'][start] = np.asarray(jvh_frames * 65535, dtype=np.ushort)
                            dset.variables['lon'][start] = lonlati[0]
                            dset.variables['lat'][start] = lonlati[1]
                        gc.collect()
