import cartopy.crs as ccrs
from pyproj import crs
from scipy.spatial import cKDTree
import numpy as np


class Params:
    '''
    Define params that should NOT change, or should rarely change
    '''

    def __init__(self, args=None):
        self.fits_pattern = 'hlsp_opal_hst_wfc3-uvis_jupiter-([0-9]{4})([ab])_(f[q]?[0-9nw]{4})_v1_globalmap.fits'

        self.dist_map_path = 'data/dist_map_'
        self.hst_nc_dir = 'data/NC/HST/'
        self.jvh_nc_dir = 'data/NC/JVH/'

        self.hst_segments_dir = 'data/Segments/HST/'
        self.jvh_segments_dir = 'data/Segments/JVH/'

        self.pairs_save_dir = 'data/training_pairs/'

        # Define latitude ranges for different zones/belts.
        self.zones_lats = {
            'GRS': [-23.0, -21.0],  # Great Red Spot
            'EZ': [-7.0, 7.0],  # Equatorial Zone
            'NEB': [7.0, 21.0],  # North Equatorial Belt
            'NTB': [25.0, 30.0],  # North Temperate Belt
            'NTrZ': [21.0, 25.0],  # North Tropical Zone
            'SEB': [-21.0, -7.0],  # South Equatorial Belt
            'STB': [-30.0, -25.0],  # South Temperate Belt
            'STrZ': [-25.0, -21.0],  # South Tropical Zone
            'NPR': [30.0, 70.0],  # North Polar Region
            'SPR': [-70.0, -30.0]  # South Polar Region
        }

        # these are the filters we are interested in
        self.filts = sorted(['f275w', 'f395n', 'f502n', 'f631n', 'fq889n'])  # UV, B, G, R, Methane

        self.flat = 0.06487
        self.r_eq = 71492e3  # equator radius in meters
        self.r_po = self.r_eq * (1 - self.flat)  # polar radius in meters
        self.globe = ccrs.Globe(semimajor_axis=self.r_eq, semiminor_axis=self.r_po)

        # 12e6 is the physical size 10000Km of the segment with the buffer
        self.seg_size = 12e6

        self.dataset_params = {
            'HST': {
                'lon_resolution': 3600,
                'lat_resolution': 1800,
                'lon_range': np.linspace(-180, 180, 3600),
                'lat_range': np.linspace(90, -90, 1800),
            },
            'JVH': {
                'lon_resolution': 9000,
                'lat_resolution': 4500,
                'lon_range': np.linspace(-180, 180, 9000),
                'lat_range': np.linspace(90, -90, 4500),
            }
        }

        # define the ellipsoid and datum for Jupiter
        self.ellipsoid = crs.datum.CustomEllipsoid('Jupiter', semi_major_axis=71492e3, semi_minor_axis=66854e3)
        self.primem = crs.datum.CustomPrimeMeridian(longitude=0, name='Jupiter Prime Meridian')
        self.datum = crs.datum.CustomDatum('Jupiter', ellipsoid=self.ellipsoid, prime_meridian=self.primem)

        # this is the base cylindrical projection for Jupiter
        self.jupiter_crs = crs.GeographicCRS('Jupiter', datum=self.datum)

        # Merge the process-specific args
        if args is not None:
            self.__dict__.update(args.__dict__)


    def set_dataset_params(self, key: str):
        print(f'Setting dataset/coordinate parameters for {key}')
        lon_range = self.dataset_params[key]['lon_range']
        lat_range = self.dataset_params[key]['lat_range']
        lon_resolution = self.dataset_params[key]['lon_resolution']
        lat_resolution = self.dataset_params[key]['lat_resolution']

        # Create a meshgrid for the dataset
        lon, lat = np.meshgrid(lon_range, lat_range)
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()

        # Create a KDTree for the lon/lat space
        tree = cKDTree(np.vstack((lon_flat, lat_flat)).T)

        # Store the KDTree in the class
        self.dataset_params[key]['lonlat_tree'] = tree
        self.dataset_params[key]['LON'] = lon
        self.dataset_params[key]['LAT'] = lat
        self.dataset_params[key]['coordinates'] = np.vstack((lon_flat, lat_flat)).T

        x_range = np.arange(lon_resolution)
        y_range = np.arange(lat_resolution)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        # Create a KDTree for the pixel space
        self.dataset_params[key]['pix_tree'] = cKDTree(np.vstack((x_flat, y_flat)).T)
