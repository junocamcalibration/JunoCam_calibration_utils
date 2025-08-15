# Post-processing the ML model output

There are three parts to processing the model outputs:

1. Merging the filters (RGB + UV + Methane)
2. Generating the GeoTIFF files from the merged segments
3. Generating the FITS output for the mosaics

## Merging the filters

There are two separate models that produce the RGB filter segments and the UV + Methane filter segments, respectively. To combine them, run the following script:

```bash
usage: merge_npys.py [-h] [-bgr_files BGR_FILES] [-uvm_files UVM_FILES] [-save_dir SAVE_DIR]

options:
  -h, --help            show this help message and exit
  -bgr_files BGR_FILES, --bgr_files BGR_FILES
                        Path to folder containing the BGR npy files
  -uvm_files UVM_FILES, --uvm_files UVM_FILES
                        Path to folder containing the UVM npy files
  -save_dir SAVE_DIR, --save_dir SAVE_DIR
                        Root directory to save the npy files
```

This will create a set of `.npy` files in the `save_dir` folder which contain the segments with the filters (in order) UV, B, G, R and Methane.

## Generating the GeoTIFF outputs

With the merged segments, the GeoTIFF can be generated using the `create_geotiff_exports` script, as follows:

```bash
usage: create_geotiff_exports.py [-h] --input_metadata INPUT_METADATA --path-to-images PATH_TO_IMAGES --path-to-output PATH_TO_OUTPUT [--max_dist MAX_DIST]

options:
  -h, --help            show this help message and exit
  --input_metadata INPUT_METADATA
                        Path to the metadata file containing the metadata for each segment
  --path-to-images PATH_TO_IMAGES
                        Path to folder containing the images
  --path-to-output PATH_TO_OUTPUT
                        Path to output folder to store GeoTIFFs
  --max_dist MAX_DIST   Distance from edge of segment from the center [in Mm]
```

The metadata file holds information about each segments and each row must contain the following columns:

- longitude [the System III longitude of the segment center in degrees]
- latitude [planetographic latitude of the segment center in degrees]
- filepath_npy [path to the `.npy` file containing the image]

The `.npy` file must be of the shape [`height`, `width`, `filter`]. This code assumes that the segments are square
with a constant resolution in the LAEA reference frame (see the `create_dataset` functions).

## Generating the FITS output

The FITS output is for the mosaics to be stored alongside their associated metadata. This can be created using the `create_FITS_export.py` script, as follows:

```bash
usage: create_FITS_export.py [-h] [-path PATH] [-bgr_path BGR_PATH] [-uv_path UV_PATH] [-methane_path METHANE_PATH] [-metadata METADATA] [-output OUTPUT] [-overwrite]

options:
  -h, --help            show this help message and exit
  -path PATH, --path PATH
                        Path to .npy file containing the mosaic
  -bgr_path BGR_PATH, --bgr_path BGR_PATH
                        Path to .npy file containing the BGR mosaic
  -uv_path UV_PATH, --uv_path UV_PATH
                        Path to .npy file containing the uv mosaic
  -methane_path METHANE_PATH, --methane_path METHANE_PATH
                        Path to .npy file containing the methane mosaic
  -metadata METADATA, --metadata METADATA
                        Path to metadata .json file
  -output OUTPUT, --output OUTPUT
                        Output FITS filename (no extension)
  -overwrite, --overwrite
                        overwrite existing FITS export
```

To run this script, you will need to either pass in one 5-channel mosaic (using the `--path` keyword) or the BGR, UV and methane mosaics separately
(using `--bgr_path`, `--uv_path` and `--methane_path`, respectively). The script also requires a JSON metadata file which contains information about the
model, the Juno perijove and the HST cycle used to generate the data. An example of this file is below, and
the FITS export will raise a warning if the following keys are not defined:

```JSON
{
    "ckpt": "junocam_calibration_C25_PJ15_npy_SB_ZonePair_fixBug",
    "modelep": 50,
    "junopj": 15,
    "HSTcyc": 25
}
```

This script will create five FITS files for each HST filter (F275W, F395N, F502N, F631N and FQ889N).
