# Generating the global mosaic

The segments used in the model (and generated from model outputs) are in Jupiter Lambert Azimuthal Equal Area (LAEA) coordinates.
These can be stitched together to generate a global mosaic using the following script:

```bash
python3 create_mosaic.py \
    --input [path to the CSV metadata containing information about each segment] \
    --path_to_images [path to folder containing the images] \
    --subsample_rate [sample every n images to reduce the number of images used to generate the mosaic, default 1] \
    --output [output .npy file to hold the completed mosaic, do not add extension]
    --input_type [either img for using RGB images as input, UVM for using UV/Methane .npy files, or npy for multi-channel npy files]
```
