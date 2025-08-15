# Create dataset
Instructions on generating the unpaired JunoCam-HST segment dataset used for training an image-to-image translation method for photometric calibration of JunoCam observations.
The segments are sampled from mosaics that have already been processed using raw observations (e.g., OPAL for HST). For the pre-processing of the raw JunoCam images see https://github.com/ramanakumars/JunoCamProjection/.

### Data
Run the get_links.sh in data/ in order to download the HST Cycles (fits files) from OPAL https://archive.stsci.edu/hlsp/opal.
Then download the JunoCam maps from each perijove from: https://github.com/ramanakumars/JuDE/tree/main/backend/PJimgs and store them under data/JVH_PJimgs.
See the header of create_databases.py for the expected folder structure.

### Generating the unpaired segments dataset for HST/JunoCam
The process first creates NC databases before saving the individual segments as npy or png files. 
To create the NC databases for all zones and both HST and JunoCam:
```
sh create_nc.sh
```
This can take several hours. Alternatively, if interested in a specific perijove/zone combination from JunoCam:
```
python create_databases.py --perijove 27 --zones GRS
```
or if interested in a specific cycle/zone combination from HST:
```
python create_databases.py --cycle 28 --zones SEB --nside 16
```
For the latitude ranges used for each zone please see params.py. We note that we use a sparser sampling of the HST segments (--nside 16 instead of the default 32) to avoid having a significant imbalance between the JunoCam and HST number of segments.

After the generation of the NC databases, the segments and their metadata for all perijoves/cycles and zones can be created: 
```
sh create_seg.sh
```
Alternatively, if interested for a specific zone:
```
python create_segments.py --save_imgs --zone GRS
```
By default, the segments are created in npy format with channel sequence UV, B, G, R, Methane for HST, and B, G, R for JunoCam.
Finally, to organize the data in the unpaired image-to-image translation dataset: 
```
python copy_data_for_unsb.py --unsb_root_dir ../../UNSB/
```
The resulting dataset is copied directly under the dataset/ folder and can be used directly to train UNSB (https://github.com/cyclomon/unsb).