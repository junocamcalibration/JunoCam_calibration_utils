
import numpy as np
import os
import argparse
from params import Params
import shutil
import random
from astropy.io import ascii
from astropy.table import Table

'''
Copy and organize the generated HST JunoCam data for directly training with UNSB
This should be ran after create_segments.py
'''

def collect_files(segments_dir, types):
    zone_dirs = os.listdir(segments_dir)

    # initialize dict
    files_dict = {}
    for data_type in types:
        files_dict[data_type] = {}

    for data_type in types:
        file_list = []

        for zd in zone_dirs:
            files_path = segments_dir + zd + '/' + data_type + "/"
            zd_file_list = os.listdir(files_path)
            zd_file_list = [ files_path+x for x in zd_file_list ]
            file_list = zd_file_list
            files_dict[data_type][zd] = file_list
    
    # Collect all metadata files
    domain = segments_dir.split('/')[-2]
    metadata_files = []
    for zd in zone_dirs:
        metadata_files.append(segments_dir+zd+"/"+domain+"_metadata.csv")

    return files_dict, metadata_files


def merge_metadata(meta_files, HST=False):
    # Merge metadata files from different zones
    meta_files.sort()
    if HST:
        table_headers = {'index':[], 'latitude':[], 'longitude':[], 'rotation':[], 'cycle':[], 'filepath_npy':[], 'filepath_img':[]}
        dtypeList = [int, float, float, str, int, str, str]
    else:
        table_headers = {'index':[], 'latitude':[], 'longitude':[], 'perijove':[], 'filepath_npy':[], 'filepath_img':[]}
        dtypeList = [int, float, float, int, str, str]
    
    index_count=0
    for mfile in meta_files:
        meta = ascii.read(mfile, format='csv')

        for k in meta.keys():
            if (k == 'index') and (index_count>0):
                new_index = [x+index_count for x in list(meta[k])]
                table_headers[k] += new_index
                index_count += len(meta[k])
            elif (k == 'index') and (index_count==0):
                index_count += len(meta[k])
                table_headers[k] += list(meta[k])
            else:
                table_headers[k] += list(meta[k])

    n_entries = len(table_headers['index'])
    # Write the merged csv
    table = Table(names=list(table_headers.keys()), dtype=dtypeList)
    for i in range(n_entries):
        row = []
        for k in table_headers.keys():
            row.append(table_headers[k][i])
        table.add_row(row)
    return table



def copy_files(files, dataset_dir):
    for i in range(len(files)):
        source_file = files[i]
        dest_file = dataset_dir + source_file.split('/')[-1]
        print("Copying", source_file, "to", dest_file)
        shutil.copy(source_file, dest_file)


def copy_to_unsb(params, data_type, files_dict, table, data_id="A"):
    dataset_root = params.unsb_root_dir + "datasets/" + params.dataset_name + "_" + data_type + "/"
    dataset_train = dataset_root + "train" + data_id + "/"
    if not os.path.isdir(dataset_train):
        os.makedirs(dataset_train)

    table.write(dataset_root+data_id+'_metadata.csv', format='csv', overwrite=True)

    # Collect all files from the different zones
    files_train, files_test = [], []
    for zone in files_dict[data_type].keys():
        file_list = files_dict[data_type][zone]
        # divide list to train and test
        random.shuffle(file_list)

        if params.test_set_mode != "none":
            if params.test_set_mode == "disjoint":
                n_test = int(len(file_list)*params.testing_per) # number of test files
                files_test += file_list[-n_test:]
                files_train += file_list[:len(file_list)-n_test]
            else: # full option, save all images in train and test set
                files_train += file_list # keep all images in training, regardless of test set
                files_test += file_list
        else: # no test set saved
            files_train += file_list
            files_test = []

    copy_files(files_train, dataset_train)

    if params.test_set_mode != "none":
        dataset_test = dataset_root + "test" + data_id + "/"
        if not os.path.isdir(dataset_test):
            os.makedirs(dataset_test)
        copy_files(files_test, dataset_test)


def copy_to_unsb_per_zone(params, data_type, files_dict, meta_files, data_id="A"):
    dataset_root = params.unsb_root_dir + "datasets/" + params.dataset_name + "_" + data_type + "_per_zone/"

    # Collect all files from the different zones and store them separately
    for zone in files_dict[data_type].keys():

        dataset_train = dataset_root + "train" + data_id + "/" + zone + "/"
        if not os.path.isdir(dataset_train):
            os.makedirs(dataset_train)

        # Copy the metadata file
        mfile = [x for x in meta_files if zone in x][0]
        shutil.copy(mfile, dataset_root+data_id+"_"+zone+"_metadata.csv")

        file_list = files_dict[data_type][zone]
        # divide list to train and test
        random.shuffle(file_list)

        if params.test_set_mode != "none":
            if params.test_set_mode == "disjoint":
                n_test = int(len(file_list)*params.testing_per) # number of test files
                files_test = file_list[-n_test:]
                files_train = file_list[:len(file_list)-n_test]
            else: # full option, save all images in train and test set
                files_train += file_list # keep all images in training, regardless of test set
                files_test += file_list
        else: # no test set saved
            files_train += file_list
            files_test = []

        
        copy_files(files_train, dataset_train)
        
        if params.test_set_mode != "none":
            dataset_test = dataset_root + "test" + data_id + "/" + zone + "/"
            if not os.path.isdir(dataset_test):
                os.makedirs(dataset_test)
            copy_files(files_test, dataset_test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--unsb_root_dir", type=str, default="../../UNSB/", required=False)
    parser.add_argument("--dataset_name", type=str, default="junocam_calibration", required=False)
    parser.add_argument('--do_per_zone', default=False, action='store_true', help='Organize training/testing per zone')
    parser.add_argument("--test_set_mode", type=str, default="none", choices=['disjoint', 'full', 'none'], help="none creates only the training set", required=False)
    parser.add_argument("--testing_per", type=float, default=0.1, help="What percentage of data to use for testing", required=False)
    parser.add_argument('--types', nargs='+', default=["png", "npy"], help='Which data types to copy. Can do both images and npys')
    args = parser.parse_args()

    params = Params(args)

    if "png" in params.types:
        hst_types_list = params.types + ["png_UV", "png_M"]
    else:
        hst_types_list = params.types

    jvh_files_dict, jvh_metadata_files = collect_files(params.jvh_segments_dir, params.types)
    hst_files_dict, hst_metadata_files = collect_files(params.hst_segments_dir, hst_types_list)

    # copy HST dataset to unsb
    for data_type in hst_types_list: #params.types:
        if params.do_per_zone:
            # copy files under train/test in separaty zone folders
            copy_to_unsb_per_zone(params, data_type, hst_files_dict, hst_metadata_files, data_id="B")
        else:
            hst_table = merge_metadata(meta_files=hst_metadata_files, HST=True)
            copy_to_unsb(params, data_type, hst_files_dict, hst_table, data_id="B")


    # copy JVH dataset to unsb
    for data_type in params.types:
        if params.do_per_zone:
            # copy files under train/test in separaty zone folders
            copy_to_unsb_per_zone(params, data_type, jvh_files_dict, jvh_metadata_files, data_id="A")
        else:
            jvh_table = merge_metadata(meta_files=jvh_metadata_files)
            copy_to_unsb(params, data_type, jvh_files_dict, jvh_table, data_id="A")
