

import numpy as np
import os
import argparse
import torch
import cv2
from fid import FIDCalculator
from dreamsim import dreamsim
import lpips
import json
import matplotlib.pyplot as plt
from scipy.stats import chisquare, wasserstein_distance
from sklearn.neighbors import NearestNeighbors
import matplotlib.patches as patches


def read_img_lists(dataroot, folder, dtype, UVM_folder=None, domain="fake"):
    # Updated for the two-stage approach
    file_dir = dataroot+dtype+"/"+folder
    img_files = os.listdir(file_dir)
    img_files.sort()
    img_files = [file_dir+x for x in img_files]

    if UVM_folder is not None:
        UVM_dir = dataroot+dtype+"/"+UVM_folder
        uvm_files = os.listdir(UVM_dir)
        uvm_files.sort()
        uvm_files = [UVM_dir+x for x in uvm_files]

    imgs = []
    for i in range(len(img_files)):

        if dtype=="images":
            im = cv2.imread(img_files[i])
            im = im[:, :, ::-1] # Convert from BGR to RGB
        elif dtype=="npys":
            # If domain=source (JunoCam) then channels are B, G, R 
            # If domain=target then UV, B, G, R, Methane
            # If domain=fake then:
            # img_file: B, G, R
            # uvm_file: UV, Methane
            im = np.load(img_files[i])
            
            # Order to R, G, B -- (plus UV, M for fake and target cases)
            if domain=="source":
                order = np.asarray([2, 1, 0])
                im = im[:, :, order]

            elif domain=="target":
                im = np.transpose(im, (1, 2, 0)) # target files are C x H x W
                order = np.asarray([3, 2, 1, 0, 4])
                im = im[:, :, order]
            
            else:
                order = np.asarray([2, 1, 0])
                im = im[:, :, order] # Converting to RGB
                if UVM_folder is not None:
                    assert img_files[i].split('/')[-1] == uvm_files[i].split('/')[-1], "BGR and UVM files do not correspond to each other!"
                    uvm = np.load(uvm_files[i]) # H x W x 2
                    im = np.concatenate((im, uvm), axis=2) # R, G, B, UV, M
                    # Trained model uses methane*5.0 during training -- rescale the predictions
                    im[:,:,4] = im[:,:,4] / 5.0
            
            im = im * 255.0
        else:
            raise Exception('Dtype not supported!')
        imgs.append(im)

    imgs = np.stack(imgs)

    return img_files, imgs


def filter_by_zone(img_files, imgs, zone):
    files_zone, imgs_zone = [], []
    for i in range(len(img_files)):
        file_zone = img_files[i].split('/')[-1].split('_')[0]
        if file_zone==zone:
            imgs_zone.append(imgs[i,:,:,:])
            files_zone.append(img_files[i])
    imgs_zone = np.stack(imgs_zone)
    return files_zone, imgs_zone


def subsample_target(img_files, imgs, max_per_zone):
    ## Subsample the target img files (currently more than 2000)
    ## Make sure each zone has representation
    zone_list = []
    # Find the zones that exist
    for i in range(len(img_files)):
        zone = img_files[i].split('/')[-1].split('_')[0]
        zone_list.append(zone)
    zone_list = list(set(zone_list))
    
    img_files_filt, imgs_filt = [], []
    for zone in zone_list:
        files_zone, imgs_zone = filter_by_zone(img_files, imgs, zone)
        ## Take just the first max_per_zone files for now instead of randomizing (in order to reproduce the results...)
        imgs_filt.append(imgs_zone[:max_per_zone, :, :, :])
        img_files_filt += files_zone[:max_per_zone]
    
    imgs_filt = np.concatenate(imgs_filt)
    return img_files_filt, imgs_filt


def preprocess_imgs(imgs_np, target_size, device, task='dreamsim'):
    if imgs_np.shape[3] == 1:
        # case for single channel UV, M
        imgs_in = np.repeat(imgs_np, 3, axis=3)  
    elif imgs_np.shape[3] > 3:
        imgs_in = imgs_np[:,:,:,:3]
    else:
        imgs_in = imgs_np
    
    if task=='lpips':
        return imgs_in

    imgs = torch.zeros((len(imgs_in), 3, target_size[0], target_size[1]), dtype=torch.float32).to(device)

    for i in range(imgs_in.shape[0]):
        im = imgs_in[i,:,:,:].copy()
        im = im.astype(np.float32) / 255
        im = cv2.resize(im, (target_size[0], target_size[1]))
        im = torch.from_numpy(im).to(device)
        im = im.permute(2, 0, 1)
        imgs[i,:,:,:] = im
    return imgs


def dreamsim_eval(source_imgs_np, fake_imgs_np, target_imgs_np, device):
    source_imgs = preprocess_imgs(source_imgs_np, target_size=(224, 224), device=device)
    fake_imgs = preprocess_imgs(fake_imgs_np, target_size=(224, 224), device=device)
    target_imgs = preprocess_imgs(target_imgs_np, target_size=(224, 224), device=device)
    model, _ = dreamsim(pretrained=True, device=device)

    with torch.no_grad():
        # Compare the calibrated (fake) images with the original JunoCam to determine whether the semantics/structure are preserved.
        source_fake_dists = []
        for i in range(len(source_imgs)):
            source_fake_dists.append(model(source_imgs[i].unsqueeze(0), fake_imgs[i].unsqueeze(0)).cpu().numpy())
        source_fake_mean = np.mean(np.asarray(source_fake_dists))

    return source_fake_mean


def lpips_eval(source_imgs_np, fake_imgs_np, target_imgs_np, device):

    source_imgs = preprocess_imgs(source_imgs_np, target_size=None, device=device, task='lpips')
    fake_imgs = preprocess_imgs(fake_imgs_np, target_size=None, device=device, task='lpips')
    target_imgs = preprocess_imgs(target_imgs_np, target_size=None, device=device, task='lpips')
    loss_fn = lpips.LPIPS(net='alex',version='0.1').to(device)

    with torch.no_grad():
        # Compare the calibrated (fake) images with the original JunoCam to determine whether the semantics/structure are preserved.
        source_fake_dists = []
        for i in range(len(source_imgs_np)):
            source_im = lpips.im2tensor(source_imgs[i]).to(device) # RGB image from [-1,1]
            fake_im = lpips.im2tensor(fake_imgs[i]).to(device)
            source_fake_dists.append(loss_fn.forward(source_im, fake_im).cpu().numpy())
        source_fake_mean = np.mean(np.asarray(source_fake_dists))

    return source_fake_mean


def histogram_metrics(imgs_np_1, imgs_np_2, bins, h_range):
    # Estimate metrics over average histograms of given image sets
    hist_mean_1, _ = est_mean_hist(imgs_np_1, bins, h_range) # bins x channels
    hist_mean_2, _ = est_mean_hist(imgs_np_2, bins, h_range)

    # Estimate the metrics for each channel
    hist_inter, hist_emd = [], []
    for ch in range(hist_mean_1.shape[1]):
        ch_hist_1 = hist_mean_1[:,ch]
        ch_hist_2 = hist_mean_2[:,ch]

        ## Intersection (Measures overlap, higher better)
        # https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
        # The metric is normalized by the total count
        inter = np.sum(np.minimum(ch_hist_1, ch_hist_2)) / np.sum(ch_hist_2)
        hist_inter.append(inter.astype(np.float64))

        ## Earth Mover's Distance (Wasserstein, smaller better): Measures the work needed to transform one histogram into another. 
        ch_hist_1_norm = ch_hist_1 / np.sum(ch_hist_1)
        ch_hist_2_norm = ch_hist_2 / np.sum(ch_hist_2)
        emd = wasserstein_distance(ch_hist_1_norm, ch_hist_2_norm)
        hist_emd.append(emd.astype(np.float64))

    return hist_inter, hist_emd


def est_mean_hist(imgs_np, bins, h_range):
    ## Estimate average histograms of given set of imgs
    n_channels = imgs_np.shape[3]
    hist_mean = np.zeros((bins,n_channels), dtype=np.float32)
    for i in range(len(imgs_np)):
        for ch in range(n_channels):
            im = imgs_np[i, :, :, ch].astype(np.float32) / 255.0
            hist, bin_edges = np.histogram(im.flat, bins=bins, range=h_range)
            hist_mean[:,ch] += hist    
    hist_mean = hist_mean / len(imgs_np)
    return hist_mean, bin_edges


def plot_histograms(imgs_np, bins, h_range, savepath):
    ## Plot average histograms of given set of imgs
    color_dict = {0:'red', 1:'green', 2:'blue', 3:'black', 4:'magenta'} # 3 is UV, 4 is Methane
    hist_mean, bin_edges = est_mean_hist(imgs_np, bins, h_range)

    # choose title
    if "fake" in savepath:
        title = "Model Calibrated JunoCam"
        xlabel = "Pseudo I/F"
    elif "source" in savepath:
        title = "JunoCam"
        xlabel = "Pixel Intensities"
    elif "target" in savepath:
        title = "HST"
        xlabel = "I/F"

    f, axs = plt.subplots(1, 1, figsize=(12, 10))
    for ch in range(imgs_np.shape[3]): # n_channels
        axs.plot(bin_edges[:-1], hist_mean[:,ch], color=color_dict[ch], linewidth=4)

    axs.set_title(title, fontsize=30)
    axs.set_xlabel(xlabel, fontsize=30)
    axs.set_ylim(0, 30000)
    axs.set_xlim(0, 1)
    axs.legend(["Red", "Green", "Blue", "UV", "Methane"], fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  
    plt.tight_layout()
    plt.savefig(savepath+".png")
    plt.close()


def get_mean_IF(imgs_np, window_size):
    N, H, W, C = imgs_np.shape
    IF = np.zeros((N, C), dtype=np.float32)
    # Pre-select windows
    x = np.random.randint(low=0, high=H-window_size-1, size=N)
    y = np.random.randint(low=0, high=W-window_size-1, size=N)
    for i in range(N):
        im = imgs_np[i,:,:,:].astype(np.float32) / 255.0
        for ch in range(C):
            IF[i,ch] = np.mean( im[x[i]:x[i]+window_size, y[i]:y[i]+window_size, ch] )
    IF = np.mean(IF, axis=0) # Mean across images -- 1 x 3
    return IF


def get_IF_from_patch(imgs_np, patch):
    # Estimate IF of a single image over a given patch
    # Patch is a bounding box: left, top, width, height
    left, top, width, height = patch
    _, H, W, C = imgs_np.shape
    IF = np.zeros((C), dtype=np.float32)
    for ch in range(C):
        im = imgs_np[0,:,:,ch].astype(np.float32) / 255.0
        IF[ch] = np.mean(im[top:top+height, left:left+width])
    return IF


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


def plot_spectra(spectrum_dict, wl_real, IF_fake, IF_real, results_dir, IF_juno=None):

    '''
    Note:
    The karkoschka spectrum is **disk-averaged** so comparing it to small regions unaffected by limb-darkening is sort of apples to oranges. 
    Might still be a useful reference since it's the community standard for checking if things are looking realistic.
    '''
    spectrum_labels = {'vims': 'GRS, Cassini VIMS - 2000', # (Carlson et al. 2016)
                       'braude': 'GRS, VLT/MUSE - 2018', # (Braude et al. 2020)
                       'karkoschka': 'Disk-averaged spectrum - 1998',
                       'naic': 'EZ spectrum, APO/NAIC - 2019'}

    if IF_juno is not None:
        n_colors = 7
    else:
        n_colors = 6

    colors = plt.cm.plasma(np.linspace(0,0.9,n_colors))
    fig, ax = plt.subplots(dpi=100)
    fig.set_size_inches(10,6)

    count=0
    for sp_k, sp_v in spectrum_dict.items():
        plt.plot(sp_v[:,0], sp_v[:,1], label=spectrum_labels[sp_k], linewidth=2, color=colors[count])
        count+=1

    plt.plot(wl_real,IF_real,'.',label='HST - 2018',markersize=15,color=colors[-2],)
    plt.plot(wl_real,IF_fake,'^',label='Calibrated JunoCam',markersize=12,color=colors[-1])

    if IF_juno is not None:
        plt.plot(wl_real[:3],IF_juno,'*',label='JunoCam',markersize=7,color=colors[-3])

    plt.title('Reflectance Spectra',size=22)
    plt.ylabel('I/F (Absolute Reflectivity)',size=20)
    plt.xlabel('Wavelength (nm)',size=20)
    # change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=17)
    plt.legend(fontsize=15, loc="lower center")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(results_dir+'spectra.png') # option to save the figure


def spectrum_eval(fake_imgs_np, fake_img_files, target_imgs_np, target_img_files, source_imgs_np, source_img_files, results_dir, patch, window_size):    
    ## Compare spectra of calibrated images to HST and other sources
    ## The quantitative evaluation is carried-out by sampling a random box from each image and using their averages

    wl_real = [631, 502, 395, 275, 889] # R, G, B, UV, Methane

    if patch is not None:
        # Case when evaluating spectra on a single image over a pre-determined patch
        IF_fake = get_IF_from_patch(fake_imgs_np, patch)
        IF_real = get_IF_from_patch(target_imgs_np, patch)
        IF_juno = get_IF_from_patch(source_imgs_np, patch)
    else:
        IF_fake = get_mean_IF(fake_imgs_np, window_size)
        IF_real = get_mean_IF(target_imgs_np, window_size)
        IF_juno = get_mean_IF(source_imgs_np, window_size)
    
    IF_dist = np.linalg.norm(IF_fake-IF_real)


    ## Find distance of IF_fake to every spectrum for the available wavelengths
    # braude_2020 and NAIC start from wavelength ~470
    braude_2020 = np.loadtxt('spectra/muse_grs_spectrum') # VLT/MUSE GRS spectrum from 2018
    vims = np.loadtxt('spectra/vims_grs_carlson2016') # Cassini/VIMS GRS spectrum from 2000
    NAIC = np.loadtxt('spectra/mu10_pj19_IF_NAIC') # APO/NAIC spectrum of EZ during reddening event in 2019
    karkoschka_init = np.loadtxt('spectra/karkoschka') # disk-averaged spectrum from ~1998
    karkoschka = np.zeros((karkoschka_init.shape[0]-150, 2))
    karkoschka[:,0] = karkoschka_init[:-150,0]
    karkoschka[:,1] = karkoschka_init[:-150,3]
    
    spectrum_dict = {'braude': braude_2020,
                     'vims': vims,
                     'karkoschka': karkoschka,
                     'naic':NAIC
                     }

    # for each spectrum find the IF that corresponds to the closest wavelength
    # we should get an IF value for each channel for each spectrum
    spectra_IF = {}
    dist_spectra = {}
    for sp_k, sp_v in spectrum_dict.items():
        sp_wl = sp_v[:,0].reshape(-1,1)
        IF = []
        for ch in range(len(wl_real)):
            wl = np.asarray([[wl_real[ch]]])
            dist, idx = nearest_neighbor(wl, sp_wl)
            IF.append(sp_v[idx[0],1])
        spectra_IF[sp_k] = np.asarray(IF)
        dist_spectra[sp_k] = np.linalg.norm(IF_fake-np.asarray(IF)).astype(np.float64)
    
    dist_spectra['HST'] = IF_dist.astype(np.float64)

    ## Plot all spectra along with calibrated and HST IF
    plot_spectra(spectrum_dict, wl_real, IF_fake, IF_real, results_dir)

    return dist_spectra


def run_evaluation(source_img_files,
                   target_img_files,
                   fake_img_files,
                   source_imgs_np,
                   target_imgs_np,
                   fake_imgs_np,
                   args,
                   results_dir,
                   do_fid=True,
                   patch=None):

    res_dict = {}

    ### Comparison to different spectra
    print("Doing spectra comparison...")
    dist_spectra = spectrum_eval(fake_imgs_np, fake_img_files, target_imgs_np, target_img_files, 
                                        source_imgs_np, source_img_files, results_dir, patch=patch, window_size=args.spectrum_win)
    res_dict.update(dist_spectra)

    
    if do_fid:
        ### Distribution similarity (FID and Histogram distance)
        print("Estimating FID distance...")
        print("Preprocessing images...")
        fake_imgs = preprocess_imgs(fake_imgs_np, target_size=(299, 299), device=args.device)
        target_imgs = preprocess_imgs(target_imgs_np, target_size=(299, 299), device=args.device)
        fid_calculator = FIDCalculator(device=args.device)
        print("Calculating FID...")
        fid_score = fid_calculator.calculate_fid(real_images=target_imgs, generated_images=fake_imgs, batch_size=1)

        # Calculate FID for UV and methane
        # Replicate single channels when preprocessing imgs
        fake_imgs_UV, fake_imgs_M = np.expand_dims(fake_imgs_np[:,:,:,3], axis=3), np.expand_dims(fake_imgs_np[:,:,:,4], axis=3)
        target_imgs_UV, target_imgs_M = np.expand_dims(target_imgs_np[:,:,:,3], axis=3), np.expand_dims(target_imgs_np[:,:,:,4], axis=3)

        fake_imgs_UV = preprocess_imgs(fake_imgs_UV, target_size=(299, 299), device=args.device)
        target_imgs_UV = preprocess_imgs(target_imgs_UV, target_size=(299, 299), device=args.device)
        fid_score_UV = fid_calculator.calculate_fid(real_images=target_imgs_UV, generated_images=fake_imgs_UV, batch_size=1)

        fake_imgs_M = preprocess_imgs(fake_imgs_M, target_size=(299, 299), device=args.device)
        target_imgs_M = preprocess_imgs(target_imgs_M, target_size=(299, 299), device=args.device)
        fid_score_M = fid_calculator.calculate_fid(real_images=target_imgs_M, generated_images=fake_imgs_M, batch_size=1)
        res_dict['fid_score'] = fid_score.astype(np.float64)
        res_dict['fid_score_UV'] = fid_score_UV.astype(np.float64)
        res_dict['fid_score_M'] = fid_score_M.astype(np.float64)

        print("Calculate FID between source and target...")
        source_imgs = preprocess_imgs(source_imgs_np, target_size=(299, 299), device=args.device)
        fid_score_source = fid_calculator.calculate_fid(real_images=target_imgs, generated_images=source_imgs, batch_size=1)
        res_dict['fid_score_source'] = fid_score_source.astype(np.float64)



    ### Histogram metrics (EMD, Intersection) and plots
    print("Estimating histogram metrics...")
    print("Between fake and target...")
    inter_fake_target, emd_fake_target = histogram_metrics(fake_imgs_np, target_imgs_np, bins=args.hist_bins, h_range=(0,1))
    print("Between source and target...")
    inter_source_target, emd_source_target = histogram_metrics(source_imgs_np, target_imgs_np[:,:,:,:3], bins=args.hist_bins, h_range=(0,1))
    res_dict['inter_fake_target'] = inter_fake_target
    res_dict['inter_source_target'] = inter_source_target
    res_dict['emd_fake_target'] = emd_fake_target
    res_dict['emd_source_target'] = emd_source_target

    print("Plotting histograms...")
    plot_histograms(fake_imgs_np, bins=args.hist_bins, h_range=(0,1), savepath=results_dir+"fake_hist")
    plot_histograms(target_imgs_np, bins=args.hist_bins, h_range=(0,1), savepath=results_dir+"target_hist")
    plot_histograms(source_imgs_np, bins=args.hist_bins, h_range=(0,1), savepath=results_dir+"source_hist")


    ### Perceptual similarity (DreamSim and LPIPS)
    ## Compare the calibrated (fake) images with the original JunoCam to determine whether the semantics/structure are preserved.
    print("Estimating perceptual similarity metrics...")
    source_fake_mean_dreamsim = dreamsim_eval(source_imgs_np, fake_imgs_np, target_imgs_np, args.device)
    source_fake_mean_lpips = lpips_eval(source_imgs_np, fake_imgs_np, target_imgs_np, args.device)
    res_dict['source_fake_mean_lpips'] = source_fake_mean_lpips.astype(np.float64)
    res_dict['source_fake_mean_dreamsim'] = source_fake_mean_dreamsim.astype(np.float64)

    # Write all results to disk
    with open(results_dir+"results.json", "w") as outfile:
        json.dump(res_dict, outfile, indent=4)


def save_patched_image(img, patch, path):
    left, top, width, height = patch
    fig, ax = plt.subplots(1,1, figsize=(5,5),dpi=100)
    ax.imshow(img/255.0)
    ax.add_patch(patches.Rectangle((left, top), width, height, edgecolor="blue", facecolor='none', linewidth=2) )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path+"_patch.png", bbox_inches='tight', pad_inches = 0)


'''
## Example cmd using the TINY GRS npy dataset
python evaluate_calibration.py \
--dataroot examples/GRS_TINY_data/ \
--source_dir JunoCam/ \
--target_dir HST/ \
--fake_BGR_dir JunoCam_calibrated_BGR/ \
--fake_UVM_dir JunoCam_calibrated_UVM/ \
--results_dir examples/GRS_TINY_results/ \
--dtype npys \
--eval_pairs_file eval_pairs.txt
'''


if __name__ == '__main__':

    '''
    Notes:
    The spectrum eval is zone specific. Currently this implementation loads spectra only for GRS.
    Metric outputs represented as lists show the metric for each channel. The channel order is R, G, B, UV, M
    '''


    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--dataroot", type=str, default="examples/", required=False)
    parser.add_argument("--fake_BGR_dir", type=str, default="examples/fake_imgs/", help="folder containing the JunoCam calibrated images", required=False)
    parser.add_argument("--fake_UVM_dir", type=str, default=None, help="folder containing the JunoCam calibrated UV, Methane channels", required=False)
    parser.add_argument("--source_dir", type=str, default="examples/source_imgs/", help="folder containing the original JunoCam images", required=False)
    parser.add_argument("--target_dir", type=str, default="examples/target_imgs/", help="folder containing the HST reference images", required=False)
    parser.add_argument("--eval_pairs_file", type=str, default=None, help="If provided, run evaluation on individual pre-selected pairs", required=False)
    parser.add_argument("--device", type=str, default="cuda", choices=['cpu', 'cuda'], required=False)
    parser.add_argument("--dtype", type=str, default="npys", choices=['images', 'npys'], required=False)
    parser.add_argument("--results_dir", type=str, default="examples/results/", required=False)
    parser.add_argument("--hist_bins", type=int, default=31, required=False)
    parser.add_argument("--spectrum_win", type=int, default=25, help='Window size when sampling spectra', required=False)
    args = parser.parse_args()

    print("Collecting all files...")
    # Img files are returned with channel order: R, G, B, UV, M
    fake_img_files, fake_imgs_np = read_img_lists(args.dataroot, args.fake_BGR_dir, args.dtype, UVM_folder=args.fake_UVM_dir, domain="fake") # N x H x W x 5
    source_img_files, source_imgs_np = read_img_lists(args.dataroot, args.source_dir, args.dtype, domain="source") # N x H x W x 3
    target_img_files, target_imgs_np = read_img_lists(args.dataroot, args.target_dir, args.dtype, domain="target") # N x H x W x 5

    ## Subsample the target img files
    if len(target_img_files) > 1000:
        print("Subsampling target files...")
        target_img_files, target_imgs_np = subsample_target(target_img_files, target_imgs_np, max_per_zone=200)


    if args.eval_pairs_file is not None:
        # Run evaluation only on the pre-selected pairs (individually for each pair)
        count=0
        with open(args.dataroot+args.eval_pairs_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                hst_id = line.split(' ')[0]
                juno_id = line.split(' ')[1]
                patch = ( int(line.split(' ')[2]), int(line.split(' ')[3]), int(line.split(' ')[4]), int(line.split(' ')[5]) )
                
                # Find the target file in the list of loaded images
                target_file = args.dataroot+args.dtype+"/"+args.target_dir+hst_id+".npy"
                index = list(target_img_files).index(target_file)
                target_img = target_imgs_np[index,:,:,:][None]
                # Find the source file
                source_file = args.dataroot+args.dtype+"/"+args.source_dir+juno_id+".npy"
                index = list(source_img_files).index(source_file)
                source_img = source_imgs_np[index,:,:,:][None]
                # Find the fake file
                fake_file = args.dataroot+args.dtype+"/"+args.fake_BGR_dir+juno_id+".npy"
                index = list(fake_img_files).index(fake_file)
                fake_img = fake_imgs_np[index,:,:,:][None]

                results_dir = args.results_dir + str(count) + "_" + hst_id + "_" + juno_id + "/"
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                ## Save triplet of images in the results dir (including UV,M)
                cv2.imwrite(results_dir+juno_id+".png", source_img[0,:,:,::-1])
                fake_vis, fake_UV, fake_M = fake_img[0,:,:,:3], fake_img[0,:,:,3], fake_img[0,:,:,4]
                cv2.imwrite(results_dir+juno_id+"_calibrated_RGB.png", fake_vis[:,:,::-1])
                cv2.imwrite(results_dir+juno_id+"_calibrated_UV.png", fake_UV)
                cv2.imwrite(results_dir+juno_id+"_calibrated_M.png", fake_M*5.0) # apply Methane scaling for viz

                target_vis, target_UV, target_M = target_img[0,:,:,:3], target_img[0,:,:,3], target_img[0,:,:,4]
                cv2.imwrite(results_dir+hst_id+"_RGB.png", target_vis[:,:,::-1])
                cv2.imwrite(results_dir+hst_id+"_UV.png", target_UV)
                cv2.imwrite(results_dir+hst_id+"_M.png", target_M*5.0)

                ## Save the images with the patch used for spectra evaluation
                save_patched_image(fake_vis, patch, path=results_dir+juno_id+"_calibrated_RGB")
                save_patched_image(target_vis, patch, path=results_dir+hst_id+"_RGB")
                save_patched_image(source_img[0,:,:,:], patch, path=results_dir+juno_id)

                # Run eval on a single example
                run_evaluation(source_img_files=np.asarray([source_file]),
                                target_img_files=np.asarray([target_file]),
                                fake_img_files=np.asarray([fake_file]),
                                source_imgs_np=source_img,
                                target_imgs_np=target_img,
                                fake_imgs_np=fake_img,
                                args=args,
                                results_dir=results_dir,
                                do_fid=False,
                                patch=patch)
                count+=1
                
    
    else:
        # Run on entire dataset
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        run_evaluation(source_img_files,
                        target_img_files,
                        fake_img_files,
                        source_imgs_np,
                        target_imgs_np,
                        fake_imgs_np,
                        args,
                        results_dir=args.results_dir)
