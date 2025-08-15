# Evaluation
Estimate metrics for evaluating the quality of the calibrated images. Metrics include distribution similarity, spectrum and histogram comparisons, and perceptual metrics such as DreamSim and LPIPS.
The spectrum evaluation is zone specific. Currently this implementation loads spectra only for GRS.
Quantitative results are stored in a json file. Metric outputs represented as lists show the metric for each channel. The channel order is R, G, B, UV, M.
We provide a small dataset of calibrated GRS images [here](https://drive.google.com/drive/folders/1CoB6ZGv-PQu3uLWRe1fjGy4O0oIjJYdD?usp=sharing) that can be used as a demo for running the evaluation.

To run on the small dataset and produce results for all images:
```
python evaluate_calibration.py \
--dataroot examples/GRS_TINY_data/ \
--source_dir JunoCam/ \
--target_dir HST/ \
--fake_BGR_dir JunoCam_calibrated_BGR/ \
--fake_UVM_dir JunoCam_calibrated_UVM/ \
--results_dir examples/GRS_TINY_results/ \
--dtype npys
```
We also provide the option to load a list of image pairs along with a designated ROI by using the option --eval_pairs_file. We provide the eval_pairs.txt for the small dataset as an example. The ROIs are used only during the spectrum evaluation. 

