# JunoCam Calibration Utils

Set of tools to support the photometric calibration of JunoCam observations.
We provide several modules for handling data before and after training the image-to-image translation model. Details on how to use each module are provided within the module.
In summary:
- **Create dataset:** Generate an unpaired JunoCam/HST segment dataset from flattened maps for training the image-to-image translation model.
- **Evaluation:** Provide metrics to evaluate the quality of the calibrated images such as distribution similarity (FID), spectrum and histogram comparisons, and perceptual metrics such as DreamSim and LPIPS.
- **Mosaic:** Create mosaics (flattened maps) of Jupiter using the calibrated images.
- **Postprocess**: Produce PDS compliant data products for archiving.

## Dependencies
The dependencies can be installed using `pip` or `uv`:
```bash
uv sync
```

There are optional dependencies for each component (`dataset` for creating the database, `eval` for the evaluation scripts, and `export` for the mosaic and post-processing scripts). 
You can install specific components using:
```bash
uv sync --extra [ dataset | eval | export ]
```
or all of them using:
```bash
uv sync --all-extras
```

Similarly for `pip`:
```bash
python3 -m pip install ".[ dataset | eval | export ]"
```

## Acknowledgements
This work was funded by NASA ROSES New Frontiers Data Analysis Program (NFDAP) and conducted for the project "Disentangling Jupiter’s Complex Atmospheric Processes Through the Application of Machine Learning Methods to JunoCam Data" (80NM0018F0612).


"Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology. This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons."