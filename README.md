# Gaze classification algorithm

This repo is a modified implementation of the gaze classification algorithm presented in (Larsson et al., 2015)[https://doi.org/10.1016/j.bspc.2014.12.008].
It allows to classify the 3D gaze behavior (head + eye movements) into blinks, fixations, saccades, smooth pursuit, and visual scanning events.

This code was developed to treat data from the Tobii eye-tracker embedded in the HTC Vise Pro VR headset.
It was originally used to analyze the gaze behavior of basket players using an occlusion paradigm.

## Installation
You can use the "environment.yml" file to create a conda environment with all the necessary packages.
```bash
conda env create -f environment.yml
```
If you intend on using this code, please do not hesitate to contact (eve.charbonneau.1@umontreal.ca)[mailto:eve.charbonneau.1@umontreal.ca] for any questions or comments.
