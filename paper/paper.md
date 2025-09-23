---
title: 'EyeDentify3D: A Python package for gaze behavior classification of 3D eye-tracking data'
tags:
  - Python
  - Eye-tracking
  - Classification
  - Fixation
  - Saccade
  - Smooth pursuit
  - Gaze behavior
  - Head movements
  - 3D space
authors:
  - name: Eve Charbonneau
    orcid: 0000-0002-9215-3885
    affiliation: "1, 2"
    corresponding: true
  - name: Thomas Romeas
    orcid: 0000-0002-3298-5719
    affiliation: "1, 2"
  - name: Maxime Trempe
    orcid: 0000-0002-3477-9783
    affiliation: 3
affiliations:
 - name: Université de Montréal, Canada
   index: 1
 - name: Institut national du sport du Québec, Canada
   index: 2
 - name: Bishop's University, Canada
   index: 3
date: 25 August 2025
bibliography: paper.bib
---

# Summary

With the technological advances of portable eye-tracking technologies, researchers can now place participants in 
real-world settings and measure their head and eye orientation to get the gaze orientation in 3D space. 
Although more ecological, real-world eye-tracking data must be post-processed to extract interpretable gaze behaviors 
(e.g., fixations, saccades, smooth pursuits, visual scanning) than in the raw gaze data itself. 
However, most algorithms for classifying gaze behaviour were developed for screen-based eye-tracking data (2D) acquired 
with the head fixed in space.
These algorithms are ill-suited for real-world portable eye-tracking data (3D) where the gaze-vector origin and endpoint 
are not constrained in space. 
To address this gap, we developed `EyeDentify3D`, an automated and modulable pipeline for analyzing real-world 
eye-tracking data.

# Statement of need

`EyeDentify3D` is a Python package for extracting gaze behavior from portable eye-tracking data. 
It was designed to:
1. Extract and interpret data from various eye-tracking systems (e.g., HTC Vive Pro, Pupil Invisible, 
Meta Quest Pro, Pico Neo 3 Pro Eye).
2. Provide a simple user interface, where only a few lines of code are needed to extract the desired 
gaze behaviors.
3. Enable visual inspection of the classification results.

`EyeDentify3D` was designed to be used by both neuroscience and sport researchers. It has already been used in sport 
psychology to analyze the gaze behavior of basketball players [@Trempe:2025], baseball players, and boxers. 
The toolbox is distributed openly to help researchers more reliably identify and analyze gaze behaviors in real-world 
scenarios hoping that it will help improving understanding of visual strategy mechanisms.
We cannot wait for our toolbox to be used to analyze gaze behavior in a variety of contexts!

# Background

Each trial recorded during an experiment is analyzed frame-by-frame; for each frame, the pipeline applies a classified 
based on the following criteria:
1. **Invalid**: The eye-tracker has declared having low confidence in the gaze orientation measurement (this often happen 
when the eyes are closed, the eye orientation is outside the eye-tracker's measurement range, or if the eye-tracker was 
not positioned correctly on the participant).
2. **Blink**: The eye openness is below the threshold [@Chen:2021].
3. **Saccade**: Two criteria must be met to detect a saccade. 1) The eye movement must be faster than a dynamics 
threshold. The dynamics threshold is determined using a rolling median over a user defined window size. 2) The eye 
movement acceleration must be larger than a user defined threshold for at least two frames. This ensures that the eyes 
are moving rapidly between two targets with an acceleration when leaving the first target and a deceleration when 
arriving to the second target [@Van:1987].
4. **Visual scanning**: The gaze (head + eyes) velocity is larger than a threshold [@Mcguckian:2020]. Visual scanning should usually be 
identified after saccades as visual scanning behavior would also present high eye velocity.
5. **Inter-saccadic interval**: Our inter-saccadic interval classification was adapted from 
@Larsson:2015 implementation designed for screen-based eye-tracking data. Intersaccadic intervals lasting more than a 
duration threshold are identified between the already identified frames. These intervals are subdivided into windows of 
a user defined size. Each window is classified as either coherent or incoherent based on the gaze movement (moving in a 
consistent direction or not). Adjacent coherent and incoherent windows are merged together to form segments. Then, 
these segments are further classified as either 
6. **fixation** or **smooth pursuit** behaviors based on the four criteria described in @Larsson:2015:
   * Dispersion: $p_D < \eta_D$
   * Consistent direction: $p_{CD} > \eta_{CD}$
   * Positional displacement: $p_{PD} > \eta_{PD}$
   * Spatial range: $p_R > \eta_{maxFix}$
    
All behaviors are mutually exclusive (except for invalid and blink that can happen simultaneously). For example, a frame 
cannot be classified as both a saccade and a fixation. Thus, the order of the identification is important as the first 
behavior identified will take precedence over the others. 

More details on the definition of events and how they are identified can be found in the 
[documentation](https://evecharbie.github.io/EyeDentify3d).

Finally, `EyeDentify3D` enables visualisation of the classified gaze data and extraction/export of metrics related to 
the behaviors (e.g., duration, time ratio spent in each behavior, number of occurrences, saccade amplitude, smooth 
pursuit trajectory length, etc.).

# Note on the implementation
We believe that the choices made in `EyeDentify3D` are the most suitable for the analysis of gaze behavior in 3D space 
(especially in sporting context). However, we are very open to implement other identification methods in the future.

# Acknowledgements
This project was supported by a Research and Creative Activity grant from Bishop’s University.

# Conflict of interest
The authors declare no conflict of interest.

# Declaration of generative AI
During the preparation of this work the developer used ChatGPT, Claude, and Copilot to speed up development and enhance 
code clarity. Aider and Claude were also used to write tests. After using these tools/services, the developer reviewed 
and edited the content as needed and takes full responsibility for the content of the repository.

# References
