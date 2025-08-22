---
title: 'EyeDentify3D: A Python package for gaze behavior classification'
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

With the technological advances of eye-tracking technologies, researchers can now place participants in real-world 
settings and measure their gaze orientation in 3D space. Although more ecological, these data are complex to analyze.
Indeed, most researchers are more interested in the extraction of gaze behaviors (e.g., fixations, saccades, 
smooth pursuits, visual scanning) than in the raw gaze data itself. However, most existing algorithms for
gaze behavior classification were developed for 2D screen-based eye-tracking data and are not suitable for 3D data. 
To address this gap, we developed `EyeDentify3D`.

# Statement of need

`EyeDentify3D` is a Python package for extracting gaze behavior from 3D eye-tracking data. 
It was designed to:
1. Extract and interpret data from various eye-tracking systems (e.g., HTC Vive Pro, Pupil Invisible, 
Meta Quest Pro, Pico Neo 3 Pro Eye).
2. Provide a simple user interface, where only a few lines of code are needed to extract the desired 
gaze behaviors.
3. Offer the possibility to inspect visually the results of the classification.

`EyeDentify3D` was designed to be used by both neuroscientists, sport scientists. It has already been used to analyze 
the gaze behavior of basketball players [@Trempe:2025], baseball player, and boxers. We cannot wait for our toolbox to 
be used to analyze gaze behavior in other contexts!

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
