CMFD_PM_video version 0.1
Matlab functions for the detection and localization of copy-move forgeries with the technique described in 
"A PatchMatch-based Dense-field Algorithm for Video Copy-Move Detection and Localization", 
written by L. D'Amiano, D. Cozzolino, G. Poggi and L. Verdoliva, in peer review, 2017.
Please refer to this paper for a more detailed description of the algorithm.

-------------------------------------------------------------------
 Copyright
-------------------------------------------------------------------

Copyright (c) 2017 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) and online at
http://www.grip.unina.it/download/LICENSE_OPEN.txt

-------------------------------------------------------------------
 Contents
-------------------------------------------------------------------

The package comprises the script "demo.m" and the directory "code".

The script "demo" calls the four main functions that implement the four versions of the Algorithm:
-Features 2D, basic approach
-Features 2D, fast approach
-Features 3D, basic approach
-Features 3D, fast approach

Each main function calls some of the following sub-functions, depending on the version aforementioned:

- [Features, timeFE] = compute_Feature_Zernike(V,ZM_diameter,ZM_order)
- [Features, timeFE] = compute_Feature_Zernike3D(V,ZM_diameter,ZM_order,p)
- [Features, timeFE] = compute_Feature_Zernike_FAST(V,ZM_diameter,ZM_order)
- [Features, timeFE] = compute_Feature_Zernike3D_FAST(V,ZM_diameter,ZM_order,p)

- [NNF, timePM] = PatchMatchBasic(features)
- [NNF, timePM] = PatchMatchFastPar(features)

- [MAP, timePP] = PostProcessingBasicFeature2D(NNF)
- [MAP, timePP] = PostProcessingBasicFeature3D(NNF)
- [MAP, timePP] = PostProcessingFastFeature2D(NNF)
- [MAP, timePP] = PostProcessingFastFeature3D(NNF)

-------------------------------------------------------------------
 Requirements
-------------------------------------------------------------------

All the functions and scripts were tested on MATLAB 2014a using image-processing and parallel toolboxes.
Experiments run on a computer with a 2x 2GHz Intel Xeon processor with 16 cores, 64GB RAM and a GPU Nvidia GeForce GTX Titan X
under a Linux platform.

-------------------------------------------------------------------
 Feedback
-------------------------------------------------------------------

If you have any comment, suggestion, or question, please do
contact Luisa Verdoliva at verdoliv@unina.it
For other information you can see http://www.grip.unina.it/
