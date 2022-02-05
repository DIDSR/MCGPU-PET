## LATEST NEWS [Feb 4, 20222]: The first version of MCGPU-PET is ready to use! A publication with details on the implemented models and validation studies is under development. Other practical documentation will be added in this readme file in the future. 

# MCGPU-PET: Open-Source Real-Time Monte Carlo PET Simulator

Monte Carlo (MC) simulations are used to model the emission, transmission, and detection of the radiation in Positron Emission Tomography (PET). In this work we introduce a new open-source MC software for PET simulation, MCGPU-PET, which has been designed to fully exploit the computing capabilities of modern GPUs to be able to simulate the acquisition of up to 1 million coincidences per second from voxelized source and material distributions. We evaluated the performance of the code, and applied it to provide fast accurate estimation of the PET coincidences (trues, scatter, randoms, and spurious coincidences). We simulated the numerical Zubal head phantom, using 8 different tissues, assuming a standard 18F-FDG uptake. A fully-3D scatter sinogram with 10 million coincidences was generated in 13 seconds in one GPU, indicating that the code might be fast enough to be used within an iterative image reconstruction process. MCGPU-PET provides estimation of True and Scatter coincidences and spurious background for non-standard isotopes at a rate 3 orders of magnitude faster than standard MC methods. This significant speed-up makes it a good candidate for providing accurate Scatter estimations within the image reconstruction process.

Authors:


- **Andreu Badal** [DIDSR, OSEL, CDRH, US Food and Drug Administration, Silver Spring, MD]
- **Joaquin L. Herraiz** and  **Alejandro López-Montes** [Complutense University of Madrid, EMFTEL, Grupo de Fisica Nuclear and IPARCOS, Madrid, Spain; Instituto de Investigacion Sanitaria del Hospital Clinico San Carlos (IdiSSC), Madrid, Spain]

---

Code presented at the IEEE NSS MIC conference (https://nssmic.ieee.org/2021/) on October 21, 2021:

- **M-07-01 – GPU-accelerated Monte Carlo-Based Scatter and Prompt-Gamma Corrections in PET**, A. López-Montes, J. Cabello, M. Conti, A. Badal, J. L. Herraiz


# How to run the sample simulation
Compile MCGPU-PET with the provided Makefile (make sure you have the CUDA SDK installed, and update the script to the compute capability of your GPU). Move the executable to the sub-folder and run it with the provided input file. Review the entire output file because it has lots of useful information, and warning messages if something didn't work well. If successful, the code will output files with the measured sinograms (Trues and Scatter), images of the voxels that emitted the detected coincidences, and an energy spectra of the detected photons (blurred by the input energy resolution, and including Compton-scattered photonas at reduced energy). 
The output binary file 'image_Trues.raw' can be opened with ImageJ (import raw -> 9x9x9, 32-bit integer values). It shows some random emission from the water cube and 3 high emission points in the central plane, as expected. The file 'MCGPU_PET.psf' shows part of the measured phase-space file in text format, but notice that the code simulates emissions one voxel at a time and therefore the PSF is not sorted by emission time.
If desired, the code can compute a 3D voxel dose distribution; but keep in mind that electron and positron tracks are not simulated, only the disintegration photons. 

     $ make 
     $ mv MCGPU-PET.x sample_simulation/
     $ cd sample_simulation/
     $ time ./MCGPU-PET.x MCGPU-PET.in | tee MCGPU-PET.out



---

# Disclaimer
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
