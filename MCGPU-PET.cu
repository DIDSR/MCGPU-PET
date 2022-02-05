//!!MCGPU-PET!!   Changes for PET marked with !!MCGPU-PET!!   (ABS, 2016-07-02)

      // *****************************************************************************
      // ***            MCGPU-PET_v0.1  (based on MC-GPU_v1.3)                     ***
      // ***                                                                       ***
      // ***  Distribution:  https://github.com/DIDSR/MCGPU-PET                    ***
      // ***                                                                       ***
      // ***  Authors:                                                             ***
      // ***                                                                       ***
      // ***   MCGPU code foundation and PET source sampling implemented by:       ***
      // ***                                                                       ***
      // ***     - Andreu Badal (Andreu.Badal-Soler[at]fda.hhs.gov)                ***
      // ***          Division of Imaging and Applied Mathematics                  ***
      // ***          Office of Science and Engineering Laboratories               ***
      // ***          Center for Devices and Radiological Health                   ***
      // ***          U.S. Food and Drug Administration                            ***
      // ***                                                                       ***
      // ***                                                                       ***
      // ***   PET detector model and sinogram reporting implemented by:           ***
      // ***                                                                       ***      
      // ***     - Joaquin L. Herraiz and Alejandro López-Montes                   ***  
      // ***         Complutense University of Madrid, EMFTEL, Grupo Fisica Nuclear***
      // ***         and IPARCOS; Instituto de Investigacion Sanitaria Hospital    ***
      // ***         Clinico San Carlos (IdiSSC), Madrid, Spain                    ***
      // ***                                                                       ***
      // ***    Code presented at the IEEE NSS MIC 2021 conference:                ***
      // ***                                                                       ***
      // ***       M-07-01 – GPU-accelerated Monte Carlo-Based Scatter and         ***
      // ***       Prompt-Gamma Corrections in PET, A. López-Montes, J. Cabello,   ***
      // ***       M. Conti, A. Badal, J. L. Herraiz                               ***
      // ***                                                                       ***
      // ***                                                                       ***
      // ***                                      Last update: 2022/02/02          ***
      // *****************************************************************************

//        ** DISCLAIMER **
//
// This software and documentation (the "Software") were developed at the Food and
// Drug Administration (FDA) by employees of the Federal Government in the course
// of their official duties. Pursuant to Title 17, Section 105 of the United States
// Code, this work is not subject to copyright protection and is in the public
// domain. Permission is hereby granted, free of charge, to any person obtaining a
// copy of the Software, to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, or sell copies of the Software or derivatives, and to permit persons
// to whom the Software is furnished to do so. FDA assumes no responsibility
// whatsoever for use by other parties of the Software, its source code,
// documentation or compiled executables, and makes no guarantees, expressed or
// implied, about its quality, reliability, or any other characteristic. Further,
// use of this code in no way implies endorsement by the FDA or confers any
// advantage in regulatory decisions.  Although this software can be redistributed
// and/or modified freely, we ask that any derivative works bear some notice that
// they are derived from it, and any modified versions bear some notice that they
// have been modified.



// *** Include header file with the structures and functions declarations
#include <MCGPU-PET.h>

// *** Include the computing kernel:
#include <MCGPU-PET_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
//!  Main program of MC-GPU: initialize the simulation enviroment, launch the GPU 
//!  kernels that perform the x ray transport and report the final results.
//!  This function reads the description of the simulation from an external file
//!  given in the command line. This input file defines the number of particles to
//!  simulate, the characteristics of the x-ray source and the detector, the number
//!  and spacing of the projections (if simulating a CT), the location of the
//!  material files containing the interaction mean free paths, and the location
//!  of the voxelized geometry file.
//!
//!                            @author  Andreu Badal
//!
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

  printf("narg= %i \n",argc);
  unsigned int *PETA;  // JLH  (PETA=1 --> True&Scatter. PETA=0 --> Bg)
  PETA = (unsigned int*)malloc(2*sizeof(unsigned int));
  PETA[0]=1;			// JLH  (PETA=1 --> True&Scatter)
  if (argc==3) PETA[0] = 0;     // JLH  (PETA=0 --> Bg)


  // -- Start time counter:
  time_t current_time = time(NULL);             // Get current time (in seconds)  
  clock_t clock_start, clock_end, clock_start_beginning;  // (requires standard header <time.h>)
  clock_start = clock();                        // Get current clock counter
  clock_start_beginning = clock_start;
  
#ifdef USING_MPI
// -- Using MPI to access multiple GPUs to simulate the x-ray projection image:
  int myID = -88, numprocs = -99, return_reduce = -1;
  MPI_Init(&argc, &argv);                       // Init MPI and get the current thread ID 
  MPI_Comm_rank(MPI_COMM_WORLD, &myID);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  
  char MPI_processor_name[81];             
  int resultlen = -1;
  MPI_Get_processor_name(MPI_processor_name, &resultlen);
    
  char* char_time = ctime(&current_time); char_time[19] = '\0';   // The time is located betwen the characters 11 and 19.
  if (numprocs>1)
  {
    printf("          >> MPI run (myId=%d, numprocs=%d) on processor \"%s\" (time: %s) <<\n", myID, numprocs, MPI_processor_name, &char_time[11]);
    fflush(stdout);   // Clear the screen output buffer
    MPI_Barrier(MPI_COMM_WORLD);   // Synchronize MPI threads  
  
    MASTER_THREAD printf("              -- Time spent initializing the MPI world (MPI_Barrier): %.3f s\n", ((double)(clock()-clock_start))/CLOCKS_PER_SEC);
  }
  
  
#else  
  int myID = 0, numprocs = 1;   // Only one CPU thread used when MPI is not activated (multiple projections will be simulated sequentially).
#endif

  MASTER_THREAD 
  { 
      printf("\n\n     *****************************************************************************\n");
      printf(    "     ***                                                                       ***\n");
      printf(    "     ***            MCGPU-PET_v0.1  (based on MC-GPU_v1.3)                     ***\n");
      printf(    "     ***                                                                       ***\n");
      printf(    "     ***     -Distribution:                                                    ***\n");
      printf(    "     ***            https://github.com/DIDSR/MCGPU-PET                         ***\n");
      printf(    "     ***                                                                       ***\n");
      printf(    "     ***     -Authors:                                                         ***\n");
      printf(    "     ***        MCGPU code foundation and PET source sampling implemented by:  ***\n");
      printf(    "     ***          - Andreu Badal (Andreu.Badal-Soler[at]fda.hhs.gov)           ***\n");
      printf(    "     ***                Division of Imaging and Applied Mathematics            ***\n");
      printf(    "     ***                Office of Science and Engineering Laboratories         ***\n");
      printf(    "     ***                Center for Devices and Radiological Health             ***\n");
      printf(    "     ***                U.S. Food and Drug Administration                      ***\n");
      printf(    "     ***                                                                       ***\n");
      printf(    "     ***        PET detector model and sinogram reporting implemented by:      ***\n");
      printf(    "     ***          - Joaquin L. Herraiz and Alejandro López-Montes              ***\n");  
      printf(    "     ***                Complutense University of Madrid, EMFTEL, Grupo Fisica ***\n");
      printf(    "     ***                Nuclear and IPARCOS; Instituto de Investigacion        ***\n");
      printf(    "     ***                Sanitaria Hospital Clinico San Carlos (IdiSSC), Spain  ***\n");
      printf(    "     ***                                                                       ***\n");
      printf(    "     ***     -Code presented at the IEEE NSS MIC 2021 conference:              ***\n");
      printf(    "     ***       M-07-01 – GPU-accelerated Monte Carlo-Based Scatter and         ***\n");
      printf(    "     ***       Prompt-Gamma Corrections in PET, A. López-Montes, J. Cabello,   ***\n");
      printf(    "     ***       M. Conti, A. Badal, J. L. Herraiz                               ***\n");
      printf(    "     ***                                                                       ***\n");
      printf(    "     ***                                      Last update: 2022/02/02          ***\n");
      printf(    "     *****************************************************************************\n\n");

      printf("****** Code execution started on: %s\n\n", ctime(&current_time));  
      fflush(stdout);
  }
    
  
#ifdef USING_CUDA
  // The "MASTER_THREAD" macro prints the messages just once when using MPI threads (it has no effect if MPI is not used):  MASTER_THREAD == "if(0==myID)"
  MASTER_THREAD printf  ("\n             *** CUDA SIMULATION IN THE GPU ***\n");
#else
  MASTER_THREAD printf  ("\n             *** SIMULATION IN THE CPU ***\n");
#endif

  MASTER_THREAD printf("\n    -- INITIALIZATION phase:\n");
  MASTER_THREAD fflush(stdout);   // Clear the screen output buffer for the master thread
  
  
///////////////////////////////////////////////////////////////////////////////////////////////////
  
  
  // *** Declare the arrays and structures that will contain the simulation data:

  struct voxel_struct voxel_data;          // Define the geometric constants of the voxel file
  struct detector_struct detector_data;  // Define an x ray detector
  struct source_struct source_data;      // Define the particles source
  struct source_energy_struct source_energy_data;    // Define the source energy spectrum
  struct linear_interp mfp_table_data;     // Constant data for the linear interpolation
  struct compton_struct  compton_table;    // Structure containing Compton sampling data (to be copied to CONSTANT memory)
  struct rayleigh_struct rayleigh_table;   // Structure containing Rayleigh sampling data (to be copied to CONSTANT memory)
  
  float3 *voxel_mat_dens = NULL;           // Poiter where voxels array will be allocated
  unsigned int voxel_mat_dens_bytes = 0;   // Size (in bytes) of the voxels array (using unsigned int to allocate up to 4.2GBytes)
  float density_max[MAX_MATERIALS];
  float density_nominal[MAX_MATERIALS];
  
  
  PSF_element_struct *PSF = NULL;    // Poiter where the PSF array will be allocated           // !!MCGPU-PET!!
  int index_PSF = 0;                 // Counter of how many PSF elements have been used
  unsigned long long int PSF_bytes = 0;      // Size of the PSF array in bytes 
  

  int mfp_table_bytes = -1, mfp_Woodcock_table_bytes = -1;   // Size of the table arrays
  float2 *mfp_Woodcock_table = NULL;                // Linear interpolation data for the Woodcock mean free path [cm]
  float3 *mfp_table_a = NULL, *mfp_table_b = NULL;  // Linear interpolation data for 3 different interactions:
                                              //  (1) inverse total mean free path (divided by density, cm^2/g)
                                              //  (2) inverse Compton mean free path (divided by density, cm^2/g)
                                              //  (3) inverse Rayleigh mean free path (divided by density, cm^2/g)
  short int dose_ROI_x_min, dose_ROI_x_max, dose_ROI_y_min, dose_ROI_y_max, dose_ROI_z_min, dose_ROI_z_max;  // Coordinates of the dose region of interest (ROI)
  ulonglong2 *voxels_Edep = NULL;           // Poiter where the voxel energy deposition array will be allocated
  int voxels_Edep_bytes = 0;                      // Size of the voxel Edep array
  
  ulonglong2 materials_dose[MAX_MATERIALS];    // Array for tally_materials_dose.     !!tally_materials_dose!!
  int kk;
  for(kk=0;kk<MAX_MATERIALS;kk++) 
  {  
    materials_dose[kk].x = 0;       // Initializing data                  !!tally_materials_dose!!
    materials_dose[kk].y = 0;
    density_nominal[kk]  =-1.0f;
  }

  clock_t clock_kernel;     // Using only cpu timers after CUDA 5.0

  double time_total_MC_init_report = 0.0, time_elapsed_MC_loop = 0.0;
  

  unsigned long long int total_histories;
  int histories_per_thread, seed_input, num_threads_per_block, gpu_id, num_projections=1;
  float fact_object;
  float E_resol;
  float E_low;
  float E_high;
  float FOVZ;
  int NROWS,NCRYSTALS,NANGLES,NRAD,NZS,NBINS,RES,NVOXS,NE,MRD,SPAN,NSEG,NSINOS;
  
  int flag_material_dose = -2;
  double D_angle=-1.0, angularROI_0=0.0, angularROI_1=360.0, initial_angle=0.0, SRotAxisD=-1.0, vertical_translation_per_projection=0.0;
  char file_name_voxels[250], file_name_materials[MAX_MATERIALS][250], file_name_output[250], file_dose_output[250], file_name_espc[250];

  // *** Read the input file given in the command line and return the significant data:
 read_input(argc, argv, myID, &total_histories, &seed_input, &gpu_id, &num_threads_per_block, &histories_per_thread, &detector_data, &PSF, &PSF_bytes, &source_data, &source_energy_data, file_name_voxels, file_name_materials, file_name_output, file_name_espc, &num_projections, &D_angle, &angularROI_0, &angularROI_1, &initial_angle, &voxels_Edep, &voxels_Edep_bytes, file_dose_output, &dose_ROI_x_min, &dose_ROI_x_max, &dose_ROI_y_min, &dose_ROI_y_max, &dose_ROI_z_min, &dose_ROI_z_max, &SRotAxisD, &vertical_translation_per_projection, &flag_material_dose, &fact_object, &E_resol, &E_low, &E_high, &FOVZ, &NROWS, &NCRYSTALS, &NANGLES, &NRAD, &NZS, &NBINS, &RES, &NVOXS, &NE, &MRD, &SPAN, &NSEG, &NSINOS) ;
    
//!!MCGPU-PET!!
//   // *** Read the energy spectrum and initialize its sampling with the Walker aliasing method:
//   MASTER_THREAD printf("    -- Reading the energy spectrum and initializing the Walker aliasing sampling algorithm.\n");
//   float mean_energy_spectrum = 0.0f;  
//   init_energy_spectrum(file_name_espc, &source_energy_data, &mean_energy_spectrum);

  
  // *** Output some of the data read to make sure everything was correctly read:
  MASTER_THREAD
  {
        printf("                   PET acquistion time = %lf s\n", source_data.acquisition_time_ps*1.0e-12);           // !!MCGPU-PET!!
        printf("                     mean life isotope = %f\n", source_data.mean_life);

        for (kk=0; kk<MAX_MATERIALS; kk++)
          if (source_data.activity[kk]>0.0f)
            printf("                  Activity material %d = %f Bq\n", kk+1, source_data.activity[kk]);

        printf("                      Input voxel file = %s\n", file_name_voxels);
        printf("                   initial random seed = %d\n", seed_input);

        if (dose_ROI_x_max>-1)
        {
          printf("                      Output dose file = %s\n", file_dose_output);
          printf("         Input region of interest dose = X[%d,%d], Y[%d,%d], Z[%d,%d]\n", dose_ROI_x_min+1, dose_ROI_x_max+1, dose_ROI_y_min+1, dose_ROI_y_max+1, dose_ROI_z_min+1, dose_ROI_z_max+1);   // Show ROI with index=1 for the first voxel instead of 0.
        }
        fflush(stdout);     
  }
  
 
// !!MCGPU-PET!! Not using the cone-beam CT trajectory option from MCGPU
//   // *** Set the detectors and sources for the CT trajectory (if needed, ie, for more than one projection):
//   if (num_projections != 1)
//   {
//     set_CT_trajectory(myID, num_projections, D_angle, angularROI_0, angularROI_1, SRotAxisD, source_data, detector_data, vertical_translation_per_projection);
//   }
  
  fflush(stdout);

  printf("\n   ----- Factor OBJECT = %f \n", fact_object);
  
  // *** Read the voxel data and allocate the density map matrix. Return the maximum density:
  load_voxels(myID, file_name_voxels, density_max, &fact_object, &voxel_data, &voxel_mat_dens, &voxel_mat_dens_bytes, &dose_ROI_x_max, &dose_ROI_y_max, &dose_ROI_z_max);
  MASTER_THREAD printf("       Total CPU memory allocated for voxels vector and data structures = %lf Mbytes\n", (double)(voxel_mat_dens_bytes+PSF_bytes+sizeof(struct voxel_struct)+sizeof(struct source_struct)+sizeof(struct detector_struct)+sizeof(struct linear_interp)+2*mfp_table_bytes+sizeof(struct rayleigh_struct)+sizeof(struct compton_struct))/(1024.0*1024.0));
  MASTER_THREAD fflush(stdout);
 
  // *** Read the material mean free paths and set the interaction table in a "linear_interp" structure:
  load_material(myID, file_name_materials, density_max, density_nominal, &mfp_table_data, &mfp_Woodcock_table, &mfp_Woodcock_table_bytes, &mfp_table_a, &mfp_table_b, &mfp_table_bytes, &rayleigh_table, &compton_table);

//--- SINOGRAM --
  int NVOX_SIM = voxel_data.num_voxels.x*voxel_data.num_voxels.y*voxel_data.num_voxels.z;
//--- VARIABLES DEFINED IN .h FILE
  // -- HOST ---
  int *True = NULL;
  True = (int*)malloc(NBINS*sizeof(int));
  int *Scatter = NULL;
  Scatter = (int*)malloc(NBINS*sizeof(int));
  int *Imagen_T = NULL;
  Imagen_T = (int*)malloc(NVOX_SIM*sizeof(int)); 
  int *Imagen_SC = NULL;
  Imagen_SC = (int*)malloc(NVOX_SIM*sizeof(int)); 
  int *Energy_Spectrum = NULL;
  Energy_Spectrum = (int*)malloc(NE*sizeof(int)); 
  

  memset(True, 0, NBINS*4);
  memset(Scatter, 0, NBINS*4);
  memset(Imagen_T, 0, NVOX_SIM*4);
  memset(Imagen_SC, 0, NVOX_SIM*4);
  memset(Energy_Spectrum, 0, NE*4);
  

  // -- DEVICE (=GPU)---
  int *True_dev = NULL;   
  int *Scatter_dev = NULL;
  int *Imagen_T_dev = NULL;
  int *Imagen_SC_dev = NULL;
  int *Energy_Spectrum_dev = NULL;

  //int* PETA_dev = NULL;  //JLH
  //unsigned int h_ff[] = {90, 50, 100};
 

//---- END OF SINOGRAM DEFINITION ------------------------------ APRIL 18


  
  if (detector_data.PSF_radius < 0.0f)   // !!MCGPU-PET!!
  {
    detector_data.PSF_center.x = voxel_data.size_bbox.x*0.5f;
    detector_data.PSF_center.y = voxel_data.size_bbox.y*0.5f;
    detector_data.PSF_center.z = voxel_data.size_bbox.z*0.5f;
    detector_data.PSF_radius   = -detector_data.PSF_radius;
  }
  
        printf("\n\n    ** INPUT PET DETECTOR PARAMETERS  **\n");
        printf(    "           Phase Space File (PSF) size =  %d\n", detector_data.PSF_size);
        printf(    "                   PSF detector center = (%.5f,%.5f,%.5f) cm\n", detector_data.PSF_center.x, detector_data.PSF_center.y, detector_data.PSF_center.z);
        printf(    "        PSF detector height and radius = %.5f, %.5f cm\n", detector_data.PSF_height, detector_data.PSF_radius);
        printf(    "                       Output PSF file = %s\n", file_name_output);

        if (detector_data.tally_TYPE==1)
          printf("                       PSF will include only True coincidences (Scatter not reported).\n");
        else if (detector_data.tally_TYPE==2)
          printf("                       PSF will include only Scatter coincidences (Trues not reported).\n");
        else
          printf("                       PSF will include True and Scatter coincidences.\n");

 
        printf("\n\n            Detector energy resolution = %f\n", E_resol);
        printf(    "     Low, high energy window threshold = %f , %f\n\n\n", E_low, E_high);

    

  // -- Check that the input material tables and the x-ray source are consistent:
  if ( (ANNIHILATION_PHOTON_ENERGY > (mfp_table_data.e0 + (mfp_table_data.num_values-1)/mfp_table_data.ide)) )                                               // !!MCGPU-PET!!
  {
    MASTER_THREAD printf("\n\n\n !!ERROR!! The input material files do not have data up to %f keV, please update material files.\n\n", ANNIHILATION_PHOTON_ENERGY*1e-3);  // !!MCGPU-PET!!
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-1);
  }


//  int source_voxels0=0, source_voxels1=0;
  
  // -- Pre-compute the total mass of each material present in the voxel phantom (to be used in "report_materials_dose"):
  double voxel_volume = 1.0 / ( ((double)voxel_data.inv_voxel_size.x) * ((double)voxel_data.inv_voxel_size.y) * ((double)voxel_data.inv_voxel_size.z) );
  double mass_materials[MAX_MATERIALS];
  for(kk=0; kk<MAX_MATERIALS; kk++)
    mass_materials[kk] = 0.0;
  for(kk=0; kk<(voxel_data.num_voxels.x*voxel_data.num_voxels.y*voxel_data.num_voxels.z); kk++)  // For each voxel in the geometry
  {
    mass_materials[((int)voxel_mat_dens[kk].x)-1] += ((double)voxel_mat_dens[kk].y)*voxel_volume;        // Add material mass = density*volume
    
//    if (((int)voxel_mat_dens[kk].x) == source_data.material[0])           // !!MCGPU-PET!!
//      source_voxels0++;
//    else if (((int)voxel_mat_dens[kk].x) == source_data.material[1])      // !!MCGPU-PET!!
//      source_voxels1++;
  }

//  MASTER_THREAD printf("    Activity for each one of the %d source voxels with material %d = %f Bq\n", source_voxels0, source_data.material[0], source_data.activity[0]);
//  MASTER_THREAD printf("    Activity for each one of the %d source voxels with material %d = %f Bq\n", source_voxels1, source_data.material[1], source_data.activity[1]);
    

  // *** Initialize the GPU using the NVIDIA CUDA libraries, if USING_CUDA parameter defined at compile time:
#ifdef USING_CUDA    
  // -- Declare the pointers to the device global memory, when using the GPU:
  float3 *voxel_mat_dens_device     = NULL;
  float2 *mfp_Woodcock_table_device = NULL;
  float3 *mfp_table_a_device        = NULL,
         *mfp_table_b_device        = NULL;
  struct rayleigh_struct *rayleigh_table_device = NULL;
  struct compton_struct  *compton_table_device  = NULL;
  ulonglong2 *voxels_Edep_device                = NULL;
  struct detector_struct *detector_data_device  = NULL;
  struct source_struct   *source_data_device    = NULL;  
  ulonglong2 *materials_dose_device = NULL;     // !!tally_materials_dose!!
  
  int* seed_input_device = NULL;        // Store latest random seed used in GPU in global memory to continue random sequence in consecutive projections.   !!DBTv1.4!!
  unsigned long long int* total_histories_device = NULL;   // Store the total number of histories simulated in the kernel        !!MCGPU-PET!!
  
  PSF_element_struct *PSF_device = NULL;    // Poiter where the PSF array will be allocated in GPU memory           // !!MCGPU-PET!!
  int *index_PSF_device = NULL;             // Poiter where the PSF counter will be allocated in GPU memory 

  // -- Sets the CUDA enabled GPU that will be used in the simulation, and allocate and copies the simulation data in the GPU global and constant memories.
  init_CUDA_device(&gpu_id, myID, numprocs, &voxel_data, &source_data, &source_energy_data, &detector_data, &mfp_table_data,  /*Variables GPU constant memory*/
        voxel_mat_dens, &voxel_mat_dens_device, voxel_mat_dens_bytes,                                                         /*Variables GPU global memory*/
        PSF, &PSF_device, PSF_bytes, &index_PSF_device,
        mfp_Woodcock_table, &mfp_Woodcock_table_device, mfp_Woodcock_table_bytes,
        mfp_table_a, mfp_table_b, &mfp_table_a_device, &mfp_table_b_device, mfp_table_bytes,
        &rayleigh_table, &rayleigh_table_device,
        &compton_table, &compton_table_device, &detector_data_device, &source_data_device,
        voxels_Edep, &voxels_Edep_device, voxels_Edep_bytes, &dose_ROI_x_min, &dose_ROI_x_max, &dose_ROI_y_min, &dose_ROI_y_max, &dose_ROI_z_min, &dose_ROI_z_max,
        materials_dose, &materials_dose_device, flag_material_dose, &seed_input_device, &seed_input, &total_histories_device, num_projections, NVOX_SIM, &True_dev,
        &Scatter_dev, &Imagen_T_dev, &Imagen_SC_dev, &Energy_Spectrum_dev, &NBINS, &NE);

  // SINOGRAM -- Aseguro que inician en zero --
  checkCudaErrors(cudaMemcpy(True_dev, True, NBINS*sizeof(int),  cudaMemcpyHostToDevice));  
  checkCudaErrors(cudaMemcpy(Scatter_dev, Scatter, NBINS*sizeof(int),  cudaMemcpyHostToDevice));  
  checkCudaErrors(cudaMemcpy(Imagen_T_dev, Imagen_T, NVOX_SIM*sizeof(int),  cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(Imagen_SC_dev, Imagen_SC, NVOX_SIM*sizeof(int),  cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(Energy_Spectrum_dev, Energy_Spectrum, NE*sizeof(int),  cudaMemcpyHostToDevice));  
  checkCudaErrors(cudaMemcpyToSymbol(PETA_DEV, PETA, 2*sizeof(unsigned int),0,cudaMemcpyHostToDevice));

  // -- Constant data already moved to the GPU: clean up unnecessary RAM memory
  free(mfp_Woodcock_table);
  free(mfp_table_a);
  free(mfp_table_b);
  if (0!=myID)    // Keep the geometry data for the MPI root because the voxel densities are still needed to compute the final doses
    free(voxel_mat_dens);
    

#endif
  
  MASTER_THREAD
  {
    current_time=time(NULL);
    printf("\n    -- INITIALIZATION finished: elapsed time = %.3f s. \n\n", ((double)(clock()-clock_start))/CLOCKS_PER_SEC);
  }
  

#ifdef USING_MPI
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);   // Synchronize MPI threads before starting the MC phase.
#endif

  
///////////////////////////////////////////////////////////////////////////////////////////////////
  
  
  
  MASTER_THREAD
  {
    current_time=time(NULL);
    printf("\n\n    -- MONTE CARLO LOOP phase. Time: %s\n\n", ctime(&current_time)); 
    fflush(stdout);    
  }


  clock_start = clock();   // Start the CPU clock

  // Prevent simulating blocks for voxels at the top of the phantom (z) where there is no activity:
  //int last_z_active = find_last_z_active(&voxel_data, &source_data, voxel_mat_dens);                                       // !!MCGPU-PET!! v.0.2
  int last_z_active = voxel_data.num_voxels.z;  //JLH
  
  // ORIGINAL: dim3 blocks(voxel_data.num_voxels.x, voxel_data.num_voxels.y, voxel_data.num_voxels.z);  // !!MCGPU-PET!! Launch one block per voxel!!
  dim3 blocks(voxel_data.num_voxels.x, voxel_data.num_voxels.y, last_z_active);  // !!MCGPU-PET!! Launch one block per voxel, except for the top Z layers without any activity
  dim3 threads(num_threads_per_block, 1, 1);


  #ifdef USING_MPI
    if (numprocs>1)
    {
      update_seed_PRNG(myID, (unsigned long long int)(123456789012), &seed_input);     // Set the random number seed far from any other MPI thread (myID)
      printf("\n        ==> CUDA (MPI process #%d in \"%s\"): simulating %d x %d x %d blocks of %d threads (random seed: %d).\n", myID, MPI_processor_name, blocks.x, blocks.y, blocks.z, threads.x, seed_input);    
      checkCudaErrors(cudaMemcpy(seed_input_device, &seed_input, sizeof(int), cudaMemcpyHostToDevice));    // Upload initial seed value to GPU memory.   !!DBTv1.4!!    
      MPI_Barrier(MPI_COMM_WORLD);   // Synchronize MPI threads to better organize output
    }
    else
      printf("\n        ==> LAUNCHING CUDA KERNEL: simulating %d x %d x %d blocks of %d threads.\n\n\n", blocks.x, blocks.y, blocks.z, threads.x);

  #else
    printf("\n\n        ==> CUDA: simulating %d x %d x %d blocks of %d threads.\n\n\n", blocks.x, blocks.y, blocks.z, threads.x);
  #endif  
  fflush(stdout); 

  clock_kernel = clock();


  // -- Launch Monte Carlo simulation kernel:

//!!MCGPU-PET!!: For PET we use acquisition_time_ps instead of histories_per_thread to finish simulation, and we don't have multiple projections (num_p):
  track_particles<<<blocks,threads>>>(total_histories_device, seed_input_device, PSF_device, index_PSF_device, voxels_Edep_device, voxel_mat_dens_device, mfp_Woodcock_table_device, mfp_table_a_device, mfp_table_b_device, rayleigh_table_device, compton_table_device, detector_data_device, source_data_device, materials_dose_device, True_dev, Scatter_dev, Imagen_T_dev, Imagen_SC_dev, Energy_Spectrum_dev, E_resol, E_low, E_high, FOVZ, NROWS, NCRYSTALS, NANGLES, NRAD, NZS, NBINS, RES, NVOXS, NE, MRD, SPAN, NSINOS);  // !!MCGPU-PET!!


  cudaDeviceSynchronize();    // Force the runtime to wait until the GPU kernel is completed
  getLastCudaError("\n\n !!Kernel execution failed while simulating particle tracks!! \n\n\n");  // Check if kernel execution generated any error
  

///////////////////////////////////////////////////////////////////////////////////////////////////


    
  // Get current time and calculate execution time in the MC loop:
  time_elapsed_MC_loop = ((double)(clock()-clock_start))/CLOCKS_PER_SEC;       
  printf("\n\n    -- MONTE CARLO LOOP finished: time tallied in main program: %.3f s\n", time_elapsed_MC_loop);
  
  
  // *** Simulation finished! Move simulation results from GPU to CPU:
    
  // -- Copy the simulated Phase Space File data and other variables from the GPU memory to the CPU: 
  
  checkCudaErrors(cudaMemcpy(&total_histories, total_histories_device, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));    // Download the total number of simulated histories           !!MCGPU-PET!!
  checkCudaErrors(cudaMemcpy(&index_PSF,       index_PSF_device,       sizeof(int),                    cudaMemcpyDeviceToHost));    // Download the total number of elements added to the PSF     !!MCGPU-PET!!  
  
  printf(    "                                  Total number of histories simulated = %lld (note: each annihilation produces 2 histories)\n", total_histories);   // !!MCGPU-PET!!

  if (index_PSF <= detector_data.PSF_size)
  {
    printf(  "                                  Number of coincidences stored in the PSF = %d\n\n\n", index_PSF/2);         // !!MCGPU-PET!!
  }
  else
  {
    printf(  "\n                       WARNING:  %d photons arrived at the detector but only %d elements could be stored in the PSF!!\n", index_PSF, detector_data.PSF_size);    // !!MCGPU-PET!!
    printf(    "                                 Edit the input file to increase the PSF size, or reduce the source activity.\n\n");
    index_PSF = detector_data.PSF_size;  // The kernel will not save more than the max value of elements allocated
  }

  clock_t clock_PSF_1 = clock();         // !!October2017!!
  checkCudaErrors(cudaMemcpy(PSF, PSF_device, index_PSF*sizeof(PSF_element_struct), cudaMemcpyDeviceToHost) );           // Download the new elements added to the PSF              !!MCGPU-PET!!    
  clock_t clock_PSF_2 = clock();
  
  // -- Report PSF file to disk:
  if (detector_data.tally_PSF_SINOGRAM==0||detector_data.tally_PSF_SINOGRAM==1) {
  MASTER_THREAD report_PSF(file_name_output, PSF, index_PSF, total_histories, time_elapsed_MC_loop, &source_data, &detector_data, file_name_voxels);                           //   !!MCGPU-PET!!
  }
  clock_t clock_PSF_3 = clock();
  MASTER_THREAD printf("    -- Time spent downloading the PSF from the GPU to the CPU: %.4f s ; time spent reporting PSF to disk: %.4f s\n\n", ((double)(clock_PSF_2-clock_PSF_1))/CLOCKS_PER_SEC, ((double)(clock_PSF_3-clock_PSF_2))/CLOCKS_PER_SEC);      // !!October2017!!
  
// SINOGRAM  // !!March2018!!
  checkCudaErrors(cudaMemcpy(True, True_dev, NBINS*sizeof(int), cudaMemcpyDeviceToHost));             // Download the True Sinogram              !!MCGPU-PET!!    
  checkCudaErrors(cudaMemcpy(Scatter, Scatter_dev, NBINS*sizeof(int), cudaMemcpyDeviceToHost));       // Download the Scatter Sinogram           !!MCGPU-PET!!    
  checkCudaErrors(cudaMemcpy(Imagen_T, Imagen_T_dev, NVOX_SIM*sizeof(int), cudaMemcpyDeviceToHost));         // Download the Image                      !!MCGPU-PET!!
  checkCudaErrors(cudaMemcpy(Imagen_SC, Imagen_SC_dev, NVOX_SIM*sizeof(int), cudaMemcpyDeviceToHost));         // Download the Image                      !!MCGPU-PET!!      
  checkCudaErrors(cudaMemcpy(Energy_Spectrum, Energy_Spectrum_dev, NE*sizeof(int), cudaMemcpyDeviceToHost));         // Download the Spectrum                      !!MCGPU-PET!!    
// Report --
  long int sumTru=0;
  long int sumSc=0;
  long int sumImg_T=0;
  long int sumImg_SC=0;
  for (int i=0;i<NBINS;i++) {sumTru += True[i]; sumSc += Scatter[i];}
  for (int i=0;i<NVOX_SIM;i++) {sumImg_T += Imagen_T[i]; sumImg_SC += Imagen_SC[i];}    

  if (detector_data.tally_PSF_SINOGRAM==0||detector_data.tally_PSF_SINOGRAM==2) {
  if (argc==2) {

// -Modification of the output sinogram to write into compressed gzip 
  gzFile file1 = gzopen("sinogram_Trues.raw.gz","wb");
  gzwrite(file1,True,NBINS*sizeof(int));
  gzclose(file1);

  gzFile file2 = gzopen("sinogram_Scatter.raw.gz","wb");
  gzwrite(file2,Scatter,NBINS*sizeof(int));     // Corrected bug: file2 instead of file1   FEB2022
  gzclose(file2);
 
  gzFile file3 = gzopen("image_Trues.raw.gz","wb");
  gzwrite(file3,Imagen_T,NVOX_SIM*sizeof(int));
  gzclose(file3);

  gzFile file4 = gzopen("image_Scatter.raw.gz","wb");
  gzwrite(file4,Imagen_SC,NVOX_SIM*sizeof(int));
  gzclose(file4);

//  -Write uncompressed binary:
//  FILE* file1 = fopen("Trues.raw", "wb");
//  fwrite(True, sizeof(int), NBINS, file1);    
//  fclose(file1);
//  FILE* file2 = fopen("Scatter.raw", "wb");
//  fwrite(Scatter, sizeof(int), NBINS, file2);    
//  fclose(file2);
//  FILE* file3 = fopen("Image_T.raw", "wb");
//  fwrite(Imagen_T, sizeof(int), NVOX_SIM, file3);   
//  fclose(file3);
//  FILE* file4 = fopen("Image_SC.raw", "wb");
//  fwrite(Imagen_SC, sizeof(int), NVOX_SIM, file4);   
//  fclose(file3);

  }else{

  gzFile file1 = gzopen("sinogram_BG_Trues.raw.gz","wb");
  gzwrite(file1,True,NBINS*sizeof(int));
  gzclose(file1);

  gzFile file2 = gzopen("sinogram_BG_Scatter.raw.gz","wb");
  gzwrite(file2,Scatter,NBINS*sizeof(int));
  gzclose(file2);

  gzFile file3 = gzopen("BG_image_True.raw.gz","wb");
  gzwrite(file3,Imagen_T,NVOX_SIM*sizeof(int));
  gzclose(file3);

  gzFile file4 = gzopen("BG_image_Scatter.raw.gz","wb");
  gzwrite(file4,Imagen_SC,NVOX_SIM*sizeof(int));
  gzclose(file4);

//  FILE* file1 = fopen("BG_Trues.raw", "wb");
//  fwrite(True, sizeof(int), NBINS, file1);    
//  fclose(file1);
//  FILE* file2 = fopen("BG_Scatter.raw", "wb");
//  fwrite(Scatter, sizeof(int), NBINS, file2);    
//  fclose(file2);
//  FILE* file3 = fopen("BG_Image_T.raw", "wb");
//  fwrite(Imagen_T, sizeof(int), NVOX_SIM, file3);   
//  fclose(file3);
//  FILE* file4 = fopen("BG_Image_SC.raw", "wb");
//  fwrite(Imagen_SC, sizeof(int), NVOX_SIM, file4);   
//  fclose(file3);
  }

 } //write only if tally_PSF_SINOGRAM=0 or 2

  FILE* file5 = fopen("Energy_Sinogram_Spectrum.dat", "w");
  fprintf (file5, "# HISTOGRAM OF ENERGIES (keV) OF DETECTED COINCIDENCE PHOTONS (before energy window selection)\n");
  fprintf (file5, "#     Energy bin [keV]     Counts\n");
  for (int i=0 ; i<NE ; i++){ fprintf (file5, "%10d  %10d \n",i,Energy_Spectrum[i]); }
  fclose (file5);


/* !!TO DO!!  //!!MCGPU-PET!!  

  MASTER_THREAD report_PSF_file(...);  // Report PSF computed by master thread
  
  #ifdef USING_MPI
    int i;
    for(i=1;i<numprocs;i++)
    {
      // -- Send results from thread i to master, then report following previous results:
      return_reduce = MPI_Reduce(PSF_file,..., 0, MPI_COMM_WORLD);
      MASTER_THREAD report_PSF_file(...);  // Report PSF computed by thread 'i'
      ...keep track of total number of particles simulated, and total number elements in complete PSF...
    }
  #endif
    ...report total number of particles simulated, and total number elements in complete PSF...
*/  
  




  // -- Copy the simulated voxel and material doses to CPU:

  if (dose_ROI_x_max > -1)
  {   
    MASTER_THREAD clock_kernel = clock();    

    checkCudaErrors( cudaMemcpy( voxels_Edep, voxels_Edep_device, voxels_Edep_bytes, cudaMemcpyDeviceToHost) );  // Copy final dose results to host (for every MPI threads)

    MASTER_THREAD printf("       ==> CUDA: Time copying dose results from device to host: %.6f s\n", float(clock()-clock_kernel)/CLOCKS_PER_SEC);
  }
  
  if (flag_material_dose==1)
    checkCudaErrors( cudaMemcpy( materials_dose, materials_dose_device, MAX_MATERIALS*sizeof(ulonglong2), cudaMemcpyDeviceToHost) );  // Copy materials dose results to host, if tally enabled in input file.   !!tally_materials_dose!!

  // -- Clean up GPU device memory:
  clock_kernel = clock();    

  cudaFree(voxel_mat_dens_device);
  cudaFree(PSF_device);
  cudaFree(mfp_Woodcock_table_device);
  cudaFree(mfp_table_a_device);
  cudaFree(mfp_table_b_device);
  cudaFree(voxels_Edep_device);
//SINOGRAM
  cudaFree(True_dev);
  cudaFree(Scatter_dev);
  cudaFree(Imagen_T_dev);
  cudaFree(Imagen_SC_dev);
  cudaFree(Energy_Spectrum_dev);
  checkCudaErrors( cudaDeviceReset() );

  MASTER_THREAD printf("       ==> CUDA: Time freeing the device memory and ending the GPU threads: %.6f s\n", float(clock()-clock_kernel)/CLOCKS_PER_SEC);



#ifdef USING_MPI
  if (numprocs>1)
  {
    current_time=time(NULL);     // Get current time (in seconds)
    char_time = ctime(&current_time); char_time[19] = '\0';   // The time is located betwen the characters 11 and 19.  
    printf("        >> MPI thread %d in \"%s\" done! (local time: %s)\n", myID, MPI_processor_name, &char_time[11]);
    fflush(stdout);   // Clear the screen output buffer
  }
#endif


  
  // *** Report the total dose for all the projections, if the tally is not disabled (must be done after MPI_Barrier to have all the MPI threads synchronized):
  MASTER_THREAD clock_start = clock(); 
  
  if (dose_ROI_x_max > -1)
  {   
    
#ifdef USING_MPI
    if (numprocs>1)
    {
      // -- Use MPI_Reduce to accumulate the dose from all projections:      
      //    Allocate memory in the root node to combine the dose results with MPI_REDUCE:
      int num_voxels_ROI = voxels_Edep_bytes/((int)sizeof(ulonglong2));   // Number of elements allocated in the "dose" array.
      ulonglong2 *voxels_Edep_total = (ulonglong2*) malloc(voxels_Edep_bytes);
      if (voxels_Edep_total==NULL)
      {
        printf("\n\n   !!malloc ERROR!! Not enough memory to allocate %d voxels by the MPI root node for the total deposited dose (and uncertainty) array (%f Mbytes)!!\n\n", num_voxels_ROI, voxels_Edep_bytes/(1024.f*1024.f));
        exit(-2);
      }
      else
      {
        MASTER_THREAD
        {
          printf("\n        >> Array for the total deposited dose correctly allocated by the MPI root node (%f Mbytes).\n", voxels_Edep_bytes/(1024.f*1024.f));
          printf(  "           Waiting at MPI_Barrier for thread synchronization.\n");
        }
      }      
      
      
      MASTER_THREAD printf("\n        >> Calling MPI_Reduce to accumulate the dose from all projections...\n\n");    
      
      return_reduce = MPI_Reduce(voxels_Edep, voxels_Edep_total, 2*num_voxels_ROI, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);   // Sum all the doses in "voxels_Edep_total" at thread 0.
      if (MPI_SUCCESS != return_reduce)
      {
        printf("\n\n !!ERROR!! Possible error reducing (MPI_SUM) the dose results??? return_reduce = %d for thread %d\n\n\n", return_reduce, myID);
      }

      // -- Exchange the dose simulated in thread 0 for the final dose from all threads  
      MASTER_THREAD
      {
        free(voxels_Edep);
        voxels_Edep = voxels_Edep_total;    // point the voxels_Edep pointer to the final voxels_Edep array in host memory
        voxels_Edep_total = NULL;           // This pointer is not needed by now
      }
    }
#endif
        
    // -- Report the total dose for all the projections:
    MASTER_THREAD report_voxels_dose(file_dose_output, num_projections, &voxel_data, voxel_mat_dens, voxels_Edep, time_elapsed_MC_loop, total_histories, dose_ROI_x_min, dose_ROI_x_max, dose_ROI_y_min, dose_ROI_y_max, dose_ROI_z_min, dose_ROI_z_max, &source_data);        
  }
  
  
  // -- Report "tally_materials_dose" with data from all MPI threads, if tally enabled:
  if (flag_material_dose==1)
  {
  #ifdef USING_MPI
    ulonglong2 materials_dose_total[MAX_MATERIALS];
    return_reduce = MPI_Reduce(materials_dose, materials_dose_total, 2*MAX_MATERIALS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);   // !!tally_materials_dose!!
  #else
    ulonglong2 *materials_dose_total = materials_dose;  // Create a dummy pointer to the materials_dose data 
  #endif
    
    MASTER_THREAD report_materials_dose(num_projections, total_histories, density_nominal, materials_dose_total, mass_materials, file_name_materials);    // Report the material dose  !!tally_materials_dose!!
  }
  
  MASTER_THREAD clock_end = clock();
  MASTER_THREAD printf("\n\n       ==> CUDA: Time reporting the dose data: %.6f s\n", ((double)(clock_end-clock_start))/CLOCKS_PER_SEC);
  
  
  // *** Report the total dose for all the projections, if the tally is not disabled (must be done after MPI_Barrier to have all the MPI threads synchronized):
  
  // *** Clean up RAM memory. If CUDA was used, the geometry and table data were already cleaned for MPI threads other than root after copying data to the GPU:
  free(voxels_Edep);
  free(PSF);
//SINOGRAM
  free(True);
  free(Scatter);
  free(Imagen_T);
  free(Imagen_SC);
  free(Energy_Spectrum);

#ifdef USING_CUDA
  MASTER_THREAD free(voxel_mat_dens);
#else
  free(voxel_mat_dens);
  free(mfp_Woodcock_table);
  free(mfp_table_a);
  free(mfp_table_b);
#endif
   

#ifdef USING_MPI
  MPI_Finalize();   // Finalize MPI library: no more MPI calls allowed below.
#endif
 
    // -- Report output size and aditional information:

  MASTER_THREAD
  {
   printf("\n\n       ****** MCGPU-PET SINOGRAM INFORMATION ******\n\n");
   printf("           >>> Number of radial bins = %i\n", NRAD);
   printf("           >>> Number of angular bins = %i\n", NANGLES);
   printf("           >>> Total number of sinograms = %i\n", NSINOS);
   printf("           >>> Maximum Ring Difference = %i\n", MRD);
   printf("           >>> SPAN = %i\n", SPAN);
   printf("           >>> Number segments = %i\n", NSEG);
   printf("           >>> Number z slices = %i\n\n", NZS);
   printf("           >>> Number of true coincidences in the sinogram = %ld  \n",sumTru);
   printf("           >>> Number of scatter coincidences in the sinogram = %ld  \n",sumSc);
   printf("           >>> Size of the 3D sinogram = %i X %i X %i\n", NRAD,NANGLES,NSINOS);


   printf("\n\n       ****** MCGPU-PET IMAGES INFORMATION ******\n\n");
   printf("           >>> Two binary 3D matrix (32-bit integer per voxel) with same size as the input voxelized geometry.\n"); 
   printf("           >>> Report how many detected coincidences, Trues and Scatter separately, were emitted from each voxel.\n\n");
   printf("           >>> Number of voxels = (%i,%i,%i)\n", voxel_data.num_voxels.x, voxel_data.num_voxels.y, voxel_data.num_voxels.z);
      //FEB2022   Wrong reporting before?
      // printf("           >>> Number of X,Y voxels = %i\n", RES);
      // printf("           >>> Number of Z voxels = %i\n", NZS);
      // printf("           >>> Size of the images = %i X %i X %i\n\n", RES,RES,NZS);
   printf("           >>> Number of true counts in the image of trues = %ld  \n",sumImg_T);
   printf("           >>> Number of scatter counts in the image of scatter = %ld  \n",sumImg_SC);
  }


  MASTER_THREAD 
  {
    printf("\n\n\n    -- SIMULATION FINISHED!\n");
    
    time_total_MC_init_report = ((double)(clock()-clock_start_beginning))/CLOCKS_PER_SEC;

    // -- Report total performance:
    printf("\n\n       ****** TOTAL SIMULATION PERFORMANCE (including initialization and reporting) ******\n\n");  
    printf(    "          >>> Execution time including initialization, transport and report: %.3f s.\n", time_total_MC_init_report);
    printf(    "          >>> Time spent in the Monte Carlo transport only: %.3f s.\n", time_elapsed_MC_loop);
    printf(    "          >>> Time spent in initialization, reporting and clean up: %.3f s.\n", (time_total_MC_init_report-time_elapsed_MC_loop));
    printf(    "          >>> Total number of simulated histories (x-rays):  %lld  (note: each positron annihilation produces 2 histories)\n", total_histories);
    if (time_total_MC_init_report>0.0000001)
      printf(  "          >>> Total speed (using %d thread, including transport, initialization and report times) [x-rays/s]:  %.2f\n", numprocs, (double)(total_histories/time_total_MC_init_report));
    if (time_elapsed_MC_loop>0.0000001)
      printf(    "          >>> Total speed Monte Carlo transport only (using %d thread) [x-rays/s]:  %.2f\n\n", numprocs, (double)(total_histories/time_elapsed_MC_loop));
  
    current_time=time(NULL);     // Get current time (in seconds)
    
    printf("\n****** Code execution finished on: %s\n\n", ctime(&current_time));
  }
 


#ifdef USING_CUDA
  cudaDeviceReset();  // Destroy the CUDA context before ending program (flush visual debugger data).
#endif

  return 0;
}





////////////////////////////////////////////////////////////////////////////////
//! Read the input file given in the command line and return the significant data.
//! Example input file:
//!
//!    1000000          [Total number of histories to simulate]
//!    geometry.vox     [Voxelized geometry file name]
//!    material.mat     [Material data file name]
//!
//!       @param[in] argc   Command line parameters
//!       @param[in] argv   Command line parameters: name of input file
//!       @param[out] total_histories  Total number of particles to simulate
//!       @param[out] seed_input   Input random number generator seed
//!       @param[out] num_threads_per_block   Number of CUDA threads for each GPU block
//!       @param[out] detector_data
//!       @param[out] PSF
//!       @param[out] source_data
//!       @param[out] file_name_voxels
//!       @param[out] file_name_materials
//!       @param[out] file_name_output
////////////////////////////////////////////////////////////////////////////////
void read_input(int argc, char** argv, int myID, unsigned long long int* total_histories, int* seed_input, int* gpu_id, int* num_threads_per_block, int* histories_per_thread, struct detector_struct* detector_data, PSF_element_struct** PSF_ptr, unsigned long long int* PSF_bytes, struct source_struct* source_data, struct source_energy_struct* source_energy_data, char* file_name_voxels, char file_name_materials[MAX_MATERIALS][250] , char* file_name_output, char* file_name_espc, int* num_projections, double* D_angle, double* angularROI_0, double* angularROI_1, double* initial_angle, ulonglong2** voxels_Edep_ptr, int* voxels_Edep_bytes, char* file_dose_output, short int* dose_ROI_x_min, short int* dose_ROI_x_max, short int* dose_ROI_y_min, short int* dose_ROI_y_max, short int* dose_ROI_z_min, short int* dose_ROI_z_max, double* SRotAxisD, double* vertical_translation_per_projection, int* flag_material_dose, float* fact_object, float* E_resol, float* E_low, float* E_high, float* FOVZ, int* NROWS, int* NCRYSTALS, int* NANGLES, int* NRAD, int* NZS, int* NBINS, int* RES, int* NVOXS, int* NE, int* MRD, int* SPAN, int* NSEG, int*NSINOS)
{
  FILE* file_ptr = NULL;
  char new_line[250];
  char *new_line_ptr = NULL;  
  double dummy;

  // -- Read the input file name from command line, if given (otherwise keep default value):
  if (argc>=2)
  {
    file_ptr = fopen(argv[1], "r");
    if (NULL==file_ptr)
    {
      printf("\n\n   !!read_input ERROR!! Input file not found or not readable. Input file name: \'%s\'\n\n", argv[1]);      
        //  Not finalizing MPI here because we want the execution to fail if there is a problem with any MPI thread!!! MPI_Finalize();   // Finalize MPI library: no more MPI calls allowed below.
      exit(-1);
    }
  } 
//  else if (argc>2)
//  {
//    
//    MASTER_THREAD printf("\n\n   !!read_input ERROR!! Too many input parameter (argc=%d)!! Provide only the input file name.\n\n", argc);    
//    // Finalizing MPI because all threads will detect the same problem and fail together.
//    #ifdef USING_MPI
//      MPI_Finalize();
//    #endif
//    exit(-1);
//  }
  else
  {
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! Input file name not given as an execution parameter!! Try again...\n\n");
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-1);
  }

  MASTER_THREAD printf("\n    -- Reading the input file \'%s\':\n", argv[1]);


  // -- Init. [SECTION SIMULATION CONFIG v.2016-07-05]:    // !!MCGPU-PET!!
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);    // Read full line (max. 250 characters).
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION SIMULATION CONFIG v.2016-07-05\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION SIMULATION CONFIG v.2016-07-05")==NULL);   // Skip comments and empty lines until the section begins
//   new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
//     sscanf(new_line, "%lf", &dummy_double);
//     *total_histories = (unsigned long long int) (dummy_double+0.0001);  // Maximum unsigned long long value: 18446744073709551615
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", seed_input);   // Set the RANECU PRNG seed (the same seed will be used to init the 2 MLCGs in RANECU)
      if (*seed_input==0) // Generating random seed
      {
      struct timeval tv;
      gettimeofday(&tv,NULL);
       //seed = (int)tv.tv_usec;
       *seed_input = (int)((tv.tv_sec / 1000) + (tv.tv_usec * 1000));
      }

  MASTER_THREAD printf("\n    -- Seed used in the execution \'%d\':\n", *seed_input);
// END 03/11/2020

  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", gpu_id);       // GPU NUMBER WHERE SIMULATION WILL RUN
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", num_threads_per_block);  // GPU THREADS PER CUDA BLOCK
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f", fact_object);  // OBJECT PRESENT? (IF NO --> DENSITY = 0.001)
//  printf("----- Factor OBJECT = %f \n",*fact_object);
//  printf("----- Factor NUMBER OF THREADS = %i \n",*num_threads_per_block);

#ifdef USING_CUDA
  if ((*num_threads_per_block%32)!=0)
  {
    
    // --ELIMINTE NEED TO USE NUMBER OF THREADS MULTIPLE OF 32   !!PET!!
    MASTER_THREAD printf("\n   !!read_input!! The input number of GPU threads per CUDA block, %d, is NOT a multiple of 32 (warp size). Some GPU resources will not be fully utilized.\n\n", *num_threads_per_block);
    /*
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! The input number of GPU threads per CUDA block must be a multiple of 32 (warp size). Input value: %d !!\n\n", *num_threads_per_block);
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-2);
    */    
  }
#endif

// !!MCGPU-PET!!
//   new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
//     sscanf(new_line, "%d", histories_per_thread);   // HISTORIES PER GPU THREAD

    
  // -- Init. [SECTION SOURCE PET SCAN v.2017-03-14]:
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION SOURCE PET SCAN v.2017-03-14\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION SOURCE PET SCAN v.2017-03-14")==NULL);   // Skip comments and empty lines until the section begins     // !!MCGPU-PET!!

  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%lf", &dummy);                                                     // TOTAL PET SCAN ACQUISITION TIME  [seconds]
  source_data->acquisition_time_ps = (unsigned long long int)(dummy*1.0e12 + 0.5);       // Store acquistion time in picoseconds
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f", &source_data->mean_life);                                     // ISOTOPE MEAN LIFE    
	
  int ii=0, mat=0;
  float act=0.0f;
  
  for (ii=0; ii<MAX_MATERIALS; ii++)  		
    source_data->activity[ii] = 0.0f;       // Init table to 0 Bq
  
  for (ii=0; ii<MAX_MATERIALS; ii++)         // Read input material activity
  {
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);  
    sscanf(new_line, "%d %f", &mat, &act);                            // TABLE MATERIAL NUMBER AND VOXEL ACTIVITY [Bq]: 1==1st_material ; 0==end_of_list
	if (mat<1)
      break;
    else
      source_data->activity[mat-1] = act;    //  (The first material is read as number 1 but stored as number 0)
  }



 
  // -- Init. [SECTION PHASE SPACE FILE v.2016-07-05]:
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION PHASE SPACE FILE v.2016-07-05\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION PHASE SPACE FILE v.2016-07-05")==NULL);   // Skip comments and empty lines until the section begins
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    trim_name(new_line, file_name_output);   // OUTPUT PHASE SPACE FILE NAME (no spaces)
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f %f %f %f %f", &detector_data->PSF_center.x, &detector_data->PSF_center.y, &detector_data->PSF_center.z, &detector_data->PSF_height, &detector_data->PSF_radius);   // CYLINDRIC DETECTOR CENTER, HEIGHT, AND RADIUS: X, Y, Z, H, RADIUS [cm] (IF RADIUS<0: DETECTOR CENTERED AT THE CENTER OF THE VOXELIZED GEOMETRY)
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", &detector_data->PSF_size);   // PHASE SPACE FILE SIZE (MAXIMUM NUMBER OF ELEMENTS)

  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", &detector_data->tally_TYPE);   // REPORT TRUES (=1), SCATTER (=2), OR BOTH (=0)
  if (detector_data->tally_TYPE<0 || detector_data->tally_TYPE>2)
    detector_data->tally_TYPE = 0;    // Tally all by default  

  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", &detector_data->tally_PSF_SINOGRAM);   // REPORT PSF (=1), SINOGRAM (=2), OR BOTH (=0)
  if (detector_data->tally_PSF_SINOGRAM<0 || detector_data->tally_PSF_SINOGRAM>2)
    detector_data->tally_PSF_SINOGRAM = 0;    // Tally all by default  

  // -- Init. [SECTION DOSE DEPOSITION v.2012-12-12] (MC-GPU v.1.3):
  //    Electrons are not transported and therefore we are approximating that the dose is equal to the KERMA (energy released by the photons alone).
  //    This approximation is acceptable when there is electronic equilibrium and when the range of the secondary electrons is shorter than the voxel size.
  //    Usually the doses will be acceptable for photon energies below 1 MeV. The dose estimates may not be accurate at the interface of low density volumes.
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION DOSE DEPOSITION v.2012-12-12\'!!\n");
      exit(-2);
    }
    
    if (strstr(new_line,"SECTION DOSE DEPOSITION v.2011-02-18")!=NULL)  // Detect previous version of input file
    {
      MASTER_THREAD printf("\n\n   !!read_input ERROR!! Please update the input file to the new version of MC-GPU (v1.3)!!\n\n    You simply have to change the input file text line:\n         [SECTION DOSE DEPOSITION v.2011-02-18]\n\n    for these two lines:\n         [SECTION DOSE DEPOSITION v.2012-12-12]\n         NO                              # TALLY MATERIAL DOSE? [YES/NO]\n\n");
      exit(-2);
    }
    
  }
  while(strstr(new_line,"SECTION DOSE DEPOSITION v.2012-12-12")==NULL);  // Skip comments and empty lines until the section begins
    

  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);   // TALLY MATERIAL DOSE? [YES/NO]  --> turn on/off the material dose tallied adding the Edep in each material, independently of the voxels.
  if (0==strncmp("YE",new_line,2) || 0==strncmp("Ye",new_line,2) || 0==strncmp("ye",new_line,2))
  {
    *flag_material_dose = 1;
    MASTER_THREAD printf("       Material dose deposition tally ENABLED.\n");
  }
  else if (0==strncmp("NO",new_line,2) || 0==strncmp("No",new_line,2) || 0==strncmp("no",new_line,2))
  {
    *flag_material_dose = 0;  // -- NO: disabling tally
    MASTER_THREAD printf("       Material dose deposition tally DISABLED.\n");    
  }
  else
  {
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! Answer YES or NO in the first two line of \'SECTION DOSE DEPOSITION\' to enable or disable the material dose and 3D voxel dose tallies.\n                        Input text: %s\n\n",new_line);
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-2);
  }        

  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);   // TALLY 3D VOXEL DOSE? [YES/NO] 

  if (0==strncmp("YE",new_line,2) || 0==strncmp("Ye",new_line,2) || 0==strncmp("ye",new_line,2))
  {
    // -- YES: using the tally
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); trim_name(new_line, file_dose_output);   // OUTPUT DOSE FILE NAME (no spaces)
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); sscanf(new_line, "%hd %hd", dose_ROI_x_min, dose_ROI_x_max);   // # VOXELS TO TALLY DOSE: X-index min max (first voxel has index 1)
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); sscanf(new_line, "%hd %hd", dose_ROI_y_min, dose_ROI_y_max);   // # VOXELS TO TALLY DOSE: Y-index min max
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); sscanf(new_line, "%hd %hd", dose_ROI_z_min, dose_ROI_z_max);   // # VOXELS TO TALLY DOSE: Z-index min max

    *dose_ROI_x_min -= 1; *dose_ROI_x_max -= 1;  // -Re-scale input coordinates to have index=0 for the first voxel instead of 1.
    *dose_ROI_y_min -= 1; *dose_ROI_y_max -= 1;
    *dose_ROI_z_min -= 1; *dose_ROI_z_max -= 1;

    MASTER_THREAD printf("       3D voxel dose deposition tally ENABLED.\n");
    if ( ((*dose_ROI_x_min)>(*dose_ROI_x_max)) || ((*dose_ROI_y_min)>(*dose_ROI_y_max)) || ((*dose_ROI_z_min)>(*dose_ROI_z_max)) ||
          (*dose_ROI_x_min)<0 || (*dose_ROI_y_min)<0 || (*dose_ROI_z_min)<0 )
    {
      MASTER_THREAD printf("\n\n   !!read_input ERROR!! The input region-of-interest in \'SECTION DOSE DEPOSITION\' is not valid: the minimum voxel index may not be zero or larger than the maximum index.\n");
      MASTER_THREAD printf(  "                          Input data = X[%d,%d], Y[%d,%d], Z[%d,%d]\n\n", *dose_ROI_x_min+1, *dose_ROI_x_max+1, *dose_ROI_y_min+1, *dose_ROI_y_max+1, *dose_ROI_z_min+1, *dose_ROI_z_max+1);  // Show ROI with index=1 for the first voxel instead of 0.
      #ifdef USING_MPI
        MPI_Finalize();
      #endif      
      exit(-2);
    }
    if ( ((*dose_ROI_x_min)==(*dose_ROI_x_max)) && ((*dose_ROI_y_min)==(*dose_ROI_y_max)) && ((*dose_ROI_z_min)==(*dose_ROI_z_max)) ) 
    {
      MASTER_THREAD printf("\n\n   !!read_input!! According to the input region-of-interest in \'SECTION DOSE DEPOSITION\', only the dose in the voxel (%d,%d,%d) will be tallied.\n\n",*dose_ROI_x_min,*dose_ROI_y_min,*dose_ROI_z_min);
    }
    
  }
  else if (0==strncmp("NO",new_line,2) || 0==strncmp("No",new_line,2) || 0==strncmp("no",new_line,2))
  {
    // -- NO: disabling tally
    MASTER_THREAD printf("       3D voxel dose deposition tally DISABLED.\n");
    *dose_ROI_x_min = (short int) 32500; *dose_ROI_x_max = (short int) -32500;   // Set absurd values for the ROI to make sure we never get any dose tallied
    *dose_ROI_y_min = (short int) 32500; *dose_ROI_y_max = (short int) -32500;   // (the maximum values for short int variables are +-32768)
    *dose_ROI_z_min = (short int) 32500; *dose_ROI_z_max = (short int) -32500;
  }
  else
  {
      MASTER_THREAD printf("\n\n   !!read_input ERROR!! Answer YES or NO in the first two line of \'SECTION DOSE DEPOSITION\' to enable or disable the material dose and 3D voxel dose tallies.\n                        Input text: %s\n\n",new_line);
      #ifdef USING_MPI
        MPI_Finalize();
      #endif
      exit(-2);
  }
  MASTER_THREAD printf("\n");

 // -- Init. [SECTION ENERGY PARAMETERS v.2019-04-25]:

 do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION ENERGY PARAMETERS\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION ENERGY PARAMETERS v.2019-04-25")==NULL);   // Skip comments and empty lines until the section begins

  
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f", E_resol);  // ENERGY RESOLUTION FOR THE DETECTORS 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f", E_low);  // LOW THRESHOLD FOR ENERGY WINDOW 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f", E_high);  // HIGH THRESHOLD FOR ENERGY WINDOW 

// -- Init. [SECTION SINOGRAM PARAMETERS v.2019-04-25]:

 do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION SINOGRAM PARAMETERS\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION SINOGRAM PARAMETERS v.2019-04-25")==NULL);   // Skip comments and empty lines until the section begins

    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f", FOVZ);  // AXIAL FIELD OF VIEW 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", NROWS);  // TOTAL NUMBER OF ROWS 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", NCRYSTALS);  // TOTAL NUMBER OF CRYSTALS 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", NANGLES);  // NUMBER OF ANGULAR BINS 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", NRAD);  // NUMBER OF RADIAL
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", NZS);  // NUMBER OF Z SLICES 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", RES);  // IMAGE RESOLUTION (NUMBER OF BINS IN THE IMAGE) 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", NE);  // NUMBER OF ENERGY BINS 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", MRD);  // MAXIMUM RING DIFFERENCE 
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", SPAN);  // MAXIMUM RING DIFFERENCE

    
    *NSEG=2*(floor((*MRD)/(*SPAN)))+1;
    *NSINOS=(*NSEG)*(*NZS);
    for (int aaa=1; aaa<=(*NSEG);aaa=aaa+1) {
    if (aaa>1)
    {
    *NSINOS=*NSINOS-((*SPAN)+1);
    }
    if (aaa>3)
    {
    *NSINOS=*NSINOS-2*floor((aaa-2)/2)*(*SPAN);
    }
    }
    *NBINS=(*NRAD)*(*NANGLES)*(*NSINOS);
    *NVOXS=(*RES)*(*RES)*(*NZS);  
//    printf("Total Number of Segments %i", (*NSEG)); 
//    printf("Total Number of Sinograms %i", (*NSINOS));
  // -- Init. [SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]:
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION VOXELIZED GEOMETRY FILE v.2009-11-30\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION VOXELIZED GEOMETRY FILE v.2009-11-30")==NULL);   // Skip comments and empty lines until the section begins
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
  trim_name(new_line, file_name_voxels);   // VOXEL GEOMETRY FILE (penEasy 2008 format)

  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION MATERIAL FILE LIST\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION MATERIAL")==NULL);   // Skip comments and empty lines until the section begins

  int i;
  for (i=0; i<MAX_MATERIALS; i++)
  {
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
      file_name_materials[i][0]='\n';   // The input file is allowed to finish without defining all the materials
    else
      trim_name(new_line, file_name_materials[i]);
  }    

  // [Finish reading input file]


  /////////////////////////////////////////////////////////////////////////////


  // *** Allocate array for the Phase Space File:      !!MCGPU-PET!!
  *PSF_bytes = detector_data->PSF_size * (unsigned long long int)sizeof(PSF_element_struct);
  (*PSF_ptr) = (PSF_element_struct*) malloc(*PSF_bytes);
  if (*PSF_ptr==NULL)
  {
    printf("\n\n   !!malloc ERROR!! Not enough memory to allocate %d elements of the PSF file (%lf Mbytes)!!\n\n", detector_data->PSF_size, (double)(*PSF_bytes)/(1024.0*1024.0));
    exit(-2);
  }
  else
  {
    MASTER_THREAD printf("       PSF file correctly allocated (%d elements, %lf Mbytes)\n", detector_data->PSF_size, (double)(*PSF_bytes)/(1024.0*1024.0));
  }

  // *** Initialize the PSF to 0 in the CPU. The CUDA code will init it to 0 in the GPU global memory later, using kernel "init_image_array_GPU".
  memset(*PSF_ptr, 0, (*PSF_bytes));     // Init memory space to 0.   

  
  

  // *** Allocate dose and dose^2 array if tally active:
  int num_voxels_ROI = ((int)(*dose_ROI_x_max - *dose_ROI_x_min + 1)) * ((int)(*dose_ROI_y_max - *dose_ROI_y_min + 1)) * ((int)(*dose_ROI_z_max - *dose_ROI_z_min + 1));
  if ((*dose_ROI_x_max)>-1)
  {    
    *voxels_Edep_bytes = num_voxels_ROI * sizeof(ulonglong2);
    (*voxels_Edep_ptr) = (ulonglong2*) malloc(*voxels_Edep_bytes);
    if (*voxels_Edep_ptr==NULL)
    {
      printf("\n\n   !!malloc ERROR!! Not enough memory to allocate %d voxels for the deposited dose (and uncertainty) array (%f Mbytes)!!\n\n", num_voxels_ROI, (*voxels_Edep_bytes)/(1024.f*1024.f));
      exit(-2);
    }
    else
    {
      MASTER_THREAD printf("       Array for the deposited dose ROI (and uncertainty) correctly allocated (%d voxels, %f Mbytes)\n", num_voxels_ROI, (*voxels_Edep_bytes)/(1024.f*1024.f));
    }
  }
  else
  {
    (*voxels_Edep_bytes) = 0;
  }
  
  // *** Initialize the voxel dose to 0 in the CPU. Not necessary for the CUDA code if dose matrix init. in the GPU global memory using a GPU kernel, but needed if using cudaMemcpy.  
  if ((*dose_ROI_x_max)>-1)
  {    
    memset(*voxels_Edep_ptr, 0, (*voxels_Edep_bytes));     // Init memory space to 0.
  }

  return;
}



////////////////////////////////////////////////////////////////////////////////
//! Extract a file name from an input text line, trimming the initial blanks,
//! trailing comment (#) and stopping at the first blank (the file name should
//! not contain blanks).
//!
//!       @param[in] input_line   Input sentence with blanks and a trailing comment
//!       @param[out] file_name   Trimmed file name
////////////////////////////////////////////////////////////////////////////////
void trim_name(char* input_line, char* file_name)
{
  int a=0, b=0;
  
  // Discard initial blanks:
  while(' '==input_line[a])
  {
    a++;
  }

  // Read file name until a blank or a comment symbol (#) is found:
  while ((' '!=input_line[a])&&('#'!=input_line[a]))
  {
    file_name[b] = input_line[a];
    b++;
    a++;
  }
  
  file_name[b] = '\0';    // Terminate output string
}

////////////////////////////////////////////////////////////////////////////////
//! Read a line of text and trim initial blancks and trailing comments (#).
//!
//!       @param[in] num   Characters to read
//!       @param[in] file_ptr   Pointer to the input file stream
//!       @param[out] trimmed_line   Trimmed line from input file, skipping empty lines and comments
////////////////////////////////////////////////////////////////////////////////
char* fgets_trimmed(char* trimmed_line, int num, FILE* file_ptr)
{
  char  new_line[250];
  char *new_line_ptr = NULL;
  int a=0, b=0;
  trimmed_line[0] = '\0';   //  Init with a mark that means no file input
  
  do
  {
    a=0; b=0;
    new_line_ptr = fgets(new_line, num, file_ptr);   // Read new line
    if (new_line_ptr != NULL)
    {
      // Discard initial blanks:
      while(' '==new_line[a])
      {
        a++;
      }
      // Read file until a comment symbol (#) or end-of-line are found:
      while (('\n'!=new_line[a])&&('#'!=new_line[a]))
      {
        trimmed_line[b] = new_line[a];
        b++;
        a++;
      }
    }
  } while(new_line_ptr!=NULL &&  '\0'==trimmed_line[0]);   // Keep reading lines until end-of-file or a line that is not empty or only comment is found
  
  trimmed_line[b] = '\0';    // Terminate output string
  return new_line_ptr;
}



////////////////////////////////////////////////////////////////////////////////
//! Read the voxel data and allocate the material and density matrix.
//! Also find and report the maximum density defined in the geometry.
//!
// -- Sample voxel geometry file:
//
//   #  (comment lines...)
//   #
//   #   Voxel order: X runs first, then Y, then Z.
//   #
//   [SECTION VOXELS HEADER v.2008-04-13]
//   411  190  113      No. OF VOXELS IN X,Y,Z
//   5.000e-02  5.000e-02  5.000e-02    VOXEL SIZE (cm) ALONG X,Y,Z
//   1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED
//   2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED
//   1                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)
//   [END OF VXH SECTION]
//   1 0.00120479
//   1 0.00120479
//   ...
//
//!       @param[in] file_name_voxels  Name of the voxelized geometry file.
//!       @param[out] density_max  Array with the maximum density for each material in the voxels.
//!       @param[out] voxel_data   Pointer to a structure containing the voxel number and size.
//!       @param[out] voxel_mat_dens_ptr   Pointer to the vector with the voxel materials,densities and activities //JLH
//!       @param[in] dose_ROI_x/y/z_max   Size of the dose ROI: can not be larger than the total number of voxels in the geometry.
////////////////////////////////////////////////////////////////////////////////
void load_voxels(int myID, char* file_name_voxels, float* density_max,  float* fact_object, struct voxel_struct* voxel_data, float3** voxel_mat_dens_ptr, unsigned int* voxel_mat_dens_bytes, short int* dose_ROI_x_max, short int* dose_ROI_y_max, short int* dose_ROI_z_max)
{
  char new_line[250];
  char *new_line_ptr = NULL;  
      

  MASTER_THREAD if (strstr(file_name_voxels,".zip")!=NULL)
    printf("\n\n    -- WARNING load_voxels! The input voxel file name has the extension \'.zip\'. Only \'.gz\' compression is allowed!!\n\n");     // !!zlib!!
    
  gzFile file_ptr = gzopen(file_name_voxels, "rb");  // Open the file with zlib: the file can be compressed with gzip or uncompressed.   !!zlib!!  
  
  if (file_ptr==NULL)
  {
    printf("\n\n   !! fopen ERROR load_voxels!! File %s does not exist!!\n", file_name_voxels);
    exit(-2);
  }
  MASTER_THREAD 
  {
    printf("\n    -- Reading voxel file \'%s\':\n",file_name_voxels);
    if (strstr(file_name_voxels,".gz")==NULL)
      printf("         (note that MC-GPU can also read voxel and material files compressed with gzip)\n");     // !!zlib!!  
    fflush(stdout);
  }
  do
  {    
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    
    if (new_line_ptr==NULL)
    {
      MASTER_THREAD printf("\n\n   !!Reading ERROR load_voxels!! File is not readable or does not contain the string \'[SECTION VOXELS HEADER\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"[SECTION VOXELS")==NULL);   // Skip comments and empty lines until the header begins

  float3 voxel_size;
  new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!   // Read full line (max. 250 characters).
  sscanf(new_line, "%d %d %d",&voxel_data->num_voxels.x, &voxel_data->num_voxels.y, &voxel_data->num_voxels.z);
  new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
  sscanf(new_line, "%f %f %f", &voxel_size.x, &voxel_size.y, &voxel_size.z);
  do
  {
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    if (new_line_ptr==NULL)
    {
      MASTER_THREAD printf("\n\n   !!Reading ERROR load_voxels!! File is not readable or does not contain the string \'[END OF VXH SECTION]\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"[END OF VXH SECTION")==NULL);   // Skip rest of the header

  // -- Store the size of the voxel bounding box (used in the source function):
  voxel_data->size_bbox.x = voxel_data->num_voxels.x * voxel_size.x;
  voxel_data->size_bbox.y = voxel_data->num_voxels.y * voxel_size.y;
  voxel_data->size_bbox.z = voxel_data->num_voxels.z * voxel_size.z;
  
  MASTER_THREAD printf("       Number of voxels in the input geometry file: %d x %d x %d =  %d\n", voxel_data->num_voxels.x, voxel_data->num_voxels.y, voxel_data->num_voxels.z, (voxel_data->num_voxels.x*voxel_data->num_voxels.y*voxel_data->num_voxels.z));
  MASTER_THREAD printf("       Size of the input voxels: %f x %f x %f cm  (voxel volume=%f cm^3)\n", voxel_size.x, voxel_size.y, voxel_size.z, voxel_size.x*voxel_size.y*voxel_size.z);
  MASTER_THREAD printf("       Voxel bounding box size: %f x %f x %f cm\n", voxel_data->size_bbox.x, voxel_data->size_bbox.y,  voxel_data->size_bbox.z);
  // printf("       The geometry must be given in two columns, with the voxel density in the second column.\n");
  // printf("       The  X,Y-cycles may, or may not, be separated by blank lines.\n");

  // -- Make sure the input number of voxels in the vox file is compatible with the input dose ROI (ROI assumes first voxel is index 0):
  if ( (*dose_ROI_x_max+1)>(voxel_data->num_voxels.x) || (*dose_ROI_y_max+1)>(voxel_data->num_voxels.y) || (*dose_ROI_z_max+1)>(voxel_data->num_voxels.z) )
  {
   MASTER_THREAD printf("\n       The input region of interest for the dose deposition is larger than the size of the voxelized geometry:\n");
   *dose_ROI_x_max = min_value(voxel_data->num_voxels.x-1, *dose_ROI_x_max);
   *dose_ROI_y_max = min_value(voxel_data->num_voxels.y-1, *dose_ROI_y_max);
   *dose_ROI_z_max = min_value(voxel_data->num_voxels.z-1, *dose_ROI_z_max);
   MASTER_THREAD printf(  "       updating the ROI max limits to fit the geometry -> dose_ROI_max=(%d, %d, %d)\n",*dose_ROI_x_max+1,*dose_ROI_y_max+1,*dose_ROI_z_max+1); 
   // Allowing the input of an ROI larger than the voxel volume: in this case some of the allocated memory will be wasted but the program will run ok.
  }
  
  if ( (*dose_ROI_x_max+1)==(voxel_data->num_voxels.x) && (*dose_ROI_y_max+1)==(voxel_data->num_voxels.y) && (*dose_ROI_z_max+1)==(voxel_data->num_voxels.z) )
    MASTER_THREAD printf("       The voxel dose tally ROI covers the entire voxelized phantom: the dose to every voxel will be tallied.\n");
  else
    MASTER_THREAD printf("       The voxel dose tally ROI covers only a fraction of the voxelized phantom: the dose to voxels outside the ROI will not be tallied.\n");
 
  // -- Store the inverse of the pixel sides (in cm) to speed up the particle location in voxels.
  voxel_data->inv_voxel_size.x = 1.0f/(voxel_size.x);
  voxel_data->inv_voxel_size.y = 1.0f/(voxel_size.y);
  voxel_data->inv_voxel_size.z = 1.0f/(voxel_size.z);
  
  // -- Allocate the voxel matrix and store array size:
  *voxel_mat_dens_bytes = sizeof(float3)*(voxel_data->num_voxels.x)*(voxel_data->num_voxels.y)*(voxel_data->num_voxels.z);
  *voxel_mat_dens_ptr    = (float3*) malloc(*voxel_mat_dens_bytes);
  if (*voxel_mat_dens_ptr==NULL)
  {
    printf("\n\n   !!malloc ERROR load_voxels!! Not enough memory to allocate %d voxels (%f Mbytes)!!\n\n", (voxel_data->num_voxels.x*voxel_data->num_voxels.y*voxel_data->num_voxels.z), (*voxel_mat_dens_bytes)/(1024.f*1024.f));
    exit(-2);
  }
  MASTER_THREAD printf("\n    -- Initializing the voxel material and density vector (%f Mbytes)...\n", (*voxel_mat_dens_bytes)/(1024.f*1024.f));
  MASTER_THREAD fflush(stdout);
  
  // -- Read the voxel densities:
  //   MASTER_THREAD printf("       Reading the voxel densities... ");
  int i, j, k, read_lines=0, dummy_material, read_items = -99;
  float dummy_density;
  float dummy_activity;
  float3 *voxels_ptr = *voxel_mat_dens_ptr;

  for (k=0; k<MAX_MATERIALS; k++)
    density_max[k] = -999.0f;   // Init array with an impossible low density value
  
  int Nijk = (voxel_data->num_voxels.z)*(voxel_data->num_voxels.y)*(voxel_data->num_voxels.x);
  
  for(k=0; k<(voxel_data->num_voxels.z); k++) {
  for(j=0; j<(voxel_data->num_voxels.y); j++) {
  for(i=0; i<(voxel_data->num_voxels.x); i++){        
        do { new_line_ptr = gzgets(file_ptr, new_line, 250);  }
        while (('\n'==new_line[0])||('\n'==new_line[1])||('#'==new_line[0])||('#'==new_line[1]));   // Skip empty lines and comments.
        read_items = sscanf(new_line, "%d %f %f", &dummy_material, &dummy_density, &dummy_activity);    // Read the next 3 numbers

	dummy_density *= *fact_object;

        if (read_items!=3) {
          printf("\n   !!WARNING load_voxels!! Expecting to read 3 items per voxel (material, density & activity). read_items=%d, read_lines=%d, vd=%d \n", read_items, read_lines, Nijk);
	        //printf("%d %f %f \n",dummy_material, dummy_density, dummy_activity);
  	      break;
        }

        if (dummy_material>MAX_MATERIALS)
        {
          printf("\n\n   !!ERROR load_voxels!! Voxel material number too high!! #mat=%d, MAX_MATERIALS=%d, voxel number=%d\n\n", dummy_material, MAX_MATERIALS, read_lines+1);
          exit(-2);
        }
        if (dummy_material<1)
        {
          printf("\n\n   !!ERROR load_voxels!! Voxel material number can not be zero or negative!! #mat=%d, voxel number=%dd\n\n", dummy_material, read_lines+1);
          exit(-2);
        }
        
        if (dummy_density < 1.0e-9f)
        {
          printf("\n\n   !!ERROR load_voxels!! Voxel density can not be 0 or negative: #mat=%d, density=%f, voxel number=%d\n\n", dummy_material, dummy_density, read_lines+1);
          exit(-2);
        }        
        
        if (dummy_density > density_max[dummy_material-1])
          density_max[dummy_material-1] = dummy_density;  // Store maximum density for each material

        (*voxels_ptr).x = (float)(dummy_material)+0.0001f;  // Assign material value as float (the integer value will be recovered by truncation)
        (*voxels_ptr).y = dummy_density;      	            // Assign density value
	(*voxels_ptr).z = dummy_activity;                   // Assign Activity (Bq) as float
        voxels_ptr++;                                       // Move to next voxel

        read_lines++;
      }
    }
  }
  MASTER_THREAD printf("       Total number of voxels read: %d\n",read_lines);
  gzclose(file_ptr);           // Close input file    !!zlib!!
}


////////////////////////////////////////////////////////////////////////////////
//! Read the material input files and set the mean free paths and the "linear_interp" structures.
//! Find the material nominal density. Set the Woodcock trick data.
//
// -- Sample material data file (data obtained from the PENELOPE 2006 database and models):
//
//    [MATERIAL NAME]
//     Water
//    [NOMINAL DENSITY (g/cm^3)]
//     1.000
//    [NUMBER OF DATA VALUES]
//     4096
//    [MEAN FREE PATHS :: Energy (eV) || Rayleigh | Compton | Photoelectric | Pair-production | TOTAL (cm)]
//     1.00000E+03  7.27451E-01  9.43363E+01  2.45451E-04  1.00000E+35  2.45367E-04
//     5.00000E+03  1.80004E+00  8.35996E+00  2.38881E-02  1.00000E+35  2.35089E-02
//     1.00000E+04  4.34941E+00  6.26746E+00  2.02568E-01  1.00000E+35  1.87755E-01
//     ...
//     #[RAYLEIGH INTERACTIONS (RITA sampling  of atomic form factor from EPDL database)]
//     ...
//     #[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
//     ...
//
//!       @param[in] file_name_materials    Array with the names of the material files.
//!       @param[in] density_max   maximum density in the geometry (needed to set Woodcock trick)
//!       @param[out] density_nominal   Array with the nominal density of the materials read
//!       @param[out] mfp_table_data   Constant values for the linear interpolation
//!       @param[out] mfp_table_a_ptr   First element for the linear interpolation.
//!       @param[out] mfp_table_b_ptr   Second element for the linear interpolation.
////////////////////////////////////////////////////////////////////////////////
void load_material(int myID, char file_name_materials[MAX_MATERIALS][250], float* density_max, float* density_nominal, struct linear_interp* mfp_table_data, float2** mfp_Woodcock_table_ptr, int* mfp_Woodcock_table_bytes, float3** mfp_table_a_ptr, float3** mfp_table_b_ptr, int* mfp_table_bytes, struct rayleigh_struct *rayleigh_table_ptr, struct compton_struct *compton_table_ptr)
{
  char new_line[250];
  char *new_line_ptr = NULL;
  int mat, i, bin, input_num_values = 0, input_rayleigh_values = 0, input_num_shells = 0;
  double delta_e=-99999.0;

  // -- Init the number of shells to 0 for all materials
  for (mat=0; mat<MAX_MATERIALS; mat++)
    compton_table_ptr->noscco[mat] = 0;
    

  // --Read the material data files:
  MASTER_THREAD printf("\n    -- Reading the material data files (MAX_MATERIALS=%d):\n", MAX_MATERIALS);
  for (mat=0; mat<MAX_MATERIALS; mat++)
  {
    if ((file_name_materials[mat][0]=='\0') || (file_name_materials[mat][0]=='\n'))  //  Empty file name
       continue;   // Re-start loop for next material

    MASTER_THREAD printf("         Mat %d: File \'%s\'\n", mat+1, file_name_materials[mat]);
//     printf("    -- Reading material file #%d: \'%s\'\n", mat, file_name_materials[mat]);

    gzFile file_ptr = gzopen(file_name_materials[mat], "rb");    // !!zlib!!  
    if (file_ptr==NULL)
    {
      printf("\n\n   !!fopen ERROR!! File %d \'%s\' does not exist!!\n", mat, file_name_materials[mat]);
      exit(-2);
    }
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   // Read full line (max. 250 characters).   //  !!zlib!!
      if (new_line_ptr==NULL)
      {
        printf("\n\n   !!Reading ERROR!! File is not readable or does not contain the string \'[NOMINAL DENSITY\'!!\n");
        exit(-2);
      }
    }
    while(strstr(new_line,"[NOMINAL DENSITY")==NULL);   // Skip rest of the header

    // Read the material nominal density:
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    sscanf(new_line, "# %f", &density_nominal[mat]);
    
    if (density_max[mat]>0)    //  Material found in the voxels
    {
      MASTER_THREAD printf("                Nominal density = %f g/cm^3; Max density in voxels = %f g/cm^3\n", density_nominal[mat], density_max[mat]);
    }
    else                       //  Material NOT found in the voxels
    {
      MASTER_THREAD printf("                This material is not used in any voxel.\n");
      
      // Do not lose time reading the data for materials not found in the voxels, except for the first one (needed to determine the size of the input data).      
      if (0 == mat)
        density_max[mat] = 0.01f*density_nominal[mat];   // Assign a small but positive density; this material will not be used anyway.
      else
        continue;     //  Move on to next material          
    }
      

    // --For the first material, set the number of energy values and allocate table arrays:
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    sscanf(new_line, "# %d", &input_num_values);
    if (0==mat)
    {
      mfp_table_data->num_values = input_num_values;
      MASTER_THREAD printf("                Number of energy values in the mean free path database: %d.\n", input_num_values);

      // Allocate memory for the linear interpolation arrays:
      *mfp_Woodcock_table_bytes = sizeof(float2)*input_num_values;
      *mfp_Woodcock_table_ptr   = (float2*) malloc(*mfp_Woodcock_table_bytes);  // Allocate space for the 2 parameter table
      *mfp_table_bytes = sizeof(float3)*input_num_values*MAX_MATERIALS;
      *mfp_table_a_ptr = (float3*) malloc(*mfp_table_bytes);  // Allocate space for the 4 MFP tables
      *mfp_table_b_ptr = (float3*) malloc(*mfp_table_bytes);
      *mfp_table_bytes = sizeof(float3)*input_num_values*MAX_MATERIALS;

      if (input_num_values>MAX_ENERGYBINS_RAYLEIGH)
      {
        printf("\n\n   !!load_material ERROR!! Too many energy bins (Input bins=%d): increase parameter MAX_ENERGYBINS_RAYLEIGH=%d!!\n\n", input_num_values, MAX_ENERGYBINS_RAYLEIGH);
        exit(-2);
      }
      
      if ((NULL==*mfp_Woodcock_table_ptr)||(NULL==*mfp_table_a_ptr)||(NULL==*mfp_table_b_ptr))
      {
        printf("\n\n   !!malloc ERROR!! Not enough memory to allocate the linear interpolation data: %d bytes!!\n\n", (*mfp_Woodcock_table_bytes+2*(*mfp_table_bytes)));
        exit(-2);
      }
      else
      {
        MASTER_THREAD printf("                Linear interpolation data correctly allocated (%f Mbytes)\n", (*mfp_Woodcock_table_bytes+2*(*mfp_table_bytes))/(1024.f*1024.f));
      }
      for (i=0; i<input_num_values; i++)
      {
        (*mfp_Woodcock_table_ptr)[i].x = 99999999.99f;    // Init this array with a huge MFP, the minimum values are calculated below
      }
    }
    else   // Materials after first
    {
      if (input_num_values != mfp_table_data->num_values)
      {
        printf("\n\n   !!load_material ERROR!! Incorrect number of energy values given in material \'%s\': input=%d, expected=%d\n",file_name_materials[mat], input_num_values, mfp_table_data->num_values);
        exit(-2);
      }
    }

    // -- Read the mean free paths (and Rayleigh cumulative prob):
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    double d_energy, d_rayleigh, d_compton, d_photelectric, d_total_mfp, d_pmax, e_last=-1.0;
    
    for (i=0; i<input_num_values; i++)
    {

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line,"  %le  %le  %le  %le  %le  %le", &d_energy, &d_rayleigh, &d_compton, &d_photelectric, &d_total_mfp, &d_pmax);

      // Find and store the minimum total MFP at the current energy, for every material's maximum density:
      float temp_mfp = d_total_mfp*(density_nominal[mat])/(density_max[mat]);
      if (temp_mfp < (*mfp_Woodcock_table_ptr)[i].x)
        (*mfp_Woodcock_table_ptr)[i].x = temp_mfp;       // Store minimum total mfp [cm]

      // Store the inverse MFP data points with [num_values rows]*[MAX_MATERIALS columns]
      // Scaling the table to the nominal density so that I can re-scale in the kernel to the actual local density:
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].x = 1.0/(d_total_mfp*density_nominal[mat]);   // inverse TOTAL mfp * nominal density
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].y = 1.0/(d_compton  *density_nominal[mat]);   // inverse Compton mfp * nominal density
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].z = 1.0/(d_rayleigh *density_nominal[mat]);   // inverse Rayleigh mfp * nominal density

      rayleigh_table_ptr->pmax[i*(MAX_MATERIALS)+mat] = d_pmax;    // Store the maximum cumulative probability of atomic form factor F^2 for

      if (0==i && 0==mat)
      {
        mfp_table_data->e0  = d_energy;   // Store the first energy of the first material
      }

      if (0==i)
      {
        if (fabs(d_energy-mfp_table_data->e0)>1.0e-9)
        {
          printf("\n\n   !!load_material ERROR!! Incorrect first energy value given in material \'%s\': input=%f, expected=%f\n", file_name_materials[mat], d_energy, mfp_table_data->e0);
          exit(-2);
        }
      }
      else if (1==i)
      {
        delta_e = d_energy-e_last;
      }
      else if (i>1)
      {
        if (((fabs((d_energy-e_last)-delta_e))/delta_e)>0.001)  // Tolerate up to a 0.1% relative variation in the delta e (for each bin) to account for possible precission errors reading the energy values
        {
          printf("  !!ERROR reading material data!! The energy step between mean free path values is not constant!!\n      (maybe not enough decimals given for the energy values)\n      #value = %d, First delta: %f , New delta: %f, Energy: %f ; Rel.Dif=%f\n", i, delta_e, (d_energy-e_last), d_energy,((fabs((d_energy-e_last)-delta_e))/delta_e));
          exit(-2);
        }
      }
      e_last = d_energy;
    }
    
    if (0==mat) MASTER_THREAD printf("                Lowest energy first bin = %f eV, last bin = %f eV; bin width = %f eV\n", (mfp_table_data->e0), e_last, delta_e);

    // -- Store the inverse of delta energy:
    mfp_table_data->ide = 1.0f/delta_e;

    // -- Store MFP data slope 'b' (.y for Woodcock):
    for (i=0; i<(input_num_values-1); i++)
    {
      bin = i*MAX_MATERIALS+mat;                   // Set current bin, skipping MAX_MATERIALS columns
      (*mfp_table_b_ptr)[bin].x = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].x - (*mfp_table_a_ptr)[bin].x) / delta_e;
      (*mfp_table_b_ptr)[bin].y = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].y - (*mfp_table_a_ptr)[bin].y) / delta_e;
      (*mfp_table_b_ptr)[bin].z = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].z - (*mfp_table_a_ptr)[bin].z) / delta_e;
    }
    // After maximum energy (last bin), assume constant slope:
    (*mfp_table_b_ptr)[(input_num_values-1)*MAX_MATERIALS+mat] = (*mfp_table_b_ptr)[(input_num_values-2)*MAX_MATERIALS+mat];

    // -- Rescale the 'a' parameter (.x for Woodcock) as if the bin started at energy = 0: we will not have to rescale to the bin minimum energy every time
    for (i=0; i<input_num_values; i++)
    {
      d_energy = mfp_table_data->e0 + i*delta_e;   // Set current bin lowest energy value
      bin = i*MAX_MATERIALS+mat;                   // Set current bin, skipping MAX_MATERIALS columns
      (*mfp_table_a_ptr)[bin].x = (*mfp_table_a_ptr)[bin].x - d_energy*(*mfp_table_b_ptr)[bin].x;
      (*mfp_table_a_ptr)[bin].y = (*mfp_table_a_ptr)[bin].y - d_energy*(*mfp_table_b_ptr)[bin].y;
      (*mfp_table_a_ptr)[bin].z = (*mfp_table_a_ptr)[bin].z - d_energy*(*mfp_table_b_ptr)[bin].z;
    }

    // -- Reading data for RAYLEIGH INTERACTIONS (RITA sampling  of atomic form factor from EPDL database):
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      if (gzeof(file_ptr)!=0)                           //  !!zlib!!
      {
        printf("\n\n   !!End-of-file ERROR!! Rayleigh data not found: \"#[DATA VALUES...\" in file \'%s\'. Last line read: %s\n\n", file_name_materials[mat], new_line);
        exit(-2);
      }
    }
    while(strstr(new_line,"[DATA VALUES")==NULL);   // Skip all lines until this text is found
      
    new_line_ptr = gzgets(file_ptr, new_line, 250);   // Read the number of data points in Rayleigh     //  !!zlib!! 
    sscanf(new_line, "# %d", &input_rayleigh_values);
        
    if (input_rayleigh_values != NP_RAYLEIGH)
    {
      printf("\n\n   !!ERROR!! The number of values for Rayleigh sampling is different than the allocated space: input=%d, NP_RAYLEIGH=%d. File=\'%s\'\n", input_rayleigh_values, NP_RAYLEIGH, file_name_materials[mat]);
      exit(-2);
    }
    new_line_ptr = gzgets(file_ptr, new_line, 250);    // Comment line:  #[SAMPLING DATA FROM COMMON/CGRA/: X, P, A, B, ITL, ITU]     //  !!zlib!!
    for (i=0; i<input_rayleigh_values; i++)
    {
      int itlco_tmp, ituco_tmp;
      bin = NP_RAYLEIGH*mat + i;

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line,"  %e  %e  %e  %e  %d  %d", &(rayleigh_table_ptr->xco[bin]), &(rayleigh_table_ptr->pco[bin]),
                                                  &(rayleigh_table_ptr->aco[bin]), &(rayleigh_table_ptr->bco[bin]),
                                                  &itlco_tmp, &ituco_tmp);

      rayleigh_table_ptr->itlco[bin] = (unsigned char) itlco_tmp;
      rayleigh_table_ptr->ituco[bin] = (unsigned char) ituco_tmp;
                                                  
    }
    //  printf("    -- Rayleigh sampling data read. Input values = %d\n",input_rayleigh_values);

    // -- Reading COMPTON INTERACTIONS data (relativistic impulse model with approximated one-electron analytical profiles):
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      if (gzeof(file_ptr)!=0)                           //  !!zlib!!
      {
        printf("\n\n   !!End-of-file ERROR!! Compton data not found: \"[NUMBER OF SHELLS]\" in file \'%s\'. Last line read: %s\n\n", file_name_materials[mat], new_line);
        exit(-2);
      }
    }
    while(strstr(new_line,"[NUMBER OF SHELLS")==NULL);   // Skip all lines until this text is found
    new_line_ptr = gzgets(file_ptr, new_line, 250);
    sscanf(new_line, "# %d", &input_num_shells);      // Read the NUMBER OF SHELLS
    if (input_num_shells>MAX_SHELLS)
    {
      printf("\n\n   !!ERROR!! Too many shells for Compton interactions in file \'%s\': input=%d, MAX_SHELLS=%d\n", file_name_materials[mat], input_num_shells, MAX_SHELLS);
      exit(-2);
    }
    compton_table_ptr->noscco[mat] = input_num_shells;   // Store number of shells for this material in structure
    new_line_ptr = gzgets(file_ptr, new_line, 250);      // Comment line:  #[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]
    int kzco_dummy, ksco_dummy;
    for (i=0; i<input_num_shells; i++)
    {

      bin = mat + i*MAX_MATERIALS;

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line," %e  %e  %e  %d  %d", &(compton_table_ptr->fco[bin]), &(compton_table_ptr->uico[bin]),
                                              &(compton_table_ptr->fj0[bin]), &kzco_dummy, &ksco_dummy);
    }
  
    gzclose(file_ptr);    // Material data read. Close the current material input file.           //  !!zlib!!
    
  }  // ["for" loop: continue with next material]


  // -- Store Woodcock MFP slope in component '.y':
  for (i=0; i<(mfp_table_data->num_values-1); i++)
    (*mfp_Woodcock_table_ptr)[i].y = ((*mfp_Woodcock_table_ptr)[i+1].x - (*mfp_Woodcock_table_ptr)[i].x)/delta_e;

  // -- Rescale the first parameter in component .x for Woodcock
  for (i=0; i<mfp_table_data->num_values; i++)
  {
    (*mfp_Woodcock_table_ptr)[i].x = (*mfp_Woodcock_table_ptr)[i].x - (mfp_table_data->e0 + i*delta_e)*(*mfp_Woodcock_table_ptr)[i].y;
  }
  
}
////////////////////////////////////////////////////////////////////////////////



#ifdef USING_CUDA
////////////////////////////////////////////////////////////////////////////////
//!  Select and initialize the CUDA-enabled GPU that will be used in the simulation.
//!  Allocates and copies the simulation data in the GPU global and constant memories.
//!
////////////////////////////////////////////////////////////////////////////////
void init_CUDA_device( int* gpu_id, int myID, int numprocs,
      /*Variables to GPU constant memory:*/ struct voxel_struct* voxel_data, struct source_struct* source_data, struct source_energy_struct* source_energy_data, struct detector_struct* detector_data, struct linear_interp* mfp_table_data,
      /*Variables to GPU global memory:*/ float3* voxel_mat_dens, float3** voxel_mat_dens_device, unsigned int voxel_mat_dens_bytes,
        PSF_element_struct* PSF, PSF_element_struct** PSF_device, unsigned long long int PSF_bytes, int** index_PSF_device,
        float2* mfp_Woodcock_table, float2** mfp_Woodcock_table_device, int mfp_Woodcock_table_bytes,
        float3* mfp_table_a, float3* mfp_table_b, float3** mfp_table_a_device, float3** mfp_table_b_device, int mfp_table_bytes,
        struct rayleigh_struct* rayleigh_table, struct rayleigh_struct** rayleigh_table_device,
        struct compton_struct* compton_table, struct compton_struct** compton_table_device, 
        struct detector_struct** detector_data_device, struct source_struct** source_data_device,
        ulonglong2* voxels_Edep, ulonglong2** voxels_Edep_device, int voxels_Edep_bytes, short int* dose_ROI_x_min, short int* dose_ROI_x_max, short int* dose_ROI_y_min, short int* dose_ROI_y_max, short int* dose_ROI_z_min, short int* dose_ROI_z_max,
        ulonglong2* materials_dose, ulonglong2** materials_dose_device, int flag_material_dose, int** seed_input_device, int* seed_input, unsigned long long int** total_histories_device, int num_projections, int NVOX_SIM, int** True_dev, int** Scatter_dev, int** Imagen_T_dev, int** Imagen_SC_dev, int** Energy_Spectrum_dev, int* NBINS, int* NE )
{    
  cudaDeviceProp deviceProp;
  int deviceCount;  
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  if (0==deviceCount)
  {
    printf("\n  !!ERROR!! No CUDA enabled GPU detected by thread #%d!!\n\n", myID);
    exit(-1);
  }  
  
  
#ifdef USING_MPI      
  if (numprocs>1)
  {      
    // *** Select the appropriate GPUs in the different workstations in the MPI hostfile:
    //     The idea is that each threads will wait for the previous thread to send a messages with its processor name and GPU id, 
    //     then it will assign the current GPU, and finally it will notify the following thread:    
    const int NODE_NAME_LENGTH = 31;
    char processor_name[NODE_NAME_LENGTH+1], previous_processor_name[NODE_NAME_LENGTH+1];
    int resultlen = -1;
    
    MPI_Get_processor_name(processor_name, &resultlen);
    
    MPI_Status status;
    
    int gpu_id_to_avoid = *gpu_id;

    clock_t clock_start;
    if (myID == (numprocs-1))
      clock_start = clock();        

    // Unless we are the first thread, wait for a message from the previous thread:
    // The MPI_Recv command will block the execution of the code until the previous threads have communicated and shared the appropriate information.
    if (0!=myID)
    {     
      MPI_Recv(previous_processor_name, NODE_NAME_LENGTH, MPI_CHAR, myID-1, 111, MPI_COMM_WORLD, &status);   // Receive the processor name and gpu_id from the previous thread
          // printf("\n -> MPI_Recv thread %d: gpu_id=%d, %s\n", myID, (int)previous_processor_name[NODE_NAME_LENGTH-1], previous_processor_name); fflush(stdout);  //!!Verbose!! 
    }
    
    // Compare the 30 first characters of the 2 names to see if we changed the node, except for the first thread that allways gets GPU 0:
    if ((0==myID) || (0!=strncmp(processor_name, previous_processor_name, NODE_NAME_LENGTH-1)))
    { 
      *gpu_id = 0;    // Thread in a new node: assign to GPU 0:
    }
    else
    {
      // Current thread in the same node as the previous one: assign next GPU (previous GPU id given in element NODE_NAME_LENGTH-1 of the array)
      *gpu_id = (int)previous_processor_name[NODE_NAME_LENGTH-1] + 1;
    }

    // Set the following GPU if this is the one to be skipped (given in the input file):
    if (*gpu_id == gpu_id_to_avoid)
    {
      *gpu_id = *gpu_id + 1;  
      printf("             Skipping GPU %d in thread %d (%s), as selected in the input file: gpu_id=%d\n", gpu_id_to_avoid, myID, processor_name, *gpu_id); fflush(stdout);
    }
    
  
  
    //!!DeBuG!! MC-GPU_v1.4!! Skip GPUs connected to a monitor, if more GPUs available:
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *gpu_id));    
    if (0!=deviceProp.kernelExecTimeoutEnabled)                                 //!!DeBuG!! 
    {
      if((*gpu_id)<(deviceCount-1))                                             //!!DeBuG!! 
      {      
        printf("\n       ==> CUDA: GPU #%d is connected to a display and the CUDA driver would limit the kernel run time. Skipping this GPU!!\n", *gpu_id); //!!DeBuG!!
        *gpu_id = (*gpu_id)+1;                                                  //!!DeBuG!!
      }
    }
  
       
    // Send the processor and GPU id to the following thread, unless we are the last thread:
    if (myID != (numprocs-1))
    { 
      processor_name[NODE_NAME_LENGTH-1] = (char)(*gpu_id);  // Store GPU number in the last element of the array
      
          // printf(" <- MPI_Send thread %d: gpu_id=%d, %s\n", myID, (int)processor_name[NODE_NAME_LENGTH-1], processor_name); fflush(stdout);  //!!Verbose!!
      MPI_Send(processor_name, NODE_NAME_LENGTH, MPI_CHAR, myID+1, 111, MPI_COMM_WORLD);  // Send processor name and gpu_id to the following thread (tag is the current thread id)
    }
    else
    {
      printf("           -- Time spent communicating between threads to determine the GPU id to use in each thread: %.6f s\n", ((double)(clock()-clock_start))/CLOCKS_PER_SEC); fflush(stdout);
    }    
  }  
#endif  


  if (*gpu_id>=deviceCount)
  {
    printf("\n\n  !!WARNING!! The selected GPU number is too high, this device number does not exist!! GPU_id (starting at 0)=%d, deviceCount=%d\n", (*gpu_id), deviceCount); fflush(stdout);
    if (numprocs==1)
    {
      *gpu_id = gpuGetMaxGflopsDeviceId();
      printf("            Selecting the fastest GPU available using gpuGetMaxGflopsDeviceId(): GPU_id = %d\n\n", (*gpu_id)); fflush(stdout);
    }    
    else
    {
      exit(-1);    
    }
  }     

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *gpu_id));   // Re-load card properties in case we chaged gpu_id
  if (deviceProp.major>99 || deviceProp.minor>99)
  {
    printf("\n\n\n  !!ERROR!! The selected GPU device does not support CUDA!! GPU_id=%d, deviceCount=%d, compute capability=%d.%d\n\n\n", (*gpu_id), deviceCount, deviceProp.major,deviceProp.minor);
    exit(-1);
  }
  
  checkCudaErrors(cudaSetDevice(*gpu_id));   // Set the GPU device. (optionally use: cutGetMaxGflopsDeviceId())
        
  if (deviceProp.major>1)
  {
    
#ifdef LARGE_CACHE  
    // -- Compute capability > 1: set a large L1 cache for the global memory, reducing the size of the shared memory:
    //       cudaFuncCachePreferShared: shared memory is 48 KB
    //       cudaFuncCachePreferL1: shared memory is 16 KB
    //       cudaFuncCachePreferNone: no preference
    printf("\n       ==> CUDA: LARGE_CACHE defined --> setting a large global memory cache (L1) and a small shared memory (cudaFuncCachePreferL1).\n");
    cudaFuncSetCacheConfig(track_particles, cudaFuncCachePreferL1);            // -- Set a large cache instead of a large shared memory.
        // #else
        // -- Using default:
        // printf("\n       ==> CUDA: LARGE_CACHE not defined --> setting a large shared memory and a small global memory cache (cudaFuncCachePreferShared).\n");
        //    cudaFuncSetCacheConfig(track_particles, cudaFuncCachePreferShared);            // !!DeBuG!! Setting size of shared memory/global cache
#endif

  }

  // DISCONTINUED CUDA FUNCTION!   register int GPU_cores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;    // CUDA SDK function to get the number of GPU cores

  // -- Reading the device properties:
  
#ifdef USING_MPI   
  printf("\n       ==> CUDA (MPI process #%d): %d CUDA enabled GPU detected! Using device #%d: \"%s\"\n", myID, deviceCount, (*gpu_id), deviceProp.name);    
#else  
  printf("\n       ==> CUDA: %d CUDA enabled GPU detected! Using device #%d: \"%s\"\n", deviceCount, (*gpu_id), deviceProp.name);    
#endif
  printf("                 Compute capability: %d.%d, Number multiprocessors: %d\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    // printf("                 Compute capability: %d.%d, Number multiprocessors: %d, Number cores: %d\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount, GPU_cores);
  printf("                 Clock rate: %.2f GHz, Global memory: %.3f Mbyte, Constant memory: %.2f kbyte\n", deviceProp.clockRate*1.0e-6f, deviceProp.totalGlobalMem/(1024.f*1024.f), deviceProp.totalConstMem/1024.f);
  printf("                 Shared memory per block: %.2f kbyte, Registers per block: %.2f kbyte\n", deviceProp.sharedMemPerBlock/1024.f, deviceProp.regsPerBlock/1024.f);
  int driverVersion = 0, runtimeVersion = 0;  
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("                 CUDA Driver Version: %d.%d, Runtime Version: %d.%d\n\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);

  if (0!=deviceProp.kernelExecTimeoutEnabled)
  {
    printf("\n\n\n   !!WARNING!! The selected GPU is connected to a display and therefore CUDA driver will limit the kernel run time to 5 seconds and the simulation will likely fail!!\n");
    printf( "              You can fix this by executing the simulation in a different GPU (select number in the input file) or by turning off the window manager and using the text-only Linux shell.\n\n\n");
    // exit(-1);
  }    

  fflush(stdout);
  
  clock_t clock_init = clock();    

  // -- Allocate the constant variables in the device:
  checkCudaErrors(cudaMemcpyToSymbol(voxel_data_CONST,     voxel_data,     sizeof(struct voxel_struct)));
  checkCudaErrors(cudaMemcpyToSymbol(source_energy_data_CONST, source_energy_data, sizeof(struct source_energy_struct)));
  
// Source, detector data now copied to global memory and transfered to shared memory in the kernel. OLD CODE:  checkCudaErrors(cudaMemcpyToSymbol(detector_data_CONST,  detector_data,  sizeof(struct detector_struct)));
  
  checkCudaErrors(cudaMemcpyToSymbol(mfp_table_data_CONST, mfp_table_data, sizeof(struct linear_interp)));

  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_x_min_CONST, dose_ROI_x_min, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_x_max_CONST, dose_ROI_x_max, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_y_min_CONST, dose_ROI_y_min, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_y_max_CONST, dose_ROI_y_max, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_z_min_CONST, dose_ROI_z_min, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_z_max_CONST, dose_ROI_z_max, sizeof(short int)));
  

  double total_mem = sizeof(struct voxel_struct)+sizeof(struct source_struct)+sizeof(struct detector_struct)+sizeof(struct linear_interp) + 6*sizeof(short int);
  MASTER_THREAD printf("       ==> CUDA: Constant data successfully copied to the device. CONSTANT memory used: %lf kbytes (%.1lf%%)\n", total_mem/1024.0, 100.0*total_mem/deviceProp.totalConstMem);
  

  // -- Allocate the device global memory:

  if (*dose_ROI_x_max > -1)  // Allocate dose array only if the tally is not disabled
  {
    checkCudaErrors(cudaMalloc((void**) voxels_Edep_device, voxels_Edep_bytes));
    if (*voxels_Edep_device==NULL)
    {
      printf("\n cudaMalloc ERROR!! Error allocating the dose array on the device global memory!! (%lf Mbytes)\n", voxels_Edep_bytes/(1024.0*1024.0));
      exit(-1);
    }
  }
  
  checkCudaErrors(cudaMalloc((void**) voxel_mat_dens_device, voxel_mat_dens_bytes));
  checkCudaErrors(cudaMalloc((void**) mfp_Woodcock_table_device, mfp_Woodcock_table_bytes));
  checkCudaErrors(cudaMalloc((void**) mfp_table_a_device,    mfp_table_bytes));
  checkCudaErrors(cudaMalloc((void**) mfp_table_b_device,    mfp_table_bytes));
  checkCudaErrors(cudaMalloc((void**) rayleigh_table_device, sizeof(struct rayleigh_struct)));
  checkCudaErrors(cudaMalloc((void**) compton_table_device,  sizeof(struct compton_struct))); 
  checkCudaErrors(cudaMalloc((void**) detector_data_device,  num_projections*sizeof(struct detector_struct)));
  checkCudaErrors(cudaMalloc((void**) source_data_device,    num_projections*sizeof(struct source_struct)));    // The array of detectors, sources has "MAX_NUM_PROJECTIONS" elements but I am allocating only the used "num_projections" elements to the GPU
  
  checkCudaErrors(cudaMalloc((void**) seed_input_device, sizeof(int)));  // Store latest random seed used in GPU in global memory to continue random sequence in consecutive projections.   !!DBTv1.4!!
  checkCudaErrors(cudaMalloc((void**) total_histories_device, sizeof(unsigned long long int)));  

  checkCudaErrors(cudaMalloc((void**) PSF_device,       PSF_bytes));      // !!MCGPU-PET!!
  checkCudaErrors(cudaMalloc((void**) index_PSF_device, sizeof(int)));    // !!MCGPU-PET!!
  
  checkCudaErrors(cudaMalloc((void**) True_dev, (*NBINS)*sizeof(int)));   
  checkCudaErrors(cudaMalloc((void**) Scatter_dev, (*NBINS)*sizeof(int)));    
  checkCudaErrors(cudaMalloc((void**) Imagen_T_dev, NVOX_SIM*sizeof(int)));    
  checkCudaErrors(cudaMalloc((void**) Imagen_SC_dev, NVOX_SIM*sizeof(int)));    
  checkCudaErrors(cudaMalloc((void**) Energy_Spectrum_dev, (*NE)*sizeof(int))); 

  //JLH
  
  if (flag_material_dose==1)
    checkCudaErrors(cudaMalloc((void**) materials_dose_device, MAX_MATERIALS*sizeof(ulonglong2)));    // !!tally_materials_dose!!
  
  total_mem = voxels_Edep_bytes + voxel_mat_dens_bytes + PSF_bytes + mfp_Woodcock_table_bytes + 2*mfp_table_bytes + sizeof(struct compton_struct) + sizeof(struct rayleigh_struct) + num_projections*(sizeof(struct detector_struct) + sizeof(struct source_struct));
  if (*voxel_mat_dens_device==NULL || *PSF_device==NULL || *mfp_Woodcock_table_device==NULL || *mfp_table_a_device==NULL ||
      *mfp_table_a_device==NULL || *rayleigh_table_device==NULL || *compton_table_device==NULL || *detector_data_device==NULL || *source_data_device==NULL)
  {
    printf("\n cudaMalloc ERROR!! Device global memory not correctly allocated!! (%lf Mbytes)\n", total_mem/(1024.0*1024.0));
    exit(-1);
  }
  else
  {
    MASTER_THREAD printf("       ==> CUDA: Device global memory correctly allocated. GLOBAL memory used: %lf Mbytes (%.1lf%%)\n", total_mem/(1024.0*1024.0), 100.0*total_mem/deviceProp.totalGlobalMem);
  }

  // --Copy the host memory to the device:
  checkCudaErrors(cudaMemcpy(*voxel_mat_dens_device, voxel_mat_dens, voxel_mat_dens_bytes,                          cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*mfp_Woodcock_table_device, mfp_Woodcock_table, mfp_Woodcock_table_bytes,              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*mfp_table_a_device,    mfp_table_a,    mfp_table_bytes,                               cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*mfp_table_b_device,    mfp_table_b,    mfp_table_bytes,                               cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*rayleigh_table_device, rayleigh_table, sizeof(struct rayleigh_struct),                cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*compton_table_device,  compton_table,  sizeof(struct compton_struct),                 cudaMemcpyHostToDevice));  
  checkCudaErrors(cudaMemcpy(*detector_data_device,  detector_data,  num_projections*sizeof(struct detector_struct),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*source_data_device,    source_data,    num_projections*sizeof(struct source_struct),  cudaMemcpyHostToDevice));  
  
  checkCudaErrors(cudaMemcpy(*seed_input_device, seed_input, sizeof(int), cudaMemcpyHostToDevice));    // Upload initial seed value.   !!DBTv1.4!!
  
  unsigned long long int total_histories_init = (unsigned long long int)0;
  checkCudaErrors(cudaMemcpy(*total_histories_device, &total_histories_init, sizeof(unsigned long long int), cudaMemcpyHostToDevice));   // !!MCGPU-PET!!
  checkCudaErrors(cudaMemcpy(*PSF_device, PSF, PSF_bytes, cudaMemcpyHostToDevice));                                                      // !!MCGPU-PET!!
  int zero=0;
  checkCudaErrors(cudaMemcpy(*index_PSF_device, &zero, sizeof(int), cudaMemcpyHostToDevice));                                            // !!MCGPU-PET!! 

/*    !!MCGPU-PET!!
 *
  
  //   --Init the image array to 0 using a GPU kernel instead of cudaMemcpy:
  //     Simple version: checkCudaErrors( cudaMemcpy( image_device, image, image_bytes, cudaMemcpyHostToDevice) );

  int pixels_per_image = detector_data[0].num_pixels.x * detector_data[0].num_pixels.y;
  MASTER_THREAD printf("       ==> CUDA: Launching kernel to initialize the device image to 0: number of blocks = %d, threads per block = 128\n", (int)(ceil(pixels_per_image/128.0f)+0.01f) );

  init_image_array_GPU<<<(int)(ceil(pixels_per_image/128.0f)+0.01f),128>>>(*image_device, pixels_per_image);
  
  fflush(stdout);
  cudaDeviceSynchronize();      // Force the runtime to wait until all device tasks have completed
  getLastCudaError("\n\n !!Kernel execution failed initializing the image array!! ");  // Check if kernel execution generated any error:
  
*/

  //   --Init the dose array to 0 using a GPU kernel, if the tally is not disabled:
  if (*dose_ROI_x_max > -1)
  {      
    
    MASTER_THREAD printf("       ==> CUDA: Initialize the device dose deposition to 0 using cudaMemcpy.\n");
    checkCudaErrors(cudaMemcpy(*voxels_Edep_device, voxels_Edep, voxels_Edep_bytes, cudaMemcpyHostToDevice) );
   
/*  // -- OPTIONAL CODE: Launch kernel to initialize the device dose deposition to 0 (MAY FAIL IF DOSE MATRIX IS TOO BIG!)    !!DeBuG!!
    int num_voxels_dose = voxels_Edep_bytes/sizeof(ulonglong2);   // Calculate the number of voxels in the dose array
    int num_blocks, num_threads_block = 0;  
    // Select the number of threads per block making sure we don't try to launch more blocks than CUDA's maximum value:
    do
    {
      num_threads_block += 64;
      num_blocks = (int)(ceil(((double)num_voxels_dose)/((double)num_threads_block))+0.001);
    }
    while (num_blocks > 65500);    
    MASTER_THREAD printf("       ==> CUDA: Launching kernel to initialize the device dose deposition to 0: number of blocks = %d, threads per block = %d\n", num_blocks, num_threads_block);  
    init_dose_array_GPU<<<num_blocks,num_threads_block>>>(*voxels_Edep_device, num_voxels_dose);    
      cudaDeviceSynchronize();
      getLastCudaError("\n\n !!Kernel execution failed initializing the dose array!! ");  // Check if kernel execution generated any error:
*/

  }
  
  // Init materials_dose array in GPU with 0 (same as host):
  if (flag_material_dose==1)
    checkCudaErrors(cudaMemcpy(*materials_dose_device, materials_dose, MAX_MATERIALS*sizeof(ulonglong2), cudaMemcpyHostToDevice));   // !!tally_materials_dose!!
  
  MASTER_THREAD printf("                 Time spent allocating and copying memory to the device: %.6f s\n", float(clock()-clock_init)/CLOCKS_PER_SEC);    

}


////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//! Guestimate a good number of blocks to estimate the speed of different generations 
//! of GPUs. Slower GPUs will simulate less particles and hopefully the fastest GPUs 
//! will not have to wait much. If the speed is not accurately estimated in the speed test
//! some GPUs will simulate longer than others and valuable simulation time will be wasted 
//! in the idle GPUs.
//!
//! In this function the "optimum" number of blocks for the speed test is heuristically 
//! computed as the product of three GPU characteristics:
//!   [2.0] * [number of GPU cores] * [core frequency] * [major CUDA compute capability] + [100]
//!
//! The factor 2.0 is arbitrary and can be modified depending on the case (for short 
//! simulations this value may have to be reduced or the speed test will take longer 
//! than the whole simulation). The constant 100 blocks are added to try to get enough 
//! blocks for a reliable timing of slow GPUs.
//!
//! For example, an NVIDIA GeForce 290 will get:
//!   2.0 * 240 (cores) * 1.24 (GHz) * 1 (major compute capability) + 100 =  695.2 ~  695 blocks
//! An NVIDIA GeForce 580 will get:
//!   2.0 * 512 (cores) * 1.54 (GHz) * 2 (major compute capability) + 100 = 3253.9 ~ 3254 blocks 
//! In total the 580 gets 5.7 times more blocks than the 290.
//!
//!       @param[in] gpu_id   GPU number
//!       @param[out] num_blocks   Returns a number of blocks related to the expected GPU speed
////////////////////////////////////////////////////////////////////////////////
int guestimate_GPU_performance(int gpu_id)
{          
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpu_id); 
  // DISCONTINUED CUDA FUNCTION! float num_cores       = (float) _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
  float num_cores_aprox = 128 * deviceProp.multiProcessorCount;   // I can't get the exact number of cores anymore; assume 128 per multiprocessor
  float comp_capability = (float) deviceProp.major;
  float frequency       = deviceProp.clockRate*1.0e-6f;
  
  int guestimated_value = (int)(0.5f*num_cores_aprox*frequency*comp_capability + 64.0f);
  return min_value(guestimated_value, 1024);     // Limit the returned number of blocks to prevent too long speed tests   !!DBT!!
}
  
  
#endif
////////////////////////////////////////////////////////////////////////////////




/*    !!MCGPU-PET!! Eliminating image reporting...
 * 
 *
////////////////////////////////////////////////////////////////////////////////
//! Report the tallied image in ASCII and binary form (32-bit floats).
//! Separate images for primary and scatter radiation are generated.
//! 
//!
//!       @param[in] file_name_output   File where tallied image is reported
//!       @param[in] detector_data   Detector description read from the input file (pointer to detector_struct)
//!       @param[in] image  Tallied image (in meV per pixel)
//!       @param[in] time_elapsed   Time elapsed during the main loop execution (in seconds)
//!       @param[in] total_histories   Total number of x-rays simulated
////////////////////////////////////////////////////////////////////////////////
int report_image(char* file_name_output, struct detector_struct* detector_data, struct source_struct* source_data, float mean_energy_spectrum, unsigned long long int* image, double time_elapsed, unsigned long long int total_histories, int current_projection, int num_projections, double D_angle, double initial_angle, int myID, int numprocs)
{
  
  //  -Find current angle
  double current_angle = initial_angle+current_projection*D_angle;

  // -- Report data:
  printf("\n\n          *** IMAGE TALLY PERFORMANCE REPORT ***\n");
  
  if(num_projections!=1)   // Output the projection angle when simulating a CT:
  {
    printf("              CT projection %d of %d: angle from X axis = %lf \n", current_projection+1, num_projections, current_angle*RAD2DEG);
  }
  
  printf("              Simulated x rays:    %lld\n", total_histories);
  printf("              Simulation time [s]: %.2f\n", time_elapsed);
  if (time_elapsed>0.000001)
    printf("              Speed [x-rays/s]:    %.2f\n\n", ((double)total_histories)/time_elapsed);

  FILE* file_ptr = fopen(file_name_output, "w");
  
  if (file_ptr==NULL)
  {
    printf("\n\n   !!fopen ERROR report_image!! File %s can not be opened!!\n", file_name_output);
    exit(-3);
  }
  
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "#     ***         MC-GPU, version 1.3 (http://code.google.com/p/mcgpu/)         ***\n");
  fprintf(file_ptr, "#     ***                                                                       ***\n");
  fprintf(file_ptr, "#     ***                     Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)     ***\n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "# \n");  
#ifdef USING_CUDA
  fprintf(file_ptr, "#  *** SIMULATION IN THE GPU USING CUDA ***\n");
#else
  fprintf(file_ptr, "#  *** SIMULATION IN THE CPU ***\n");
#endif  
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  Image created counting the energy arriving at each pixel: ideal energy integrating detector.\n");
  fprintf(file_ptr, "#  Pixel value units: eV/cm^2 per history (energy fluence).\n");


  if(num_projections!=1)   // Output the projection angle when simulating a CT:
  {
    fprintf(file_ptr, "#  CT projection %d of %d: angle from X axis = %lf \n", current_projection+1, num_projections, current_angle*RAD2DEG);
  }  

  fprintf(file_ptr, "#  Focal spot position = (%.8f,%.8f,%.8f), cone beam direction = (%.8f,%.8f,%.8f)\n", source_data[current_projection].position.x, source_data[current_projection].position.y, source_data[current_projection].position.z, source_data[current_projection].direction.x, source_data[current_projection].direction.y, source_data[current_projection].direction.z);

  fprintf(file_ptr, "#  Pixel size:  %lf x %lf = %lf cm^2\n", 1.0/(double)(detector_data[0].inv_pixel_size_X), 1.0/(double)(detector_data[0].inv_pixel_size_Z), 1.0/(double)(detector_data[0].inv_pixel_size_X*detector_data[0].inv_pixel_size_Z));
  
  fprintf(file_ptr, "#  Number of pixels in X and Z:  %d  %d\n", detector_data[0].num_pixels.x, detector_data[0].num_pixels.y);
  fprintf(file_ptr, "#  (X rows given first, a blank line separates the different Z values)\n");
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#  [NON-SCATTERED] [COMPTON] [RAYLEIGH] [MULTIPLE-SCATTING]\n");
  fprintf(file_ptr, "# ==========================================================\n");

  const double SCALE = 1.0/SCALE_eV;    // conversion to eV using the inverse of the constant used in the "tally_image" kernel function (defined in the header file)
  const double NORM = SCALE * detector_data[0].inv_pixel_size_X * detector_data[0].inv_pixel_size_Z / ((double)total_histories);  // ==> [eV/cm^2 per history]
  double energy_noScatter, energy_compton, energy_rayleigh, energy_multiscatter;
  double energy_integral = 0.0;   // Integrate (add) the energy in the image pixels [meV]
  double maximum_energy_pixel = -100.0;  // Find maximum pixel signal
  int maximum_energy_pixel_x=0, maximum_energy_pixel_y=0, maximum_energy_pixel_number=0;   

  int pixels_per_image = (detector_data[0].num_pixels.x*detector_data[0].num_pixels.y), pixel=0;
  int i, j;
  for(j=0; j<detector_data[0].num_pixels.y; j++)
  {
    for(i=0; i<detector_data[0].num_pixels.x; i++)
    {
      energy_noScatter    = (double)(image[pixel]);
      energy_compton      = (double)(image[pixel +   pixels_per_image]);
      energy_rayleigh     = (double)(image[pixel + 2*pixels_per_image]);
      energy_multiscatter = (double)(image[pixel + 3*pixels_per_image]);

      // -- Write the results in an external file; the image corresponding to all particles not written: it has to be infered adding all images
      fprintf(file_ptr, "%.8lf %.8lf %.8lf %.8lf\n", NORM*energy_noScatter, NORM*energy_compton, NORM*energy_rayleigh, NORM*energy_multiscatter);
      
      register double total_energy_pixel = energy_noScatter + energy_compton + energy_rayleigh + energy_multiscatter;   // Find and report the pixel with maximum signal
      if (total_energy_pixel>maximum_energy_pixel)
      {
        maximum_energy_pixel = total_energy_pixel;
        maximum_energy_pixel_x = i;
        maximum_energy_pixel_y = j;
        maximum_energy_pixel_number = pixel;
      }            
      energy_integral += total_energy_pixel;   // Count total energy in the whole image      
      
      pixel++;   // Move to next pixel
    }
    fprintf(file_ptr, "\n");     // Separate rows with an empty line for visualization with gnuplot.
  }
  
  fprintf(file_ptr, "#   *** Simulation REPORT: ***\n");
  fprintf(file_ptr, "#       Fraction of energy detected (over the mean energy of the spectrum): %.3lf%%\n", 100.0*SCALE*(energy_integral/(double)(total_histories))/(double)(mean_energy_spectrum));
  fprintf(file_ptr, "#       Maximum energy detected in pixel %i: (x,y)=(%i,%i) -> pixel value = %lf eV/cm^2\n", maximum_energy_pixel_number, maximum_energy_pixel_x, maximum_energy_pixel_y, NORM*maximum_energy_pixel);
  fprintf(file_ptr, "#       Simulated x rays:    %lld\n", total_histories);
  fprintf(file_ptr, "#       Simulation time [s]: %.2f\n", time_elapsed);
  if (time_elapsed>0.000001)
    fprintf(file_ptr, "#       Speed [x-rays/sec]:  %.2f\n\n", ((double)total_histories)/time_elapsed);
   
  fclose(file_ptr);  // Close output file and flush stream

  printf("              Fraction of initial energy arriving at the detector (over the mean energy of the spectrum):  %.3lf%%\n", 100.0*SCALE*(energy_integral/(double)(total_histories))/(double)(mean_energy_spectrum));
  printf("              Maximum energy detected in pixel %i: (x,y)=(%i,%i). Maximum pixel value = %lf eV/cm^2\n\n", maximum_energy_pixel_number, maximum_energy_pixel_x, maximum_energy_pixel_y, NORM*maximum_energy_pixel);  
  fflush(stdout);
  
  
  // -- Binary output:   
  float energy_float;
  char file_binary[250];
  strncpy (file_binary, file_name_output, 250);
  strcat(file_binary,".raw");                       // !!BINARY!! 
  FILE* file_binary_ptr = fopen(file_binary, "w");  // !!BINARY!!
  if (file_binary_ptr==NULL)
  {
    printf("\n\n   !!fopen ERROR report_image!! Binary file %s can not be opened for writing!!\n", file_binary);
    exit(-3);
  }
  
  for(i=0; i<pixels_per_image; i++)
  {
    energy_float = (float)( NORM * (double)(image[i] + image[i + pixels_per_image] + image[i + 2*pixels_per_image] + image[i + 3*pixels_per_image]) );  // Total image (scatter + primary)
    fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);   // Write pixel data in a binary file that can be easyly open in imageJ. !!BINARY!!
  }
  for(i=0; i<pixels_per_image; i++)
  {
    energy_float = (float)( NORM * (double)(image[i]) );  // Non-scattered image
    fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  }
  for(i=0; i<pixels_per_image; i++)
  {
    energy_float = (float)( NORM * (double)(image[i + pixels_per_image]) );  // Compton image
    fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  }
  for(i=0; i<pixels_per_image; i++)
  {
    energy_float = (float)( NORM * (double)(image[i + 2*pixels_per_image]) );  // Rayleigh image
    fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  }
  for(i=0; i<pixels_per_image; i++)
  {
    energy_float = (float)( NORM * (double)(image[i + 3*pixels_per_image]) );  // Multiple-scatter image
    fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  }       
  
  fclose(file_binary_ptr);    
  
    
  return 0;     // Report could return not 0 to continue the simulation...
}

*/


///////////////////////////////////////////////////////////////////////////////




///////////////////////////////////////////////////////////////////////////////
//! Report the total tallied 3D voxel dose deposition for all projections.
//! The voxel doses in the input ROI and their respective uncertainties are reported 
//! in binary form (32-bit floats) in two separate .raw files.
//! The dose in a single plane at the level of the focal spot is also reported in  
//! ASCII format for simple visualization with GNUPLOT.
//! The total dose deposited in each different material is reported to the standard output.
//! The material dose is calculated adding the energy deposited in the individual voxels 
//! within the dose ROI, and dividing by the total mass of the material in the ROI.
//!
//!       @param[in] file_dose_output   File where tallied image is reported
//!       @param[in] detector_data   Detector description read from the input file (pointer to detector_struct)
//!       @param[in] image  Tallied image (in meV per pixel)
//!       @param[in] time_elapsed   Time elapsed during the main loop execution (in seconds)
//!       @param[in] total_histories   Total number of x-rays simulated
//!       @param[in] source_data   Data required to compute the voxel plane to report in ASCII format: Z at the level of the source, 1st projection
////////////////////////////////////////////////////////////////////////////////
int report_voxels_dose(char* file_dose_output, int num_projections, struct voxel_struct* voxel_data, float3* voxel_mat_dens, ulonglong2* voxels_Edep, double time_total_MC_init_report, unsigned long long int total_histories, short int dose_ROI_x_min, short int dose_ROI_x_max, short int dose_ROI_y_min, short int dose_ROI_y_max, short int dose_ROI_z_min, short int dose_ROI_z_max, struct source_struct* source_data)
{

  printf("\n\n          *** VOXEL ROI DOSE TALLY REPORT ***\n\n");
    
  FILE* file_ptr = fopen(file_dose_output, "w");
  if (file_ptr==NULL)
  {
    printf("\n\n   !!fopen ERROR report_voxels_dose!! File %s can not be opened!!\n", file_dose_output);
    exit(-3);
  }    
    
  // -- Binary output:                                         // !!BINARY!!  
  char file_binary_mean[250], file_binary_sigma[250];
  strncpy (file_binary_mean, file_dose_output, 250);
  strcat(file_binary_mean,".raw");                     
  strncpy (file_binary_sigma, file_dose_output, 250);
  strcat(file_binary_sigma,"_PercentRelError2sigma.raw");    
  FILE* file_binary_mean_ptr  = fopen(file_binary_mean, "w");  // !!BINARY!!
  FILE* file_binary_sigma_ptr = fopen(file_binary_sigma, "w");       // !!BINARY!!
  if (file_binary_mean_ptr==NULL)
  {
    printf("\n\n   !!fopen ERROR report_voxels_dose!! Binary file %s can not be opened!!\n", file_dose_output);
    exit(-3);
  }
  
  int DX = dose_ROI_x_max - dose_ROI_x_min + 1,
      DY = dose_ROI_y_max - dose_ROI_y_min + 1,
      DZ = dose_ROI_z_max - dose_ROI_z_min + 1;           
      
  // -- Calculate the dose plane that will be output as ASCII text:
  int z_plane_dose = (dose_ROI_z_max+dose_ROI_z_min)/2;
  
  int z_plane_dose_ROI = z_plane_dose - dose_ROI_z_min;

  printf("              Reporting the 3D voxel dose distribution as binary floats in the .raw file, and the 2D dose for Z plane %d as ASCII text.\n", z_plane_dose);
//   printf("              Also reporting the dose to each material inside the input ROI adding the energy deposited in each individual voxel\n");
//   printf("              (these material dose results will be equal to the materials dose tally below if the ROI covers all the voxels).\n");
  
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "#     ***         MC-GPU, version 1.3 (http://code.google.com/p/mcgpu/)         ***\n");
  fprintf(file_ptr, "#     ***                                                                       ***\n");
  fprintf(file_ptr, "#     ***                     Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)     ***\n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "# \n");  
#ifdef USING_CUDA
  fprintf(file_ptr, "#  *** SIMULATION IN THE GPU USING CUDA ***\n");
#else
  fprintf(file_ptr, "#  *** SIMULATION IN THE CPU ***\n");
#endif
  fprintf(file_ptr, "#\n");
  
  
  // Report only one dose plane in ASCII, all the other data in binary only:

  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  3D dose deposition map (and dose uncertainty) created tallying the energy deposited by photons inside each voxel of the input geometry.\n");
  fprintf(file_ptr, "#  Electrons were not transported and therefore we are approximating that the dose is equal to the KERMA (energy released by the photons alone).\n");
  fprintf(file_ptr, "#  This approximation is acceptable when there is electronic equilibrium and when the range of the secondary electrons is shorter than the voxel size.\n");
  fprintf(file_ptr, "#  Usually the doses will be acceptable for photon energies below 1 MeV. The dose estimates may not be accurate at the interface of low density volumes.\n");
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  The 3D dose deposition is reported in binary form in the .raw files (data given as 32-bit floats). \n");
  
  fprintf(file_ptr, "#  The %% relative error in the voxel dose at 2 standard deviations [=100*2*sigma/voxel_dose] is reported in the *_PercentRelError2sigma.raw file (32-bit floats). \n");   //  !!SPIE2013!!   Report relative error 
   
  fprintf(file_ptr, "#  To reduce the memory use and the reporting time this text output reports only the 2D dose at the Z plane at the level\n"); 
  fprintf(file_ptr, "#  of the source focal spot: z_coord = %d (z_coord in ROI = %d)\n", z_plane_dose, z_plane_dose_ROI);
  fprintf(file_ptr, "#\n");  
  fprintf(file_ptr, "#  The total dose deposited in each different material is reported to the standard output.\n");
  fprintf(file_ptr, "#  The dose is calculated adding the energy deposited in the individual voxels within the dose ROI and dividing by the total mass of the material in the ROI.\n");
  fprintf(file_ptr, "#\n");  
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  Voxel size:  %lf x %lf x %lf = %lf cm^3\n", 1.0/(double)(voxel_data->inv_voxel_size.x), 1.0/(double)(voxel_data->inv_voxel_size.y), 1.0/(double)(voxel_data->inv_voxel_size.z), 1.0/(double)(voxel_data->inv_voxel_size.x*voxel_data->inv_voxel_size.y*voxel_data->inv_voxel_size.z));
  fprintf(file_ptr, "#  Number of voxels in the reported region of interest (ROI) X, Y and Z:\n");
  fprintf(file_ptr, "#      %d  %d  %d\n", DX, DY, DZ);
  fprintf(file_ptr, "#  Coordinates of the ROI inside the voxel volume = X[%d,%d], Y[%d,%d], Z[%d,%d]\n", dose_ROI_x_min+1, dose_ROI_x_max+1, dose_ROI_y_min+1, dose_ROI_y_max+1, dose_ROI_z_min+1, dose_ROI_z_max+1);  // Show ROI with index=1 for the first voxel instead of 0.
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  Voxel dose units: eV/g per history\n");
  fprintf(file_ptr, "#  X rows given first, then Y, then Z. One blank line separates the different Y, and two blanks the Z values (GNUPLOT format).\n");
  fprintf(file_ptr, "#  The dose distribution is also reported with binary FLOAT values (.raw file) for easy visualization in ImageJ.\n");
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#    [DOSE]   [2*standard_deviation]\n");
  fprintf(file_ptr, "# =====================================\n");
  fflush(file_ptr);
  
  double voxel_dose, max_voxel_dose[MAX_MATERIALS], max_voxel_dose_std_dev[MAX_MATERIALS], max_voxel_dose_all_mat=0.0, max_voxel_dose_std_dev_all_mat=0.0;
  int max_voxel_dose_x[MAX_MATERIALS], max_voxel_dose_y[MAX_MATERIALS], max_voxel_dose_z[MAX_MATERIALS];
  unsigned long long int total_energy_deposited = 0;
  double inv_SCALE_eV = 1.0 / SCALE_eV,      // conversion to eV using the inverse of the constant used in the tally function (defined in the header file).         
                inv_N = 1.0 / (double)(total_histories*((unsigned long long int)num_projections));
                                
  register int i, j, k, voxel=0;
    
  double mat_Edep[MAX_MATERIALS], mat_Edep2[MAX_MATERIALS], mat_mass_ROI[MAX_MATERIALS];    // Arrays with the total energy, energy squared and mass of each material inside the ROI (mass and dose outside the ROI was not tallied).
  unsigned int mat_voxels[MAX_MATERIALS];
  for(i=0; i<MAX_MATERIALS; i++)
  {
     mat_Edep[i]  = 0.0;
     mat_Edep2[i] = 0.0;
     mat_mass_ROI[i]  = 0.0;
     mat_voxels[i]= 0;
     max_voxel_dose[i]        =-1.0;
     max_voxel_dose_std_dev[i]= 1.0e-15;
     max_voxel_dose_x[i]      = 0;
     max_voxel_dose_y[i]      = 0;
     max_voxel_dose_z[i]      = 0;       
  }
  
  double voxel_volume = 1.0 / ( ((double)voxel_data->inv_voxel_size.x) * ((double)voxel_data->inv_voxel_size.y) * ((double)voxel_data->inv_voxel_size.z) );
    
  for(k=0; k<DZ; k++)
  {
    for(j=0; j<DY; j++)
    {
      for(i=0; i<DX; i++)
      {
        register int voxel_geometry = (i+dose_ROI_x_min) + (j+dose_ROI_y_min)*voxel_data->num_voxels.x + (k+dose_ROI_z_min)*voxel_data->num_voxels.x*voxel_data->num_voxels.y;
        register double inv_voxel_mass = 1.0 / (voxel_mat_dens[voxel_geometry].y*voxel_volume);

        register int mat_number = (int)(voxel_mat_dens[voxel_geometry].x) - 1 ;  // Material number, starting at 0.
        mat_mass_ROI[mat_number]  += voxel_mat_dens[voxel_geometry].y*voxel_volume;   // Estimate mass and energy deposited in this material
        mat_Edep[mat_number]  += (double)voxels_Edep[voxel].x;        // Using doubles to avoid overflow
        mat_Edep2[mat_number] += (double)voxels_Edep[voxel].y;
        mat_voxels[mat_number]++;                                                // Count voxels made of this material
        
                
              // Optional code to eliminate dose deposited in air (first material).  Sometimes useful for visualization (dose to air irrelevant, noisy)
              //   if (voxel_mat_dens[voxel_geometry].x < 1.1f)
              //   {
              //     voxels_Edep[voxel].x = 0.0f;
              //     voxels_Edep[voxel].y = 0.0f;
              //   }
      // Do not report dose in materials with low density (air)       // !!MCGPU-PET!!
      if (voxel_mat_dens[voxel_geometry].y < 0.1f)
      {
        voxels_Edep[voxel].x = 0.0f;
        voxels_Edep[voxel].y = 0.0f;
      }
                
        // -- Convert total energy deposited to dose [eV/gram] per history:                        
        
//  !!DeBuG!! BUG in first version MC-GPU v1.3, corrected for v1.4 [2013-01-31]. Edep2 is NOT scaled by SCALE_eV!! Also, division by voxel_mass must be done at the end!
//  !!DeBuG!!   Wrong:  voxel_dose = ((double)voxels_Edep[voxel].x) * inv_N * inv_voxel_mass * inv_SCALE_eV;
//  !!DeBuG!!   Wrong:  register double voxel_std_dev = (((double)voxels_Edep[voxel].y) * inv_N * inv_SCALE_eV * inv_voxel_mass - voxel_dose*voxel_dose) * inv_N;

        voxel_dose = ((double)voxels_Edep[voxel].x) * inv_N * inv_SCALE_eV;    // [<Edep> == Edep / N_hist /scaling_factor ;  dose == <Edep> / mass]
        total_energy_deposited += voxels_Edep[voxel].x;
               
        register double voxel_std_dev = (((double)voxels_Edep[voxel].y) * inv_N - voxel_dose*voxel_dose) * inv_N * inv_voxel_mass;   // [sigma_Edep^2 = (<Edep^2> - <Edep>^2) / N_hist] ; [sigma_dose^2 = sigma_Edep/mass] (not using SCALE_eV for std_dev to prevent overflow)  

        if (voxel_std_dev>0.0)
          voxel_std_dev = sqrt(voxel_std_dev);
        
        voxel_dose *= inv_voxel_mass;    // [dose == <Edep> / mass]
        
        if (voxel_dose > max_voxel_dose[mat_number])    // Tally peak dose for each material!
        {
          // Find the voxel that has the maximum dose:
          max_voxel_dose[mat_number]          = voxel_dose;
          max_voxel_dose_std_dev[mat_number]  = voxel_std_dev;
          max_voxel_dose_x[mat_number]        = i+dose_ROI_x_min;
          max_voxel_dose_y[mat_number]        = j+dose_ROI_y_min;
          max_voxel_dose_z[mat_number]        = k+dose_ROI_z_min;
          if (voxel_dose > max_voxel_dose_all_mat)
          {
            max_voxel_dose_all_mat = voxel_dose;
            max_voxel_dose_std_dev_all_mat = voxel_std_dev;
          }
        }
        
        // Report only one dose plane in ASCII:
        if (k == z_plane_dose_ROI) 
          fprintf(file_ptr, "%.6lf %.6lf\n", voxel_dose, 2.0*voxel_std_dev);        
        
        float voxel_dose_float  = (float)voxel_dose;         // After dividing by the number of histories I can report FLOAT bc the number of significant digits will be low.  
        
        fwrite(&voxel_dose_float,  sizeof(float), 1, file_binary_mean_ptr);    // Write dose data in a binary file that can be easyly open in imageJ.   !!BINARY!!

       
        // !!DeBuG!! OLD version, reporting sigma: float voxel_sigma_float = 2.0f * (float)(voxel_std_dev);  fwrite(&voxel_sigma_float, sizeof(float), 1, file_binary_sigma_ptr);
        float voxel_relErr_float = 0.0f;
        if (voxel_dose > 0.0)
          voxel_relErr_float = 200.0f*(float)(voxel_std_dev/voxel_dose);        //  New in MC-GPU v1.4: Report relative error for 2*sigma, in %  (avoid dividing by 0)
        fwrite(&voxel_relErr_float, sizeof(float), 1, file_binary_sigma_ptr);
        
        
        voxel++;
      }
      if (k == z_plane_dose_ROI) 
        fprintf(file_ptr, "\n");     // Separate Ys with an empty line for visualization with gnuplot.
    }
    if (k == z_plane_dose_ROI) 
      fprintf(file_ptr, "\n");     // Separate Zs.
  }

  
  fprintf(file_ptr, "#   ****** DOSE REPORT: TOTAL SIMULATION PERFORMANCE FOR ALL PROJECTIONS ******\n");
  fprintf(file_ptr, "#       Total number of simulated x rays: %lld\n", total_histories*((unsigned long long int)num_projections));
  fprintf(file_ptr, "#       Simulated x rays per projection:  %lld\n", total_histories);
  fprintf(file_ptr, "#       Total simulation time [s]:  %.2f\n", time_total_MC_init_report);
  if (time_total_MC_init_report>0.000001)
    fprintf(file_ptr, "#       Total speed [x-rays/s]:  %.2f\n", (double)(total_histories*((unsigned long long int)num_projections))/time_total_MC_init_report);

  
  fprintf(file_ptr, "\n#       Total energy absorved inside the dose ROI: %.5lf keV/hist\n\n", 0.001*((double)total_energy_deposited)*inv_N*inv_SCALE_eV);
  
  // Output data to standard input:
  printf("\n              Total energy absorved inside the dose deposition ROI: %.5lf keV/hist\n", 0.001*((double)total_energy_deposited)*inv_N*inv_SCALE_eV);
  printf(  "              Maximum voxel dose (+-2 sigma): %lf +- %lf eV/g per history.\n", max_voxel_dose_all_mat, max_voxel_dose_std_dev_all_mat);  
  
      // OLD:   register double voxel_mass_max_dose = voxel_volume*voxel_mat_dens[max_dose_voxel_geometry].y; 
      // OLD:   printf(  "              Maximum voxel dose (+-2 sigma): %lf +- %lf eV/g per history (E_dep_voxel=%lf eV/hist)\n", max_voxel_dose, max_voxel_dose_std_dev, (max_voxel_dose*voxel_mass_max_dose));
      // OLD:   printf(  "              for the voxel: material=%d, density=%.8f g/cm^3, voxel_mass=%.8lf g, voxel coord in geometry=(%d,%d,%d)\n\n", (int)voxel_mat_dens[max_dose_voxel_geometry].x, voxel_mat_dens[max_dose_voxel_geometry].y, voxel_mass_max_dose, max_voxel_dose_x, max_voxel_dose_y, max_voxel_dose_z);
  
  
  // -- Report dose deposited in each material:  
  printf("              Dose deposited in the different materials inside the input ROI computed post-processing the 3D voxel dose results:\n\n");
  
//  OLD reporting without peak dose (v1.3):    printf("    [MATERIAL]  [DOSE_ROI, eV/g/hist]  [2*std_dev]  [Rel error 2*std_dev, %%]  [E_dep [eV/hist]  [MASS_ROI, g]  [NUM_VOXELS_ROI]\n");
  printf("  [MAT]  [DOSE_ROI eV/g/hist]  [2*std_dev]  [Rel error %%]  [Peak voxel dose]  [2*std_dev]  [Rel error %%]  [Peak voxel coord]  [E_dep eV/hist]  [MASS_ROI g]  [NUM_VOXELS_ROI]\n");
  printf(" ===============================================================================================================================================================================\n");  
  
  for(i=0; i<MAX_MATERIALS; i++)
  {
    if(mat_voxels[i]>0)   // Report only for materials found at least in 1 voxel of the input geometry (prevent dividing by 0 mass).
    {
      
      double Edep = mat_Edep[i] * inv_N * inv_SCALE_eV;    // [dose == Edep/Mass/N_hist]
      // !!DeBuG!! BUG in version 1.2: I have to divide by mass after computing the mean and sigma!!!
      // !!DeBuG!! WRONG code:  double material_dose = mat_Edep[i] * inv_N  * inv_SCALE_eV / mat_mass_ROI[i];    // [dose == Edep/Mass/N_hist]
      // !!DeBuG!! WRONG code:  double material_std_dev = (mat_Edep2[i] * inv_N  * inv_SCALE_eV / mat_mass_ROI[i] - material_dose*material_dose) * inv_N;   // [sigma^2 = (<Edep^2> - <Edep>^2) / N_hist]      
      
      double material_std_dev = (mat_Edep2[i] * inv_N - Edep*Edep) * inv_N;   // [sigma^2 = (<Edep^2> - <Edep>^2) / N_hist]   (mat_Edep2 not scaled by SCALE_eV in kernel to prevent overflow)
      if (material_std_dev>0.0)
        material_std_dev = sqrt(material_std_dev);
      
      double material_dose = Edep / mat_mass_ROI[i];
      material_std_dev = material_std_dev / mat_mass_ROI[i];
      
      double rel_diff=0.0, rel_diff_peak=0.0;
      if (material_dose>0.0)
      {
        rel_diff = material_std_dev/material_dose;
        rel_diff_peak = max_voxel_dose_std_dev[i]/max_voxel_dose[i];
      }
    
      printf("\t%d\t%.5lf\t%.5lf\t%.3lf\t\t%.5lf\t%.5lf\t%.3lf\t(%d,%d,%d)\t\t%.5lf\t%.5lf\t%u\n", (i+1), material_dose, 2.0*material_std_dev, (200.0*rel_diff), max_voxel_dose[i], 2.0*max_voxel_dose_std_dev[i], (200.0*rel_diff_peak), max_voxel_dose_x[i], max_voxel_dose_y[i], max_voxel_dose_z[i], Edep, mat_mass_ROI[i], mat_voxels[i]);
      //  OLD reporting without peak dose (v1.3):   printf("\t%d\t%.5lf\t\t%.5lf\t\t%.2lf\t\t%.2lf\t\t%.5lf\t%u\n", (i+1), material_dose, 2.0*material_std_dev, (2.0*100.0*rel_diff), Edep, mat_mass_ROI[i], mat_voxels[i]);            
 
    }    
  }       
  printf("\n");
        
  
  fflush(stdout);          
  fclose(file_ptr);  // Close output file and flush stream
  fclose(file_binary_mean_ptr);
  fclose(file_binary_sigma_ptr);

  return 0;   // Report could return not 0 to continue the simulation...
}
///////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////
//! Report the tallied dose to each material number, accounting for different 
//! densities in different regions with the same material number. 
//!
//!       @param[in] num_projections   Number of projections simulated
//!       @param[in] total_histories   Total number of x-rays simulated per projection
//!       @param[out] density_nominal   Array with the nominal densities of materials given in the input file; -1 for materials not defined. Used to report only defined materials.
//!       @param[in] materials_dose   Tallied dose and dose^2 arrays
////////////////////////////////////////////////////////////////////////////////
int report_materials_dose(int num_projections, unsigned long long int total_histories, float *density_nominal, ulonglong2 *materials_dose, double *mass_materials, char file_name_materials[MAX_MATERIALS][250])  // !!tally_materials_dose!!
{

  printf("\n\n          *** MATERIALS TOTAL DOSE TALLY REPORT ***\n\n");  
  printf("              Dose deposited in each material defined in the input file (tallied directly per material, not per voxel):\n");
  printf("              The results of this tally should be equal to the voxel tally doses for an ROI covering all voxels.\n");
  
  printf("              Total number of simulated x rays: %lld\n", total_histories*((unsigned long long int)num_projections));       // !!DBT!!
  
  if (num_projections>1)
    printf("              Simulated x rays for each of %d projections:  %lld\n\n", num_projections, total_histories);
  
  printf("\t [MAT]  [DOSE eV/g/hist]  [2*std_dev]  [Rel_error 2*std_dev, %%]  [E_dep eV/hist]  [DOSE mGy]  [Material mass g]  [Material file name]\n");
  printf("\t======================================================================================================================================\n");
  
  double dose, Edep, std_dev, rel_diff, inv_N = 1.0 / (double)(total_histories*((unsigned long long int)num_projections));
  int i, flag=0, max_mat=0;
  for(i=0; i<MAX_MATERIALS; i++)
  {
    if (density_nominal[i]<0.0f)
      break;  // Skip report for materials not defined in the input file
      
    // Report the material file names removing the absolute file system path for clarity:
    char file_name_material_without_path[250];
    char* last_slash = strrchr(file_name_materials[i],'/');    // Return a pointer to the last character '/' in the input name, or NULL if not found
    if (last_slash==NULL)
      strcpy(file_name_material_without_path, file_name_materials[i]);
    else
      strcpy(file_name_material_without_path, (last_slash+1));
    
    
    Edep    = ((double)materials_dose[i].x) / SCALE_eV * inv_N;    
    std_dev = sqrt( (((double)materials_dose[i].y)*inv_N - Edep*Edep) * inv_N );   // [sigma^2 = (<Edep^2> - <Edep>^2) / N_hist]   (not scaling "materials_dose[i].y" by SCALE_eV in kernel to prevent overflow).
    
    if (Edep>0.0)
      rel_diff = std_dev/Edep;
    else
      rel_diff = 0.0;

    dose    = Edep / max_value(mass_materials[i], 0.00001);     // Prevent division by 0
    std_dev = std_dev / max_value(mass_materials[i], 0.00001);
    
 
    printf("\t%d\t%.5lf\t\t%.5lf\t\t%.2lf\t\t%.5lf\t\t%.5lf\t\t%.5lf\t\t%s\n", (i+1), dose, 2.0*std_dev, 2.0*100.0*rel_diff, Edep, ((double)materials_dose[i].x)/SCALE_eV/max_value(mass_materials[i], 0.00001)*(1.0e3/6.2415e15), mass_materials[i], file_name_material_without_path);
    
    if (materials_dose[i].x>1e16 || dose!=abs(dose) || std_dev!=abs(std_dev))  // !!DeBuG!!  Try to detect a possible overflow in any material: large counter or negative, nan value
    {
      flag = 1;
      if (materials_dose[i].x>materials_dose[max_mat].x) 
        max_mat = i;
    }
  }
  
  if (flag!=0)    // !!DeBuG!! Try to detect a possible overflow: large counter or negative, nan value. The value of SCALE_eV can be reduced to prevent this overflow in some cases.
  {
    printf("\n     WARNING: it is possible that the unsigned long long int counter used to tally the standard deviation overflowed (>2^64).\n");  // !!DeBuG!! 
    printf("              The standard deviation may be incorrectly measured, but it will surely be very small (<< 1%%).\n");
    printf("              Max counter (mat=%d): E_dep = %llu , E_dep^2 = %llu\n\n", max_mat+1, materials_dose[max_mat].x, materials_dose[max_mat].y);
  }
  fflush(stdout);  
  return 0;
}


///////////////////////////////////////////////////////////////////////////////


/*

///////////////////////////////////////////////////////////////////////////////
//!  Sets the CT trajectory: store in memory the source and detector rotations
//!  that are needed to calculate the multiple projections.
//!  The first projection (0) was previously initialized in function "read_input".
//!  
//!
//!  ASSUMPTIONS: the CT scan plane must be perpendicular to the Z axis, ie,
//!               the initial direction of the particles must have w=0!
//!
///////////////////////////////////////////////////////////////////////////////
void set_CT_trajectory(int myID, int num_projections, double D_angle, double angularROI_0, double angularROI_1, double SRotAxisD, struct source_struct* source_data, struct detector_struct* detector_data, double vertical_translation_per_projection)
{
  MASTER_THREAD printf("\n    -- Setting the sources and detectors for the %d CT projections (MAX_NUM_PROJECTIONS=%d):\n", num_projections, MAX_NUM_PROJECTIONS);
  double cos_rX, cos_rZ, sin_rX, sin_rZ, current_angle;

  // --Set center of rotation at the input distance between source and detector:
  float3 center_rotation;
  center_rotation.x =  source_data[0].position.x + source_data[0].direction.x * SRotAxisD;
  center_rotation.y =  source_data[0].position.y + source_data[0].direction.y * SRotAxisD;
  center_rotation.z =  source_data[0].position.z;      //  + source_data[0].direction.z * SRotAxisD;   // w=0 all the time!!

  // --Angular span between projections:

  //  -Set initial angle for the source (180 degress less than the detector pointed by the direction vector; the zero angle is the X axis, increasing to +Y axis).
  current_angle = acos((double)source_data[0].direction.x);
  if (source_data[0].direction.y<0)
    current_angle = -current_angle;     // Correct for the fact that positive and negative angles have the same ACOS
  if (current_angle<0.0)
    current_angle += 2.0*PI;   // Make sure the angle is not negative, between [0,360) degrees.
  current_angle = current_angle - PI;   // Correct the fact that the source is opposite to the detector (180 degrees difference).
  if (current_angle<0.0)
    current_angle += 2.0*PI;   // Make sure the angle is not negative, between [0,360) degrees..

  MASTER_THREAD printf("         << Projection #1 >> initial_angle=%.8f , D_angle=%.8f\n", current_angle*RAD2DEG, D_angle*RAD2DEG);
  MASTER_THREAD printf("                             Source direction=(%.8f,%.8f,%.8f), position=(%.8f,%.8f,%.8f)\n", source_data[0].direction.x,source_data[0].direction.y,source_data[0].direction.z, source_data[0].position.x,source_data[0].position.y,source_data[0].position.z);

  int i;
  for (i=1; i<num_projections; i++)   // The first projection (i=0) was initialized in function "read_input".
  {
    // --Init constant parameters to the values in projection 0:
    source_data[i].cos_theta_low = source_data[0].cos_theta_low;
    source_data[i].phi_low = source_data[0].phi_low;
    source_data[i].D_cos_theta = source_data[0].D_cos_theta;
    source_data[i].D_phi = source_data[0].D_phi;
    source_data[i].max_height_at_y1cm = source_data[0].max_height_at_y1cm;    
    detector_data[i].sdd = detector_data[0].sdd;
    detector_data[i].width_X = detector_data[0].width_X;
    detector_data[i].height_Z = detector_data[0].height_Z;
    detector_data[i].inv_pixel_size_X = detector_data[0].inv_pixel_size_X;
    detector_data[i].inv_pixel_size_Z = detector_data[0].inv_pixel_size_Z;
    detector_data[i].num_pixels = detector_data[0].num_pixels;
    detector_data[i].total_num_pixels = detector_data[0].total_num_pixels;
    detector_data[i].rotation_flag = detector_data[0].rotation_flag;
        
        
    // --Set the new source location and direction, for the current CT projection:
    current_angle += D_angle;
    if (current_angle>=(2.0*PI-0.0001))
      current_angle -= 2.0*PI;   // Make sure the angle is not above or equal to 360 degrees.

    source_data[i].position.x = center_rotation.x + SRotAxisD*cos(current_angle);
    source_data[i].position.y = center_rotation.y + SRotAxisD*sin(current_angle);
    source_data[i].position.z = source_data[i-1].position.z + vertical_translation_per_projection;   //  The Z position can increase between projections for a helical scan. But rotation still around Z always: (w=0)!!

    source_data[i].direction.x = center_rotation.x - source_data[i].position.x;
    source_data[i].direction.y = center_rotation.y - source_data[i].position.y;
    source_data[i].direction.z = 0.0f;    //  center_rotation.z - source_data[0].position.z;   !! w=0 all the time!!  

    double norm = 1.0/sqrt((double)source_data[i].direction.x*(double)source_data[i].direction.x + (double)source_data[i].direction.y*(double)source_data[i].direction.y);    // + source_data[i].direction.z*source_data[i].direction.z
    source_data[i].direction.x = (float)(((double)source_data[i].direction.x)*norm);
    source_data[i].direction.y = (float)(((double)source_data[i].direction.y)*norm);
      // source_data[i].direction.z = (float)(((double)source_data[i].direction.z)*norm);

    // --Set the new detector in front of the new source:
    detector_data[i].center.x = source_data[i].position.x + source_data[i].direction.x * detector_data[i].sdd;   // Set the center of the detector straight ahead of the focal spot.
    detector_data[i].center.y = source_data[i].position.y + source_data[i].direction.y * detector_data[i].sdd;
    detector_data[i].center.z = source_data[i].position.z;    //  + source_data[i].direction.z * detector_data[i].sdd;   !! w=0 all the time!!

    double rotX, rotZ;

      //  detector_data[0].rotation_flag = 1;   //  Already set in read_input!

    // -- Rotate the detector center to +Y:
    //    Set the rotation that will bring particles from the detector plane to +Y=(0,+1,0) through a rotation around X and around Z (counter-clock):
    rotX = 0.0;   // !! w=0 all the time!!  CORRECT CALCULATION:  acos(source_data[0].direction.z) - 0.5*PI;  // Rotate to +Y = (0,+1,0) --> rotX_0 =  -PI/2

    if ( (source_data[i].direction.x*source_data[i].direction.x + source_data[i].direction.y*source_data[i].direction.y) > 1.0e-8 )   // == u^2+v^2 > 0
      if (source_data[i].direction.y >= 0.0f)
        rotZ = 0.5*PI - acos(source_data[i].direction.x/sqrt(source_data[i].direction.x*source_data[i].direction.x + source_data[i].direction.y*source_data[i].direction.y));
      else
        rotZ = 0.5*PI - (-acos(source_data[i].direction.x/sqrt(source_data[i].direction.x*source_data[i].direction.x + source_data[i].direction.y*source_data[i].direction.y)));
    else
      rotZ = 0.0;   // Vector pointing to +Z, do not rotate around Z then.

    MASTER_THREAD printf("         << Projection #%d >> current_angle=%.8f degrees (rotation around Z axis = %.8f)\n", (i+1), current_angle*RAD2DEG, rotZ*RAD2DEG);
    MASTER_THREAD printf("                             Source direction = (%.8f,%.8f,%.8f) , position = (%.8f,%.8f,%.8f)\n", source_data[i].direction.x,source_data[i].direction.y,source_data[i].direction.z, source_data[i].position.x,source_data[i].position.y,source_data[i].position.z);

    cos_rX = cos(rotX);
    cos_rZ = cos(rotZ);
    sin_rX = sin(rotX);
    sin_rZ = sin(rotZ);
    detector_data[i].rot_inv[0] =  cos_rZ;    // Rotation matrix RxRz:
    detector_data[i].rot_inv[1] = -sin_rZ;
    detector_data[i].rot_inv[2] =  0.0f;
    detector_data[i].rot_inv[3] =  cos_rX*sin_rZ;
    detector_data[i].rot_inv[4] =  cos_rX*cos_rZ;
    detector_data[i].rot_inv[5] = -sin_rX;
    detector_data[i].rot_inv[6] =  sin_rX*sin_rZ;
    detector_data[i].rot_inv[7] =  sin_rX*cos_rZ;
    detector_data[i].rot_inv[8] =  cos_rX;


    detector_data[i].corner_min_rotated_to_Y.x = detector_data[i].center.x*detector_data[i].rot_inv[0] + detector_data[i].center.y*detector_data[i].rot_inv[1] + detector_data[i].center.z*detector_data[i].rot_inv[2];
    detector_data[i].corner_min_rotated_to_Y.y = detector_data[i].center.x*detector_data[i].rot_inv[3] + detector_data[i].center.y*detector_data[i].rot_inv[4] + detector_data[i].center.z*detector_data[i].rot_inv[5];
    detector_data[i].corner_min_rotated_to_Y.z = detector_data[i].center.x*detector_data[i].rot_inv[6] + detector_data[i].center.y*detector_data[i].rot_inv[7] + detector_data[i].center.z*detector_data[i].rot_inv[8];

    // -- Set the lower corner (minimum) coordinates at the normalized orientation: +Y. The detector has thickness 0.
    detector_data[i].corner_min_rotated_to_Y.x = detector_data[i].corner_min_rotated_to_Y.x - 0.5*detector_data[i].width_X;
//  detector_data[i].corner_min_rotated_to_Y.y = detector_data[i].corner_min_rotated_to_Y.y;
    detector_data[i].corner_min_rotated_to_Y.z = detector_data[i].corner_min_rotated_to_Y.z - 0.5*detector_data[i].height_Z;

    // *** Init the fan beam source model:

      rotZ = -rotZ;   // The source rotation is the inverse of the detector.
      cos_rX = cos(rotX);
      cos_rZ = cos(rotZ);
      sin_rX = sin(rotX);
      sin_rZ = sin(rotZ);
      // --Rotation around X (alpha) and then around Z (phi): Rz*Rx (oposite of detector rotation)
      source_data[i].rot_fan[0] =  cos_rZ;
      source_data[i].rot_fan[1] = -cos_rX*sin_rZ;
      source_data[i].rot_fan[2] =  sin_rX*sin_rZ;
      source_data[i].rot_fan[3] =  sin_rZ;
      source_data[i].rot_fan[4] =  cos_rX*cos_rZ;
      source_data[i].rot_fan[5] = -sin_rX*cos_rZ;
      source_data[i].rot_fan[6] =  0.0f;
      source_data[i].rot_fan[7] =  sin_rX;
      source_data[i].rot_fan[8] =  cos_rX;

        // printf("\n    -- Source location and direction for the following CT projection:\n");   // !!Verbose!! 
        // printf("                 angle between projections = %lf degrees\n", D_angle*RAD2DEG);
        // printf("                             current angle = %lf degrees\n", current_angle*RAD2DEG);
        // printf("                   new focal spot position = (%f, %f, %f)\n", source_data[i].position.x, source_data[i].position.y, source_data[i].position.z);
        // printf("                      new source direction = (%f, %f, %f)\n", source_data[i].direction.x, source_data[i].direction.y, source_data[i].direction.z);
        // printf("                       new detector center = (%f, %f, %f)\n", detector_center.x, detector_center.y, detector_center.z);
        // printf("           new detector low corner (at +Y) = (%f, %f, %f)\n", detector_data->corner_min_rotated_to_Y[i].x, detector_data->corner_min_rotated_to_Y[i].y, detector_data->corner_min_rotated_to_Y[i].z);
        // printf("                        center of rotation = (%f, %f, %f)\n", center_rotation.x, center_rotation.y, center_rotation.z);
        // printf("         detector width (X) and height (Z) = %f , %f cm\n", detector_data->width_X, detector_data->height_Z);
        // printf("            rotations to +Y around Z and X = %f , %f degrees\n", rotZ*RAD2DEG, rotX*RAD2DEG);
  }
}

*/

///////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//! Initialize the first seed of the pseudo-random number generator (PRNG) 
//! RANECU to a position far away from the previous history (leap frog technique).
//! This function is equivalent to "init_PRNG" but only updates one of the seeds.
//!
//! Note that if we use the same seed number to initialize the 2 MLCGs of the PRNG
//! we can only warranty that the first MLCG will be uncorrelated for each value
//! generated by "update_seed_PRNG". There is a tiny chance that the final PRNs will
//! be correlated because the leap frog on the first MLCG will probably go over the
//! repetition cycle of the MLCG, which is much smaller than the full RANECU. But any
//! correlataion is extremely unlikely. Function "init_PRNG" doesn't have this issue.
//!
//!       @param[in] batch_number   Elements to skip (eg, MPI thread_number).
//!       @param[in] total_histories   Histories to skip.
//!       @param[in,out] seed   Initial PRNG seeds; returns the updated seed.
////////////////////////////////////////////////////////////////////////////////
inline void update_seed_PRNG(int batch_number, unsigned long long int total_histories, int* seed)
{
  if (0==batch_number)
    return;
    
  unsigned long long int leap = total_histories * (batch_number * LEAP_DISTANCE);
  int y = 1;
  int z = a1_RANECU;
  // -- Calculate the modulo power '(a^leap)MOD(m)' using a divide-and-conquer algorithm adapted to modulo arithmetic
  for(;;)
  {
    // (A2) Halve n, and store the integer part and the residue
    if (0!=(leap&01))  // (bit-wise operation for MOD(leap,2), or leap%2 ==> proceed if leap is an odd number)  Equivalent: t=(short)(leap%2);
    {
      leap >>= 1;     // Halve n moving the bits 1 position right. Equivalent to:  leap=(leap/2);  
      y = abMODm(m1_RANECU,z,y);      // (A3) Multiply y by z:  y = [z*y] MOD m
      if (0==leap) break;         // (A4) leap==0? ==> finish
    }
    else           // (leap is even)
    {
      leap>>= 1;     // Halve leap moving the bits 1 position right. Equivalent to:  leap=(leap/2);
    }
    z = abMODm(m1_RANECU,z,z);        // (A5) Square z:  z = [z*z] MOD m
  }
  // AjMODm1 = y;                 // Exponentiation finished:  AjMODm = expMOD = y = a^j
  // -- Compute and display the seeds S(i+j), from the present seed S(i), using the previously calculated value of (a^j)MOD(m):
  //         S(i+j) = [(a**j MOD m)*S(i)] MOD m
  //         S_i = abMODm(m,S_i,AjMODm)
  *seed = abMODm(m1_RANECU, *seed, y);
}


///////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//! Read the energy spectrum file and initialize the Walker aliasing sampling.
//!
//!       @param[in] file_name_espc   File containing the energy spectrum (lower energy value in each bin and its emission probability).
//!       @param[in,out] source_energy_data   Energy spectrum and other source data. The Walker alias and cutoffs are initialized in this function.
//!       @param[out] mean_energy_spectrum   Mean energy in the input x-ray energy spectrum.
////////////////////////////////////////////////////////////////////////////////
void init_energy_spectrum(char* file_name_espc, struct source_energy_struct* source_energy_data, float *mean_energy_spectrum)
{
  char *new_line_ptr = NULL, new_line[250];    
  float lower_energy_bin, prob;
  float prob_espc_bin[MAX_ENERGY_BINS];    // The input probabilities of each energy bin will be discarded after Walker is initialized

  // -- Read spectrum from file:
  FILE* file_ptr = fopen(file_name_espc, "r");
  if (NULL==file_ptr)
  {
    printf("\n\n   !!init_energy_spectrum ERROR!! Error trying to read the energy spectrum input file \"%s\".\n\n", file_name_espc);
    exit(-1);
  }
  
  int current_bin = -1;
  do 
  {
    current_bin++;  // Update bin counter
    
    if (current_bin >= MAX_ENERGY_BINS)
    {
      printf("\n !!init_energy_spectrum ERROR!!: too many energy bins in the input spectrum. Increase the value of MAX_ENERGY_BINS=%d.\n", MAX_ENERGY_BINS);
      printf(  "            A negative probability marks the end of the spectrum.\n\n");
      exit(-1);
    }

    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);   // Read the following line of text skipping comments and extra spaces
    
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!init_energy_spectrum ERROR!! The input file for the x ray spectrum (%s) is not readable or incomplete (a negative probability marks the end of the spectrum).\n", file_name_espc);
      exit(-1);
    }
    
    prob = -123456789.0f;  
    
    sscanf(new_line, "%f %f", &lower_energy_bin, &prob);     // Extract the lowest energy in the bin and the corresponding emission probability from the line read 
            
    prob_espc_bin[current_bin]     = prob;
    source_energy_data->espc[current_bin] = lower_energy_bin;           
    
    if (prob == -123456789.0f)
    {
      printf("\n !!init_energy_spectrum ERROR!!: invalid energy bin number %d?\n\n", current_bin);
      exit(-1);
    }
    else if (lower_energy_bin < source_energy_data->espc[max_value(current_bin-1,0)])    // (Avoid a negative index using the macro "max_value" defined in the header file)
    {
      printf("\n !!init_energy_spectrum ERROR!!: input energy bins with decreasing energy? espc(%d)=%f, espc(%d)=%f\n\n", current_bin-1, source_energy_data->espc[max_value(current_bin-1,0)], current_bin, lower_energy_bin);
      exit(-1);
    }
    
  } 
  while (prob > -1.0e-11f);     // A negative probability marks the end of the spectrum


  // Store the number of bins read from the input energy spectrum file:
  source_energy_data->num_bins_espc = current_bin;


  // Init the remaining bins (which will not be used) with the last energy read (will be assumed as the highest energy in the last bin) and 0 probability of emission.
  register int i;
  for (i=current_bin; i<MAX_ENERGY_BINS; i++)
  {
    source_energy_data->espc[i] = lower_energy_bin;
    prob_espc_bin[i]     = 0.0f;
  }


  // Compute the mean energy in the spectrum, taking into account the energy and prob of each bin:
  float all_energy = 0.0f;
  float all_prob = 0.0f;
  for(i=0; i<source_energy_data->num_bins_espc; i++)
  {
    all_energy += 0.5f*(source_energy_data->espc[i]+source_energy_data->espc[i+1])*prob_espc_bin[i];
    all_prob   += prob_espc_bin[i];
  }  
  *mean_energy_spectrum = all_energy/all_prob;
  
          
// -- Init the Walker aliasing sampling method (as it is done in PENELOPE):
  IRND0(prob_espc_bin, source_energy_data->espc_cutoff, source_energy_data->espc_alias, source_energy_data->num_bins_espc);   //!!Walker!! Calling PENELOPE's function to init the Walker method
       
// !!Verbose!! Test sampling
// Sampling the x ray energy using the Walker aliasing algorithm from PENELOPE:
// int sampled_bin = seeki_walker(source_energy_data->espc_cutoff, source_energy_data->espc_alias, 0.5, source_energy_data->num_bins_espc);
// float e = source_energy_data->espc[sampled_bin] + ranecu(seed) * (source_energy_data->espc[sampled_bin+1] - source_energy_data->espc[sampled_bin]);    // Linear interpolation of the final energy within the sampled energy bin
// printf("\n\n !!Walker!! Energy center bin %d = %f keV\n", sampled_bin, 0.001f*e);
  
}       



      
//********************************************************************
//!    Finds the interval (x(i),x(i+1)] containing the input value    
//!    using Walker's aliasing method.                                
//!                                                                   
//!    Input:                                                         
//!      cutoff(1..n) -> interval cutoff values for the Walker method 
//!      cutoff(1..n) -> alias for the upper part of each interval    
//!      randno       -> point to be located                          
//!      n            -> no. of data points                           
//!    Output:                                                        
//!      index i of the semiopen interval where randno lies           
//!    Comments:                                                      
//!      -> The cutoff and alias values have to be previously         
//!         initialised calling the penelope subroutine IRND0.        
//!                                                                   
//!                                                                   
//!    Algorithm implementation based on the PENELOPE code developed   
//!    by Francesc Salvat at the University of Barcelona. For more     
//!    info: www.oecd-nea.org/science/pubs/2009/nea6416-penelope.pdf  
//!                                                                   
//CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//C  PENELOPE/PENGEOM (version 2006)                                     C
//C  Copyright (c) 2001-2006                                             C
//C  Universitat de Barcelona                                            C
//C                                                                      C
//C  Permission to use, copy, modify, distribute and sell this software  C
//C  and its documentation for any purpose is hereby granted without     C
//C  fee, provided that the above copyright notice appears in all        C
//C  copies and that both that copyright notice and this permission      C
//C  notice appear in all supporting documentation. The Universitat de   C
//C  Barcelona makes no representations about the suitability of this    C
//C  software for any purpose. It is provided "as is" without express    C
//C  or implied warranty.                                                C
//CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
inline int seeki_walker(float *cutoff, short int *alias, float randno, int n)
{
   float RN = randno * n;                         // Find initial interval (array starting at 0):   
   int int_part = (int)(RN);                      //   -- Integer part
   float fraction_part = RN - ((float)int_part);  //   -- Fractional part

   if (fraction_part < cutoff[int_part])          // Check if we are in the aliased part
      return int_part;                            // Below the cutoff: return current value
   else
      return (int)alias[int_part];                // Above the cutoff: return alias
}     




//****************************************************************** *
//*                    SUBROUTINE IRND0                              *
//********************************************************************
//*                                                                   
//!  Initialisation of Walker's aliasing algorithm for random         
//!  sampling from discrete probability distributions.                
//!                                                                   
//! Input arguments:                                                  
//!   N ........ number of different values of the random variable.   
//!   W(1:N) ... corresponding point probabilities (not necessarily   
//!              normalised to unity).                                
//! Output arguments:                                                 
//!   F(1:N) ... cutoff values.                                       
//!   K(1:N) ... alias values.                                        
//!                                                                   
//!                                                                   
//!  This subroutine is part of the PENELOPE 2006 code developed      
//!  by Francesc Salvat at the University of Barcelona. For more       
//!  info: www.oecd-nea.org/science/pubs/2009/nea6416-penelope.pdf    
//*                                                                   
//CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//C  PENELOPE/PENGEOM (version 2006)                                     C
//C  Copyright (c) 2001-2006                                             C
//C  Universitat de Barcelona                                            C
//C                                                                      C
//C  Permission to use, copy, modify, distribute and sell this software  C
//C  and its documentation for any purpose is hereby granted without     C
//C  fee, provided that the above copyright notice appears in all        C
//C  copies and that both that copyright notice and this permission      C
//C  notice appear in all supporting documentation. The Universitat de   C
//C  Barcelona makes no representations about the suitability of this    C
//C  software for any purpose. It is provided "as is" without express    C
//C  or implied warranty.                                                C
//CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
void IRND0(float *W, float *F, short int *K, int N)
{
   register int I;
  
   //  ****  Renormalisation.
   double WS=0.0;
   for (I=0; I<N; I++)
   {   
      if(W[I] < 0.0f) 
      {
         printf("\n\n !!ERROR!! IRND0: Walker sampling initialization. Negative point probability? W(%d)=%f\n\n", I, W[I]);
         exit(-1);
      }
      WS = WS + W[I];
   }
   WS = ((double)N) / WS; 
  
   for (I=0; I<N; I++)
   {
      K[I] = I;
      F[I] = W[I] * WS;
   }
    
   if (N==1) 
      return;
     
   //  ****  Cutoff and alias values.
   float HLOW, HIGH;
   int   ILOW, IHIGH, J;
   for (I=0; I<N-1; I++)
   {
      HLOW = 1.0f;
      HIGH = 1.0f;
      ILOW = -1;
      IHIGH= -1;
      for (J=0; J<N; J++)
      {
         if(K[J]==J)
         {
            if(F[J]<HLOW)
            {
               HLOW = F[J];
               ILOW = J;
            }
            else if(F[J]>HIGH)
            {
               HIGH  = F[J];
               IHIGH = J;
            }
         }
      }
      
      if((ILOW==-1) || (IHIGH==-1)) 
        return;

      K[ILOW] = IHIGH;
      F[IHIGH]= HIGH + HLOW - 1.0f;
   }
   return;
}



///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//** Report Phase Space File data to disk:               !!MCGPU-PET!!       //
///////////////////////////////////////////////////////////////////////////////
void report_PSF(char* file_name_output, PSF_element_struct* PSF, int index_PSF, unsigned long long int total_histories, double time_elapsed, struct source_struct* source_data, struct detector_struct* detector_data, char* file_name_voxels)      // !!MCGPU-PET!!
{
  int i;
  FILE* file_ptr = fopen(file_name_output, "w");
  
  if (file_ptr==NULL)
  {
    printf("\n\n   !!fopen ERROR report_PSF!! File %s can not be opened!!\n", file_name_output);
    exit(-3);
  }
     
  // Prepare binary output:               !!October2017!!
  char file_name_output_binary[250];
  strncpy (file_name_output_binary, file_name_output, 250);
  strcat(file_name_output_binary,".raw");
  FILE* file_binary_ptr = fopen(file_name_output_binary, "w");
  
  // Report data to screen:
  printf("\n\n       ****** MCGPU-PET PHASE SPACE FILE INFORMATION ******\n\n");
  printf("           >>> Number of elements in the PSF  = %d\n", index_PSF);
  printf("           >>> First elements of the PSF reported in ASCII at \'%s\' (see its header for more information); complete PSF in binary at \'%s\'\n", file_name_output, file_name_output_binary);   
  printf("           >>> PET acquistion time = %lf s\n", source_data->acquisition_time_ps*1.0e-12);  
  printf("           >>> mean life isotope   = %f\n", source_data->mean_life);
  if (detector_data->tally_TYPE==1)
    printf("           >>> PSF includes only True coincidences (Scatter not reported).\n");
  else if (detector_data->tally_TYPE==2)
    printf("           >>> PSF includes only Scatter coincidences (Trues not reported).\n");
  else
    printf("           >>> PSF includes both True and Scatter coincidences.\n");
  if (detector_data->tally_PSF_SINOGRAM==1)
    printf("           >>> Reported only PSF (Sinogram not reported).\n");
  else if (detector_data->tally_PSF_SINOGRAM==2)
    printf("           >>> Reported only Sinogram (PSF not reported).\n");
  else
    printf("           >>> Reported both PSF and Sinogram.\n\n\n");

  // Report data to text file:
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#     *****************************************************************************\n"); 
  fprintf(file_ptr, "#     ***         MC-GPU-PET, version 0.1 (https://github.com/DIDSR/MCGPU-PET)  ***\n");        // !!MCGPU-PET!!
  fprintf(file_ptr, "#     ***                                                                       ***\n");
  fprintf(file_ptr, "#     ***                     Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)     ***\n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#  *** SIMULATION IN THE GPU USING CUDA ***\n");
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  Phase space file (PSF) of particles arriving at the ideal cylindrical detector surrounding the object.\n");
  fprintf(file_ptr, "# \n");  
  fprintf(file_ptr, "#        Number of detections in the PSF = %d\n", index_PSF);
  fprintf(file_ptr, "#        Allocated PSF elements = %d\n", detector_data->PSF_size);
  fprintf(file_ptr, "#        PET acquistion time    = %lf s\n", source_data->acquisition_time_ps*1.0e-12);  
  fprintf(file_ptr, "#        mean life isotope      = %f\n", source_data->mean_life);
  fprintf(file_ptr, "#        Material activities:\n");
  
  for (i=0; i<MAX_MATERIALS; i++)
    if (source_data->activity[i]>0.0f)
      fprintf(file_ptr, "#                Mat %d: %f Bq\n", i+1, source_data->activity[i]);
   
  fprintf(file_ptr, "#        PSF detector center    = (%.5f,%.5f,%.5f) cm\n", detector_data->PSF_center.x, detector_data->PSF_center.y, detector_data->PSF_center.z);
  fprintf(file_ptr, "#        PSF detector height and radius = %.5f, %.5f cm\n", detector_data->PSF_height, detector_data->PSF_radius);
  fprintf(file_ptr, "#        Input voxel file       = %s\n", file_name_voxels);
  fprintf(file_ptr, "# \n");
  if (detector_data->tally_TYPE==1)
    fprintf(file_ptr, "#        PSF includes only True coincidences (Scatter not reported).\n");
  else if (detector_data->tally_TYPE==2)
    fprintf(file_ptr, "#        PSF includes only Scatter coincidences (Trues not reported).\n");
  else
    fprintf(file_ptr, "#        PSF includes both True and Scatter coincidences.\n");

  if (detector_data->tally_PSF_SINOGRAM==1)
    fprintf(file_ptr, "#        Reported only PSF (Sinogram not reported).\n");
  else if (detector_data->tally_PSF_SINOGRAM==2)
    fprintf(file_ptr, "#        Reported only Sinogram (PSF not reported).\n");
  else
    fprintf(file_ptr, "#        Reported both PSF and Sinogram.\n");

  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#        Reporting only the 5000 first elements below in ASCII. Complete PSF available in binary form in the file %s\n", file_name_output_binary);   //   !!October2017!!
  fprintf(file_ptr, "#        Binary size for each reported variable:\n");
  fprintf(file_ptr, "#               - emission_time (ps): unsigned long long int \n");
  fprintf(file_ptr, "#               - travel_time (ps): float \n");
  fprintf(file_ptr, "#               - emission voxel: int \n");
  fprintf(file_ptr, "#               - energy (eV), z (cm), phi (rad), vx, vy, vz: float \n");
  fprintf(file_ptr, "#               - index1 (Flag for scatter: =0 for non-scattered, =1 for Compton, =2 for Rayleigh, and =3 for multiple scatter), index2: short int  \n");
  fprintf(file_ptr, "# \n");  
  fprintf(file_ptr, "#  [emission_time (ps)]    [travel_time (ps)]    [emission voxel]  [energy (eV)]  [z (cm)]  [phi (rad)]  [vx]  [vy]  [vz]  [index1]  [index2]\n");
  fprintf(file_ptr, "# ========================================================================================================================================\n");
  
  
  // Report only 100 first elements in ASCII. The complete PSF available in binary in the *.raw output file:       !!October2017!!
  for (i=0; i<min_value(5000,index_PSF); i++)
  {
    fprintf(file_ptr, "%lld %f %d %f %f %f %f %f %f %d %d\n", PSF[i].emission_time_ps, PSF[i].travel_time_ps, PSF[i].emission_absvox, PSF[i].energy, PSF[i].z, PSF[i].phi, PSF[i].vx, PSF[i].vy, PSF[i].vz,  PSF[i].index1, PSF[i].index2);
  }
  fprintf(file_ptr, "\n#        (...)\n");
  
  // Writing the complete PSF in a single instruction for max performance. I must make sure there is not padding between variables in the struct!!  
  fwrite(PSF, sizeof(PSF_element_struct), index_PSF, file_binary_ptr);    // !!October2017!!
  
      // Check for possible padding in struct:  !!DeBuG!! 
      // printf("\n\n>>>>> sizeof(PSF_element_struct)=%d , sizeof(separate elements)=%d \n\n", (int)sizeof(PSF_element_struct), (int)(sizeof(PSF[0].emission_time_ps)+sizeof(PSF[0].travel_time_ps)+sizeof(PSF[0].emission_absvox)+sizeof(PSF[0].energy)+sizeof(PSF[0].z)+sizeof(PSF[0].phi)+sizeof(PSF[0].vx)+sizeof(PSF[0].vy)+sizeof(PSF[0].vz)+sizeof(PSF[0].index1)+sizeof(PSF[0].index2)) );
  
/*    // Old slow but safe output:
  for (i=0; i<index_PSF; i++)
  {
    fwrite(&PSF[i].emission_time_ps, sizeof(unsigned long long int), 1, file_binary_ptr);    // Writing each member separately in case there is padding between variables     !!October2017!!
    fwrite(&PSF[i].travel_time_ps, sizeof(int), 1, file_binary_ptr);
    fwrite(&PSF[i].energy, sizeof(float), 1, file_binary_ptr);
    fwrite(&PSF[i].z, sizeof(float), 1, file_binary_ptr);
    fwrite(&PSF[i].phi, sizeof(float), 1, file_binary_ptr);
    fwrite(&PSF[i].vx, sizeof(float), 1, file_binary_ptr);
    fwrite(&PSF[i].vy, sizeof(float), 1, file_binary_ptr);
    fwrite(&PSF[i].vz, sizeof(float), 1, file_binary_ptr);
    fwrite(&PSF[i].emission_absvox, sizeof(int), 1, file_binary_ptr);
    fwrite(&PSF[i].index1, sizeof(short int), 1, file_binary_ptr);
    fwrite(&PSF[i].index2, sizeof(short int), 1, file_binary_ptr);
  }
*/  


  fprintf(file_ptr, "# \n");  
  fprintf(file_ptr, "#   *** Simulation REPORT: ***\n");
  fprintf(file_ptr, "#       Simulated x rays:    %lld\n", total_histories);
  fprintf(file_ptr, "#       Simulation time [s]: %.2f\n", time_elapsed);
  if (time_elapsed>0.000001)
    fprintf(file_ptr, "#       Speed [x-rays/s]:    %.2f\n\n", ((double)total_histories)/time_elapsed);
  
  fclose(file_ptr);
  fclose(file_binary_ptr);  
}
  
  
  
///////////////////////////////////////////////////////////////////////////////    // !!MCGPU-PET!!  v.0.2
// Prevent simulating blocks for voxels at the top of the phantom (z) where there is no activity:
///////////////////////////////////////////////////////////////////////////////
  int find_last_z_active(struct voxel_struct* voxel_data, struct source_struct* source_data, float3* voxel_mat_dens)      // !!MCGPU-PET!!  v.0.2
  {
    int i, j, k, last_z_active=0;
    long long int vox = voxel_data->num_voxels.x*voxel_data->num_voxels.y*voxel_data->num_voxels.z;    // Maximum voxel number
    
    // Search the material array backwards from the top down until a material with activity>0 is found:
    for (k=voxel_data->num_voxels.z-1; k>=0; k--)
    {
      for (j=voxel_data->num_voxels.y-1; j>=0; j--)
      {
        for (i=voxel_data->num_voxels.x-1; i>=0; i--)
        {
          vox--; 
          if ((source_data->activity[(int)voxel_mat_dens[vox].x-1])>1.0e-7f)
          {
            last_z_active = k;
            break;
          }
        }
        if (last_z_active>0)
          break;
      }
      if (last_z_active>0)
          break;
    }
    return (last_z_active+1);      // !!MCGPU-PET!! 
  }  
  
