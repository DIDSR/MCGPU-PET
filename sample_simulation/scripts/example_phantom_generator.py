#
# ** Script to generate a simple voxelized geometry with the penEasy and MCGPU-PET format **
# 
#    A 9x9x9 voxel geometry, with 1 cm voxels is defined. A 5x5x5 voxel water cube is defined 
#    at the center of the geometry, surrounded by air (2-voxel wide planes). 
#    The water has a background activity of 50 Bq.
#    At the central plane of the water cube 3 1-voxel emissors are defined with activity 1000, 2000, 3000 Bq.
#    An air cavity with no activity is defined at (x,y,z)=(4,5,5).
#
#    The voxelised geometry is defined by 3 column arrays. Each row provides the Material, Density, and
#    Activity of a voxel. Voxels are stored first in teh X, then Y, then Z axis order.
#

print '\n  *** GENERATING A VOXELIZED GEOMETRY FILE COMPATIBLE WITH MCGPU-PET (and penEasy) ***\n'

nvox=[9,9,9]
dvox=[1.0,1.0,1.0]   # cm

mat_air = 1
dens_air = 0.0012
activity_air = 0.0

mat_water = 2
dens_water = 1.0
activity_insert = 1000
activity_water = activity_insert/20


out = open('phantom_9x9x9cm.vox', 'w')

# -- WRITE HEADER
out.write('[SECTION VOXELS HEADER v.2008-04-13]\n')
out.write(str(nvox[0])+' '+str(nvox[1])+' '+str(nvox[2])+'   No. OF VOXELS IN X,Y,Z\n')
out.write(str(dvox[0])+' '+str(dvox[1])+' '+str(dvox[2])+'   VOXEL SIZE (cm) ALONG X,Y,Z\n')
out.write(' 1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n')
out.write(' 2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n')
out.write(' 1                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n')
out.write('[END OF VXH SECTION]  # MCGPU-PET voxel format: Material  Density  Activity\n')

air_edge=2;

for k in range(nvox[2]):
  for j in range(nvox[1]):
    for i in range(nvox[0]):
      if ((i<air_edge)or(j<air_edge)or(k<air_edge)or(i>=nvox[0]-air_edge)or(j>=nvox[1]-air_edge)or(k>=nvox[2]-air_edge)):
        # print i,j,k
        out.write(str(mat_air)+' '+str(dens_air)+' '+str(activity_air)+'\n')  # Air 
      else:  
        # Add activity to 3 1-voxel tracer inserts and 1 air cavity inside the water cube, at the central z plane:
        if ((i==3)and(j==3)and(k==4)):
          out.write(str(mat_water)+' '+str(dens_water)+' '+str(activity_insert*1)+'\n')  # Water with tracer x1
        elif ((i==5)and(j==3)and(k==4)):
          out.write(str(mat_water)+' '+str(dens_water)+' '+str(activity_insert*2)+'\n')  # Water with tracer x2
        elif ((i==5)and(j==5)and(k==4)):
          out.write(str(mat_water)+' '+str(dens_water)+' '+str(activity_insert*3)+'\n')  # Water with tracer x3
        elif ((i==3)and(j==5)and(k==4)):
          out.write(str(mat_air)+' '+str(dens_air)+' '+str(activity_air)+'\n')           # Air cavity
        else:
          out.write(str(mat_water)+' '+str(dens_water)+' '+str(activity_water)+'\n')  # Water
    out.write('\n')
  out.write('\n')

out.write('\n')
out.close()
