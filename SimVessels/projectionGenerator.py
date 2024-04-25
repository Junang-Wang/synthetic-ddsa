"""
Generate vascular projection images with given vascular strings
"""
import math
import os
import sys
import time
from skimage import io, measure
from stl import mesh

import matplotlib.pyplot as plt
import numpy as np
from bolus.bolusInjection import bolus_injection
from vsystem.computeVoxel_fluid import process_network_fluid, resize_network
from vsystem.computeVoxel import process_network


def main():
    # ---------------parameters--------------#
    tissueVolume = (512, 512, 280)

    CTsize = (1024, 1024)  # size of final projections
    num_projection = 16  # number of projections for a volume

    inpath = r"./vessels/update_save/"
    outpath = r"./vessels/volumes/"
    OutPath_CTproj = r"./vessels/bolus_chase/"
    if not os.path.exists(OutPath_CTproj):
        os.makedirs(OutPath_CTproj)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    

    # domean_min = 10
    # domean_max = 60
    # d0std = 60.0
    # for d0mean in range(domean_min, domean_max + 1, int(d0std)):
    d0 = 35 # initial diameter
    d0mean =20
    for d in [12, 15, 20]:  # 10 times Proportion between d0 & d1 subbranch
        for epsilon in range(9, 10):  # differ Proportion between length & diameter
            for niter in range(13, 14):
                start = time.time()
                # load cooridate for generating vessels
                # for [4,L] Dim: x,y,z,diam; for each branch in vessels have 5 intermidiate coordianate point

                update = np.loadtxt(
                    inpath
                    + "update_d"
                    + str(d0mean)
                    + "_dr"
                    + str(d)
                    + "_epsilon"
                    + str(epsilon)
                    + "_iter"
                    + str(niter)
                    + ".txt"
                )
                print(" ")
                print(
                    "Load: "
                    + "update_d"
                    + str(d0mean)
                    + "_dr"
                    + str(d)
                    + "_epsilon"
                    + str(epsilon)
                    + "_iter"
                    + str(niter)
                    + ".txt"
                )

                # control the vessel's size for a same vessel
                dmin = np.nanmin(update, axis=1) * 1.1
                dmax = np.nanmax(update, axis=1) * 1.1
                print(dmin,dmax)
                # parameters for boulus injection
                sigma = 500
                v = 1
                t_start = 0
                t_end = 1
                t_step = 100
                for t in range(t_start, t_end, t_step):
                    Out_volume_name = (
                        outpath
                        + "Lnet_d{}_dr{}_epsilon{}_iter{}_SD{}_v{}_t{}_{}x{}x{}_nofluid.tiff".format(
                            d0mean,
                            d,
                            epsilon,
                            niter,
                            sigma,
                            v,
                            t,
                            tissueVolume[0],
                            tissueVolume[1],
                            tissueVolume[2],
                        )
                    )
                    if os.path.exists(Out_volume_name):
                        # continue
                        pass
                    # resize to ideal voxel size then calculate distance
                    updaten = resize_network(update, dmin, dmax, tVol=tissueVolume)
                    print("Time for resize voxels:", str(time.time() - start))
                    try:
                        updaten_bolus, max_dist = bolus_injection(
                            updaten, v, t, sigma=sigma, interp_coords_factor=0
                        )
                    except:
                        print("bad vessel")
                        break
                    # np.savetxt('D:\\update_bolus.txt', updaten_bolus)
                    print("max distance:", max_dist)
                    print("Time for injection simulated:", str(time.time() - start))
                    # rs=False means no need to resize the network
                    # img = (255*(process_network_fluid(
                    #     updaten_bolus, dmin, dmax, tVol=tissueVolume, rs=False
                    # ) )).astype("uint8")

                    # without fluid
                    img = (255*process_network(
                        update, tVol=tissueVolume
                    )).astype("uint8")

                    # verts, faces, normals, values = measure.marching_cubes(img, 1)

                    # obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

                    # for i, f in enumerate(faces):
                    #     obj_3d.vectors[i] = verts[f]
                    # obj_3d.save('test.stl')


                    print(np.max(img),np.min(img))
                    io.imsave(Out_volume_name, img, bigtiff=False)
                    # img.tofile(Out_volume_name)
                    print("Generating Projections……")
                    # -----------Volumes To CT Projection--------#
                    os.system(
                        "./synthetic-ddsa/SimVessels/SimVesselProjs.exe %s %s %d %d %d 0.15 0.15 0.15 %d %d %d"
                        % (
                            Out_volume_name,
                            OutPath_CTproj,
                            tissueVolume[2],
                            tissueVolume[1],
                            tissueVolume[0],
                            CTsize[0],
                            CTsize[1],
                            num_projection,
                        )
                    )

                    for filename in os.listdir(outpath):
                        if "raw" in filename:
                            print(filename)
                            os.remove(outpath + filename)
                    print("Time:", str(time.time() - start))
                    # sys.exit(0)


if __name__ == "__main__":
    main()
