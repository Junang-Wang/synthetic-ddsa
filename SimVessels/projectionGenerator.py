"""
Generate vascular projection images with given vascular strings
"""
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from bolus.bolusInjection import bolus_injection
from vsystem.computeVoxel_fluid import process_network_fluid, resize_network


def main():
    # ---------------parameters--------------#
    tissueVolume = (1024, 1024, 1024)

    CTsize = (1024, 1024)  # size of final projections
    num_projection = 16  # number of projections for a volume

    inpath = r"vessels\update_save\\"
    OutPath_CTproj = r"vessels\bolus_chase"
    if not os.path.exists(OutPath_CTproj):
        os.makedirs(OutPath_CTproj)
    outpath = r"vessels\volumes\\"

    domean_min = 10
    domean_max = 60
    d0std = 5.0
    for d0mean in range(domean_min, domean_max + 1, int(d0std)):
        for d in [12, 15, 20]:  # 10 times Proportion between d0 & d1
            for epsilon in range(4, 10):  # differ Proportion between length & diameter
                for niter in range(6, 14):
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

                    # parameters for boulus injection
                    sigma = 500
                    v = 1
                    t_start = 600
                    t_end = 1001
                    t_step = 1000
                    for t in range(t_start, t_end, t_step):
                        Out_volume_name = (
                            outpath
                            + "Lnet_d{}_dr{}_epsilon{}_iter{}_SD{}_v{}_t{}_{}x{}x{}.raw".format(
                                d0mean,
                                d,
                                epsilon,
                                niter,
                                sigma,
                                v,
                                t,
                                tissueVolume[2],
                                tissueVolume[1],
                                tissueVolume[0],
                            )
                        )
                        if os.path.exists(Out_volume_name):
                            continue
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
                        img = process_network_fluid(
                            updaten_bolus, dmin, dmax, tVol=tissueVolume, rs=False
                        ).astype("float32")

                        img.tofile(Out_volume_name)
                        print("Generating Projections……")
                        # -----------Volumes To CT Projection--------#
                        os.system(
                            "VascularProjection.exe %s %s %d %d %d 0.15 0.15 0.15 %d %d %d"
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
