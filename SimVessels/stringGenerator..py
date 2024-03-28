""" save vessels coordinates information to .txt files """
import os
import random
import sys

import cv2
import numpy as np
from skimage import io

sys.path.append("vsystem")

import sys

from vsystem.analyseGrammar import branching_turtle_to_coords
from vsystem.computeVoxel import process_network
from vsystem.libGenerator import setProperties
from vsystem.preprocessing import resize_stacks, resize_volume
from vsystem.utils import bezier_interpolation
from vsystem.visuals import plot_coords, print_coords
from vsystem.vSystem import F, I

# -------parameters setting-------------#
domean_min = 10
domean_max = 61
d0std = 60.0  # Standard deviation of base diameter


def main():
    outpath = "./vessels/update_save/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Lindenmayer System Parameters
    properties = {
        "k": 3,
        "epsilon": 6,  # random.uniform(4,10), # Proportion between length & diameter
        "randmarg": 3,  # Randomness margin between length & diameter
        "sigma": 5,  # Determines type deviation for Gaussian distributions
        "d": 2,
        "stochparams": True,
    }  # Whether the generated parameters will also be stochastic

    for d0mean in range(domean_min, domean_max + 1, int(d0std)):
        d0 = np.random.normal(
            d0mean, d0std
        )  # Randomly assign base diameter (no dimension)
        for niter in range(6, 14):
            for epsilon in range(4, 10):  # differ Proportion between length & diameter
                properties["epsilon"] = epsilon
                for d in [12, 15, 20]:  # ratio between d0 to its subbranch
                    properties["d"] = d / 10

                    setProperties(properties)  # Setting L-System properties

                    print(
                        "Creating image ... with %i iterations %i dosize %i d "
                        % (niter, d0mean, d)
                    )

                    """ Run L-System grammar for n iterations """
                    turtle_program = F(niter, d0)

                    """ Convert grammar into coordinates """
                    coords = branching_turtle_to_coords(turtle_program, d0)

                    """ Analyse / sort coordinate data """
                    update = bezier_interpolation(coords)

                    # print(type(update))
                    np.savetxt(
                        outpath
                        + "update_d"
                        + str(d0mean)
                        + "_dr"
                        + str(d)
                        + "_epsilon"
                        + str(epsilon)
                        + "_iter"
                        + str(niter)
                        + ".txt",
                        update,
                    )

                    """ If you fancy, plot a 2D image of the network! """
                    # plot_coords(update, array=True, bare_plot=False) # bare_plot removes the axis labels

                    """ Run 3D voxel traversal to generate binary mask of L-System network """
                    # image = process_network(update, tVol=tissueVolume)

                    """ Convert to 8-bit format """
                    # image = (255*image).astype('int8')
                    """ Save image volume """
                    # io.imsave(outpath+"Lnet_i{}_{}.tiff".format(niter,file), np.transpose(image, (2, 0, 1)), bigtiff=False)

                    # image = (1024*image).astype('float32')
                    # print(image.shape)
                    # image = np.transpose(image, (2, 0, 1))
                    # print(image.shape)
                    # image.tofile(outpath+"Lnet_i{}_{}_{}x{}x{}.raw".format(niter,file,tissueVolume[1],tissueVolume[0],tissueVolume[2]))


if __name__ == "__main__":
    main()
