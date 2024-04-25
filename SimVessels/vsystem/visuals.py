from math import isnan
from skimage import io
import napari
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("bmh")  # Use some nicer default colors


def plot_coords(coords, array=True, bare_plot=False):
    """
    Takes a list of coordinates and coverts to correct format for matplotlib
    """

    ax = plt.figure()
    if bare_plot:
        # Turns off the axis markers.
        ax = plt.axis("off")
    else:
        # Ensures equal aspect ratio.
        ax = ax.gca(projection="3d")

    if array:
        print(coords.shape)
        for i in range(coords.shape[1] - 1):
            # print('%f %f %f' %(coords[0,i], coords[1,i], coords[2,i]))
            ax.plot(
                coords[0, i : i + 2],
                coords[1, i : i + 2],
                coords[2, i : i + 2],
                color="blue",
            )
    else:
        # Converts a list of coordinates into
        # lists of X and Y values, respectively.
        X, Y, Z, alpha, beta, diam, _, _, _, _ = zip(*coords)
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        diam = np.array(diam)

        for i in range(len(X) - 1):
            # print(Z[i:i+2])
            # print('break')
            ax.plot(
                X[i : i + 2],
                Y[i : i + 2],
                Z[i : i + 2],
                linewidth=0.5 * diam[i],
                color="blue",
            )

    plt.show()


def print_coords(coords):
    for x, y, z, _, _, _, _ in coords:
        if isnan(x):
            print("<gap>")
        else:
            print("({:.2f}, {:.2f}, {:.2f})".format(x, y, z))


def plot_tiff(file, rgb=False):
    """
    function to plot stacked tif images t
    """
    data = io.imread(file)

    viewer = napari.Viewer(ndisplay=3)
    new_layer = viewer.add_image(data, rendering='minip', blending='additive', interpolation3d='linear', rgb=False)
    viewer.axes.visible = True
    viewer.camera.angles = (0, 0, 90)
    viewer.camera.zoom = 1
    napari.run()

    return viewer

def plot_tiffs(files, rgb=False):
    """
    function to plot stacked tif images t
    """
    viewer = napari.Viewer(ndisplay=3)
    for file in files:
        data = io.imread(file)

        # convert data from XYZ to ZYX in order to change left handed coordinate system to right handed in napari 
        # in napari axis 2 is x axis, axis 1 is y and axis 0 is z
        # data = np.transpose(data,(2,1,0))

        
        new_layer = viewer.add_image(data, rendering='mip', blending='additive', interpolation3d='linear', rgb=False)
        viewer.axes.visible = True
        viewer.camera.angles = (0, 90, 0)
        viewer.camera.zoom = 1
    napari.run()

    return viewer