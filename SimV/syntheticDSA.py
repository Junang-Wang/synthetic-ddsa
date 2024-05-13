import numpy as np
import os, json5
from SimV.utils import simulate_Xray
from SimV.vsystem.analyseGrammar import branching_turtle_to_coords
from SimV.vsystem.computeVoxel import process_network
from SimV.vsystem.libGenerator import setProperties
from SimV.vsystem.utils import bezier_interpolation
from SimV.vsystem.vSystem import F, I
from vedo import Volume
from skimage import io
import time

class syntheticDSA:

    def __init__(self, d0_array, niter_array, epsilon_array, d_array, tissueVolume):
        """
        d0_array:      numpy array of initial diameter of vascular
        niter_array:   numpy array of number of iteration of L-system
        epsilon_array: numpy array of the proportion between length & diameter
        d_array:       numpy array of the ratio between d0 to its subbranch
        """

        self.d0_array = d0_array
        self.niter_array = niter_array
        self.epsilon_array = epsilon_array
        self.d_array = d_array 
        self.tissueVolume = tissueVolume

    def string2stl(self, string_folder, stl_folder):
        """
        tissueVolume:  volume size
        string_folder:        save path of string
        stl_folder:        save path of stl files
        """
        
        self.stringGenerator(self.d0_array, self.niter_array, self.epsilon_array, self.d_array, string_folder)

        self.stlGenerator(self.d0_array, self.niter_array, self.epsilon_array, self.d_array, self.tissueVolume, string_folder, stl_folder)
    
    def stl2images(self, nProj, XrayConf, stl_folder, images_folder):
        """
        nProj:    number of 2D Xray projections
        XrayConf: Xray source and detector configuration
        """

        self.XrayProjector(self.d0_array, self.niter_array, self.epsilon_array, self.d_array,self.tissueVolume, nProj, stl_folder, images_folder, XrayConf)

    @staticmethod
    def stringGenerator(d0_array, niter_array, epsilon_array, d_array, string_path):
        """
        This function generate string for L-system, and save vessels coordinates information to .txt files
        ------------input--------
        d0_array:      numpy array of initial diameter of vascular
        niter_array:   numpy array of number of iteration of L-system
        epsilon_array: numpy array of the proportion between length & diameter
        d_array:       numpy array of the ratio between d0 to its subbranch
        string_path:       save path of string

        --------output----------
        string for generating L-system


        """

        if not os.path.exists(string_path):
            os.makedirs(string_path)

        # Lindenmayer System Parameters
        properties = {
            "k": 3,
            "epsilon": 6,  # random.uniform(4,10), # Proportion between length & diameter
            "randmarg": 3,  # Randomness margin between length & diameter
            "sigma": 5,  # Determines type deviation for Gaussian distributions
            "d": 2,
            "stochparams": True,
        }  # Whether the generated parameters will also be stochastic
        i = 0
        for d0 in d0_array:
            for niter in niter_array:
                for epsilon in epsilon_array:  # differ Proportion between length & diameter
                    properties["epsilon"] = epsilon
                    for d in d_array:  # ratio between d0 to its subbranch
                        properties["d"] = d / 10

                        setProperties(properties)  # Setting L-System properties

                        print(
                            "Creating image ... with %i iterations %i dosize %i d "
                            % (niter, int(d0), d)
                        )

                        """ Run L-System grammar for n iterations """
                        turtle_program = F(niter, d0)

                        """ Convert grammar into coordinates """
                        coords = branching_turtle_to_coords(turtle_program, d0)

                        """ Analyse / sort coordinate data """
                        update = bezier_interpolation(coords)

                        # print(type(update))
                        np.savetxt(
                            string_path
                            + "/"
                            + "update_d"
                            + str(int(d0))
                            + "_dr"
                            + str(d)
                            + "_epsilon"
                            + str(epsilon)
                            + "_iter"
                            + str(niter)
                            + "_"
                            + str(i).zfill(4)
                            + ".txt",
                            update,
                        )
                        i += 1
    @staticmethod
    def stlGenerator(d0_array, niter_array, epsilon_array, d_array, tissueVolume, string_folder, stl_folder):
        """
        This function generates vascular stl files with given vascular strings
        ------------input--------
        d0_array:      numpy array of initial diameter of vascular
        niter_array:   numpy array of number of iteration of L-system
        epsilon_array: numpy array of the proportion between length & diameter
        d_array:       numpy array of the ratio between d0 to its subbranch
        tissueVolume:  volume size
        string_folder:        save path of string
        stl_folder:        save path of stl files

        --------output----------
        stl files
        """
        # ---------------parameters--------------#
        # tissueVolume = (512, 512, 280)

        if not os.path.exists(stl_folder):
            os.makedirs(stl_folder)
        
        stl_n = 0
        for d0 in d0_array:
            for niter in niter_array:
                for epsilon in epsilon_array:  # differ Proportion between length & diameter
                    for d in d_array:  # ratio between d0 to its subbranch

                        # load cooridate for generating vessels
                        # for [4,L] Dim: x,y,z,diam; for each branch in vessels have 5 intermidiate coordianate point

                        update = np.loadtxt(
                            string_folder
                            +"/"
                            + "update_d"
                            + str(int(d0))
                            + "_dr"
                            + str(d)
                            + "_epsilon"
                            + str(epsilon)
                            + "_iter"
                            + str(niter)
                            + "_"
                            + str(stl_n).zfill(4)
                            + ".txt"
                        )
                        print(" ")
                        print(
                            "Load: "
                            + "update_d"
                            + str(int(d0))
                            + "_dr"
                            + str(d)
                            + "_epsilon"
                            + str(epsilon)
                            + "_iter"
                            + str(niter)
                            + "_"
                            + str(stl_n).zfill(4)
                            + ".txt"
                        )


                        Out_stl_name = (
                            stl_folder + "/"
                            + "Lnet_d{}_dr{}_epsilon{}_iter{}_i{}_{}x{}x{}_nofluid.stl".format(
                                int(d0),
                                d,
                                epsilon,
                                niter,
                                str(stl_n).zfill(4),
                                tissueVolume[0],
                                tissueVolume[1],
                                tissueVolume[2],
                            )
                        )
                        stl_n += 1

                        # generate vessel volume without fluid, if one need boulus injection please check SimVessels
                        vessel_v = (255*process_network(
                            update, tVol=tissueVolume
                        )).astype("uint8")

                        # convert volume to mesh by vedo
                        # vol = Volume(vessel_v)

                        # # isovalues = list(range(255))
                        # isovalues = [0, 255]
                        # # mesh = vol.isosurface_discrete(isovalues, nsmooth=12)
                        # mesh = vol.isosurface(isovalues).smooth(niter=50, boundary=True)

                        # mesh.write(Out_stl_name)

                        # convert volume to mesh by slicer
                        Out_volume_name = Out_stl_name[:-3]+"tiff"
                        slicer_stl_name = Out_stl_name[len(stl_folder)+1:-4]
                        slicer_path = '/home/qubot/App/Slicer-5.6.2-linux-amd64/Slicer'
                        io.imsave(Out_volume_name, vessel_v, bigtiff = False)
                        python_script_path = os.path.abspath('./SimV/slicer_script.py')
                        command = slicer_path+ ' --no-main-window'+' --python-script ' +python_script_path + " " + Out_volume_name + " " + stl_folder + " " + slicer_stl_name
                        # command = slicer_path +' --python-script ' +python_script_path + " " + Out_volume_name + " " + stl_folder + " " + slicer_stl_name
                        os.system(command=command) 

                        os.remove(Out_volume_name)

                        # for filename in os.listdir(stl_folder):
                        #     if "tiff" in filename:
                        #         print(filename)
                        #         os.remove(stl_folder + filename) 

        
    @staticmethod
    def XrayProjector(d0_array, niter_array, epsilon_array, d_array, tissueVolume, nProj, stl_folder, images_folder, XrayConf = "configuration-03.json"):
        """
        nProj:    number of 2D Xray projections
        XrayConf: Xray source and detector configuration
        """
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        #---------load config------------
        with open(XrayConf) as f:
            params = json5.load(f)
        
        r = abs(params['Source']['Position'][0])
        
        #--------set source position----------
        phi_n = 1
        theta_n = nProj
        phi = np.linspace(0, 2*np.pi, phi_n, endpoint=False)
        theta = np.linspace(np.pi/(theta_n), np.pi, theta_n, endpoint=True)
        u, v = np.meshgrid(phi, theta)
        x = r*np.cos(u) * np.sin(v) 
        y = r*np.sin(u) * np.sin(v) 
        z = r*np.cos(v) 

        lookAt = np.array([0,0,0])

        stl_n = 0
        for d0 in d0_array:
            for niter in niter_array:
                for epsilon in epsilon_array:  # differ Proportion between length & diameter
                    for d in d_array:
                        stl_name = (
                            stl_folder
                            +"/"
                            + "Segmentation_"
                            + "Lnet_d{}_dr{}_epsilon{}_iter{}_i{}_{}x{}x{}_nofluid.stl".format(
                                int(d0),
                                d,
                                epsilon,
                                niter,
                                str(stl_n).zfill(4),
                                tissueVolume[0],
                                tissueVolume[1],
                                tissueVolume[2],
                            )
                        )
                        
                        params["Samples"][0]["Path"] = stl_name

                        for i in range(theta_n):
                            for j in range(phi_n):
                                source_position = np.array([x[i,j], y[i,j],z[i,j]])        
                                save_path = (
                                            images_folder
                                            +"/"
                                            + "Lnet_d{}_dr{}_epsilon{}_iter{}_i{}_nP{}_{}x{}x{}_nofluid.tiff".format(
                                                int(d0),
                                                d,
                                                epsilon,
                                                niter,
                                                str(stl_n).zfill(4),
                                                str(i).zfill(2),
                                                tissueVolume[0],
                                                tissueVolume[1],
                                                tissueVolume[2],
                                                )
                                            )
                                x_ray_image = simulate_Xray(params, source_position, lookAt, save_path)
                                # time.sleep(0.5)

                        stl_n += 1