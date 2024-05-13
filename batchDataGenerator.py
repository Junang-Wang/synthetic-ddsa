from SimV.syntheticDSA import syntheticDSA
import numpy as np
import os

def main():
    # d0_array = np.repeat(np.arange(25,30),1)
    d0_array = np.arange(25,30,0.025)
    niter_array = np.array([8,9])
    epsilon_array = np.array([8,9])
    d_array = np.array([15,20])

    # d0_array = np.array([25])
    # niter_array = np.array([8])
    # epsilon_array = np.array([8])
    # d_array = np.array([15])

    tissueVolume = [512, 512, 395]
    nProj = 4

    string_folder = os.path.expanduser("~/syntheticDSA/string")
    stl_folder = os.path.expanduser("~/syntheticDSA/stl")
    images_folder = os.path.expanduser("~/syntheticDSA/images/proj_"+str(nProj).zfill(2))
    XrayConf = "./SimV/configuration-03.json"
    DSA_synth = syntheticDSA(d0_array, niter_array, epsilon_array, d_array, tissueVolume)
    # DSA_synth.string2stl(string_folder, stl_folder)
    DSA_synth.stl2images(nProj, XrayConf, stl_folder, images_folder)

if __name__ == "__main__":
    main()