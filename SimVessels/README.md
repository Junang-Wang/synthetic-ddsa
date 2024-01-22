# DDSA using Synthetic Training Data

This is the code for generate synthetic vascular projection images

**Paper:** Training of a Deep Learning Based Digital Subtraction Angiography Method using Synthetic Data  
**Author:** Duan, Lizhen; Eulig, Elias; Knaup, Michael; Adamus, Ralf; Lell, Michael; Kachelrieß, Marc

## Contents
```
├─ vsystem/                 Folder containing functions for generate vessels with stochastic L-system
├─ bolus/                   Folder ocntaining functions for bolus injection simulation
├─ stringGenerator.py       Generate and save vessel strings
├─ projectionGenerator.py   Generate vascular projection images with simulated bolus injections using given vessel strings
├─ SimVesselProjs.exe       Project 3D vascular images into 2D images
```

## How to use
1. Run stringGenerator.py to generate and save vessel strings.
2. Run projectionGenerator.py to generate vascular projection images with simulated bolus injections using the generated vessel strings.

## About SimVesselProjs.exe
The executable, `SimVesselProjs.exe`, projects 3D vascular images into 2D images.

To use `SimVesselProjs.exe`, follow these steps:
1. Open your terminal.
2. Run the following command: `SimVesselProjs.exe InputPath OutputPath Nx Ny Nz dx dy dz Nu Nv Na`  
where  
    * InputPath: path to a float32 volume (e.g. path\to\volume\Lnet_1024x1024x1024.raw)  
    * OutputPath: path to the output folder (e.g. path\to\outputfolder)  
    * Nx, Ny, Nz: Size of input volume  
    * dx, dy, dz: Voxel size in mm of the volume (e.g. 0.15)  
    * Nu, Nv: Number of detector rows and columns (e.g. 1024)  
    * Na: Number of projections to simulate (e.g. 16)  

Example:  
`
SimVesselProjs.exe \InputFolder\Lnet_1024x1024x1024.raw \OutputFolder 1024 1024 1024 0.15 0.15 0.15 1024 1024 16
`

## Acknowledgements
I would like to acknowledge the authors and contributors of the code that I referenced from the GitHub repository [https://github.com/psweens/V-System].
