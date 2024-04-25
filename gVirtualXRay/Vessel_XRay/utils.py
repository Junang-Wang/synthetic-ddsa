import numpy as np
import json2gvxr
from tifffile import imwrite # Write TIFF files

from gvxrPython3 import gvxr # Simulate X-ray images

def sphere_to_cartesian(s_coordinate):
    """
    input: sphere coordinate vector [r, polar angle, azimuthal angle]
    return: cartesian coordinate vector
    """
    x = s_coordinate[0]*np.sin(s_coordinate[1])*np.cos(s_coordinate[2])
    y = s_coordinate[0]*np.sin(s_coordinate[1])*np.sin(s_coordinate[2])
    z = s_coordinate[0]*np.cos(s_coordinate[1])

    c_coordinate = np.array([x,y,z])
    return c_coordinate

def cartesian_to_sphere(c_coordinate):
    """
    input: cartesian coordinate vector
    return: sphere coordinate vector [r, polar angle, azimuthal angle]
    """
    x,y,z = c_coordinate
    r = np.sqrt(np.sum(c_coordinate**2 ))
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)


    s_coordinate = np.array([r,theta,phi])
    return s_coordinate

def theta_unit_vector(s_coordinate):
    '''
    compute theta unit vector in cartesian coordinate
    '''
    r, theta, phi = s_coordinate

    dx = np.cos(theta)*np.cos(phi)
    dy = np.cos(theta)*np.sin(phi)
    dz = -np.sin(theta)

    return np.array([dx,dy,dz])



def compute_up_vector(source_position,lookAt):
    detector_unit_vector = (lookAt-source_position)/ np.linalg.norm(lookAt-source_position)

    s_detector_unit_vector = cartesian_to_sphere(detector_unit_vector)

    up_vector = theta_unit_vector(s_detector_unit_vector)

    return up_vector

def set_source_detector(params, source_position,lookAt):

    list_source_position = source_position.tolist()
    list_source_position.append('cm')
    params['Source']['Position'] = list_source_position

    list_detector_position = (lookAt-source_position).tolist()
    list_detector_position.append('cm') 
    params['Detector']['Position'] = list_detector_position

    params['Detector']['UpVector'] = compute_up_vector(source_position, lookAt)

    return params

def simulate_Xray(params, source_position, lookAt, save_path):
    '''
    simulate Xray by gvxr
    '''
    params = set_source_detector(params, source_position, lookAt)
    # create an OpenGL context
    json2gvxr.initGVXR(Params=params, renderer="OPENGL")

    # Set up a monochromatic source
    json2gvxr.initSourceGeometry()
    spectrum, unit, k, f = json2gvxr.initSpectrum()

    # Set up the detector
    json2gvxr.initDetector()

    # Load sample data
    json2gvxr.initSamples()
    print("Move the Vessel to the centre")
    gvxr.moveToCentre("Vessel")
    # gvxr.scaleScene(0.5,0.5,0.5) # rescale stl


    # Rotation stl file
    # Rotation angle, rotate_vector_x, rotate_vector_y, rotate_vector_z
    # gvxr.rotateScene(90, 0, 0, 1)


    # Compute an X-ray image
    x_ray_image = np.array(gvxr.computeXRayImage()).astype(np.single)

    # Save the X-ray image in a tiff file
    imwrite(save_path, x_ray_image)

    return x_ray_image