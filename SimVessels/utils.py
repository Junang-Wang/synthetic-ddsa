import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt

def shpere_to_cartesian(s_coordinate):
    """
    input: sphere coordinate vector [r, polar angle, azimuthal angle]
    return: cartesian coordinate vector
    """
    x = s_coordinate[0]*np.sin(s_coordinate[1])*np.cos(s_coordinate[2])
    y = s_coordinate[0]*np.sin(s_coordinate[1])*np.sin(s_coordinate[2])
    z = s_coordinate[0]*np.cos(s_coordinate[1])

    c_coordinate = np.array([x,y,z])
    return c_coordinate


def compute_rotation_matrix(n, angle):

    # quaternion
    q = np.concatenate((n*np.sin(angle/2),np.array([np.cos(angle/2)])))
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    # rotation matrix
    rvec = np.array([
        [1-2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)], 
        [2*(x*y + z*w), 1-2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x**2 + y**2)]])
    
    return rvec



class projector:

    def __init__(self, data) -> None:
        self.data = data 
        self.points = np.column_stack(np.nonzero(data))

    def camera_init(self, view_angle_x, view_angle_y, width, height):

        """
        set up intrinsic property of camera 
        view_angle_x: camera's view angle in x axis
        view_angle_y: camera's view angle in y axis
        width:        camera sensor width in pixel unit
        height:       camera sensor height in pixel unit 
        """
        # camera matrix
        self.view_angle_x = view_angle_x
        self.view_angle_y = view_angle_y

        self.W = width #pixel 
        self.H = height #pixel

        # focal length in pixel unit
        fx = self.W/2 / np.tan(self.view_angle_x/2)
        fy = self.H/2 / np.tan(self.view_angle_y/2)

        # center of the image
        cx = int(self.W/2) 
        cy = int(self.H/2)

        self.K = np.array(
            [[fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.]]
        )
        self.distCoeffs = np.zeros((5,1), dtype='float')
    

    def place_camera(self, location, lookAt):

        """
        place camera to the location point and set up the lookAt point

        The sensor plane is the tangent plane(d_phi, d_theta) of the sphere which centered at lookAt point
        ---------------------------------------------
        new_z: inverse optical axis
        ---------------------------------------------
        Return: None, but compute rotation matrix and translation vector
        """
        # inverse optical axis
        self.location = location
        self.new_z = (location-lookAt)/ np.linalg.norm(location - lookAt)

        # compute polar angle, azimuthal angle
        theta = np.arccos(self.new_z[2])
        phi = np.sign(self.new_z[1]+0.001)*np.arccos(self.new_z[0]/np.sqrt(self.new_z[0]**2 + self.new_z[1]**2))
        print("theta,phi:",theta, phi)
        # print(self.new_z)

        # compute rotation axis
        axis_z = np.array([0,0,1])
        axis_y = np.array([0,1,0])
        cross_p = np.cross(self.new_z, axis_z)
        cross_p_norm = np.linalg.norm(cross_p)

        if cross_p_norm == 0:
            cross_p = axis_y
            cross_p_norm = 0
            self.n = cross_p
            # self.new_z = axis_z
        else:
            self.n = cross_p/ cross_p_norm


        # print('n :',self.n)
        self.angle = np.arccos(self.new_z@ axis_z)
        # print(self.angle)

        axis_xy = np.array([1,1,0])/np.sqrt(2)

        # rotate coordinate toward the lookAt point
        lookAt_R = compute_rotation_matrix(axis_xy, np.pi)


        # compute rotation matrix
        self.R = lookAt_R @ compute_rotation_matrix(self.n, self.angle)
        self.tvec = -self.R@location

    def twist_camera(self, angle):
        axis_z = np.array([0,0,1])
        self.R = compute_rotation_matrix(axis_z,angle) @ self.R
        self.tvec = -self.R@self.location
        
    def project(self, opacity=0.015):
        """
        opacity: the opacity of one 255 pixel
        Refs: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#/ga1019495a2c8d1743ed5cc23fa0daff8c

        Return image(Y,X)
        """
        # project to 2D position pixel unit
        points_2D, jacobian = cv2.projectPoints(self.points.astype(float),rvec=self.R, tvec=self.tvec, cameraMatrix=self.K, distCoeffs=self.distCoeffs)

        points_2D = np.squeeze(np.round(points_2D),axis=1).astype(int)

        # mapping to image
        image = np.ones( (self.W, self.H), dtype=float )

        N_points = self.points.shape[0]
        for i in range(N_points):
            if points_2D[i,0] >0 and points_2D[i,0] < self.H and points_2D[i,1] >0 and points_2D[i,1] < self.W: 
                image[points_2D[i,1], points_2D[i,0]] *= (1-opacity*self.data[self.points[i,0],self.points[i,1],self.points[i,2]]/255)
        image = (255*(1-image)).astype('uint8')
        # image = np.flip((255*(1-image)).astype('uint8'),1)

        plt.imshow(image, cmap='gray')
        # plt.colorbar()

        return image

