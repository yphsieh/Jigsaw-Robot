#!/usr/bin/env python
import numpy as np
import math
import cv2

class calibrate:
    def __init__(self, image_points, object_points, intrinsic_matrix, dist_coeff):
        self.image_points = image_points
        self.object_points = object_points
        self.intrinsic_matrix = intrinsic_matrix
        self.dist_coeff = dist_coeff
        self.extrinsic_matrix = self.calculate_extrinsic_matrix()
    
    def calculate_extrinsic_matrix(self):
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.object_points, self.image_points, self.intrinsic_matrix, self.dist_coeff, flags = cv2.SOLVEPNP_ITERATIVE)
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        extrinsic_matrix = np.concatenate((rotation_matrix,translation_vector),axis = 1)
        return extrinsic_matrix

    def transform_pixel_to_world(self, image_point):
        imx = self.intrinsic_matrix
        emx = self.extrinsic_matrix
        s = 957
        image_point = np.concatenate((image_point,np.array([1]) ),axis=0)
        inv_intrin = np.linalg.inv(imx)
        tmp3 = np.dot(inv_intrin,image_point)
        tmp3 = tmp3 * s
        tmp4 = np.concatenate((tmp3,np.array([1])),axis=0)
        tmp5 = np.concatenate((emx,np.array([[0,0,0,1]])),axis=0)
        inv_extrin = np.linalg.inv(tmp5)
        guess_world_point = np.dot(inv_extrin,tmp4)

        # tune the ratio guess
        tmp6 = guess_world_point[0:2]
        tmp7 = np.concatenate((tmp6,np.array([0])),axis = 0)
        tmpw = np.concatenate((tmp7,np.array([1])),axis = 0)
        tmp1 = np.dot(imx,emx)
        tmp2 = np.dot(tmp1,tmpw)
        guess_pixel_point = tmp2/s
        tune_ratio = guess_pixel_point[2]
        s = s*tune_ratio

        # transform again
        tmp8 = np.dot(inv_intrin,image_point)
        tmp8 = tmp8 * s
        tmp9 = np.concatenate((tmp8,np.array([1])),axis=0)
        tmp10 = np.concatenate((emx,np.array([[0,0,0,1]])),axis=0)
        inv_extrin = np.linalg.inv(tmp10)
        guess_world_point = np.dot(inv_extrin,tmp9)

        return guess_world_point

    def transform_world_to_pixel(self, object_point):
        imx = self.intrinsic_matrix
        emx = self.extrinsic_matrix
        s = 957   
        object_point = np.concatenate((object_point,np.array([1])),axis = 0)
        tmp1 = np.dot(imx,emx)
        tmp2 = np.dot(tmp1,object_point)
        guess_pixel_point = tmp2/s
        # tune the ratio guess
        tune_ratio = guess_pixel_point[2]
        guess_pixel_point = guess_pixel_point / tune_ratio
        return guess_pixel_point

