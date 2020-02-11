# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:11:04 2020

@author: ARCHIT
"""

import imageio
import numpy as np
from PIL import Image

#convert to numpy array
def load_img(filename):
    return np.array(Image.open(filename))

def create_gaussian_kernel(size, sigma = 1.0):
    
    #We create a gaussian kernel of dim : size*size
    #size must be a odd number
    
    if(size%2==0):
        raise ValueError("Size must be an odd number")
    
    x, y = np.meshgrid(np.linspace(-2, 2, size, dtype=np.float32), np.linspace(-2, 2, size, dtype=np.float32))
    
    rv = (np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (2.0 * np.pi * sigma ** 2)
    
    rv = rv / np.sum(rv)
    
    return rv

def convolve_pixel(img, kernel, i, j):
    
    # convolves the image kernel of pixel at (i,j). returns the original pixel
    # if the kernel extends beyond the borders of the image
    
    if(len(img.shape)!=2):
        raise ValueError("Input image should be single chanelled")
    if(len(kernel.shape)!=2):
        raise ValueError("kernel should be two dimensional")
    
    k = kernel.shape[0]//2
    
    # checking whether kernel is beyong the border
    if i < k or j < k or i >= img.shape[0]-k or j >= img.shape[1]-k:
        return img[i, j]
    
    else:
        value = 0
        for u in range(-k,k+1):
            for v in range(-k,k+1):
                value += img[i-u][j-v] * kernel[k+u][k+v]
        
        return value
    
def convolve(img, kernel):
    
    # returns the convoluted image in a new variable
    # kenel should have odd dimensions and image should be single chanelled and a two dim ndarray
    
    new_img = np.array(img)     #make a copy of the original image in which we will return the result
    
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            new_img[i][j] = convolve_pixel(img, kernel, i, j)
            
    return new_img

def split(img):
    
    # splits image into 3 ndarrays, one for each channel (R,G,B)
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]
            
def merge(r,g,b):
    
    return np.dstack((r, g, b))

if __name__ == '__main__':
    input_image = load_img("disha.jpeg")
    
    kernel = create_gaussian_kernel(9)
    
    (r,g,b) = split(input_image)
    
    r = convolve(r, kernel)
    g = convolve(g, kernel)
    b = convolve(b, kernel)
    
    output_image = merge(r,g,b)
    
    imageio.imwrite('blurred_disha.jpg', output_image)
    
    
        