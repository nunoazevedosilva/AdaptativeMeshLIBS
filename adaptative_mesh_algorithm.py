# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:57:49 2021

@author: nunoa
"""


import os
from matplotlib.pyplot import *
import numpy as np
from skimage import io, filters

import matplotlib



class Block:
    
    """
    
    Instance is a block of a given mesh
    
    """
    
    def __init__(self, coordinates,level=0):
        
        #list that contains coodinates as [[x0,y0]],
        # 1--2
        # |  |
        # 0--3
        self.coordinates = np.array(coordinates)
        
        #bool to store if it is already tested to the criteria
        self.is_checked = False
        
        #number of iteration level it belongs
        self.level = level
        
        ##store connections between points to plot lines of reference
        x0 = coordinates[0][0]
        x1 = coordinates[3][0]
        y0 = coordinates[0][1]
        y1 = coordinates[1][1]
        self.block_connections = [[x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0]]
    
    def check_block(self,threshold,image):
        
        """
        Method to check if a block should be divided
        """
        
        #limits of the image
        x_min = int(self.coordinates[0][0])
        x_max = int(self.coordinates[3][0])
                                        
        y_min = int(self.coordinates[0][1])
        y_max = int(self.coordinates[1][1])
                    
        #compute the criteria
        criteria, value = image.compute_criteria([[x_min,x_max],
                                           [y_min,y_max]])
        
        self.is_checked = True
        
        #return true if it shoud be divided
        return value>threshold
    
    def __repr__(self):
        return "Block - "+ "x = [" + str(self.coordinates[0][0]) + ', ' +str(self.coordinates[3][0]) + ']\t' +   "y = [" + str(self.coordinates[0][1]) + ', ' +str(self.coordinates[1][1]) + ']\t' +", level " + str(self.level) + '\n'
    

class Mesh:
    
    """
    A mesh is a set of Blocks
    """
    
    def __init__(self, image):
        
        #list of current blocks that the present mesh has
        self.list_of_blocks = []
        
        num_divs_x = 3
        num_divs_y = 3
        
        step_x = (image.x_max//num_divs_x)
        step_y = (image.y_max//num_divs_y)
        
        for i in range(0,num_divs_x):
            for j in range(0,num_divs_y):
                xmin = image.x_min+i*step_x
                xmax = image.x_min+(i+1)*step_x
                ymin = image.x_min+j*step_y
                ymax = image.x_min+(j+1)*step_y
                
                self.list_of_blocks.append(Block([[xmin,ymin],
                                        [xmin, ymax],
                                        [xmax, ymax],
                                        [xmax, ymin]],level=0))
                
            
        # store the image, shall be of type image, defined in class
        self.image = image
    
    def update_blocks(self,threshold):
        
        """
        Check if 
        """
        
        new_list_of_blocks = []
        self.state = 0 #if 0 reached the end of update
        
        for i in range(0,len(self.list_of_blocks)):
            current_block = self.list_of_blocks[i]
            
            if not current_block.is_checked:
                self.state = 1
                    
                divide = current_block.check_block(threshold, self.image)
                
                if divide:

                    x_min = int(current_block.coordinates[0][0])
                    x_max = int(current_block.coordinates[3][0])
                    x_mean = int(x_min+(x_max-x_min)/2)

                    
                    y_min = int(current_block.coordinates[0][1])
                    y_max = int(current_block.coordinates[1][1])
                    y_mean = int(y_min+(y_max-y_min)/2)
                    
                    
                    new_block1 = Block([[x_min, y_min],
                                       [x_min, y_mean],
                                       [x_mean, y_mean],
                                       [x_mean, y_min]],level=current_block.level+1)
                    

                    new_block2 = Block([[x_min, y_mean],
                                       [x_min, y_max],
                                       [x_mean, y_max],
                                       [x_mean, y_mean]],level=current_block.level+1)
                    
                    new_block3 = Block([[x_mean, y_mean],
                                       [x_mean, y_max],
                                       [x_max, y_max],
                                       [x_max, y_mean]],level=current_block.level+1)
                    
                    
                    new_block4 = Block([[x_mean, y_min],
                                       [x_mean, y_mean],
                                       [x_max, y_mean],
                                       [x_max, y_min]],level=current_block.level+1)
                    
                    self.max_level = current_block.level+1
                    
                    new_list_of_blocks.append(new_block1)
                    new_list_of_blocks.append(new_block2)
                    new_list_of_blocks.append(new_block3)
                    new_list_of_blocks.append(new_block4)
            
                
                else:
                    new_list_of_blocks.append(current_block)
 
        
            
            else:
                new_list_of_blocks.append(current_block)
        
        
        self.list_of_blocks = new_list_of_blocks
        
    def adaptative_mesh(self,threshold, max_num_divisions=2):
        self.state = 1
        self.max_level = 0
        while self.max_level <= max_num_divisions and self.state == 1:

                self.update_blocks(threshold)
            
        
        
    def plot(self):
            coordinate_pairs = []
            levels = []
            #print(self.list_of_blocks)
            for i in range(0,len(self.list_of_blocks)):
                plot(self.list_of_blocks[i].block_connections[0],
                     self.list_of_blocks[i].block_connections[1], lw = .5, color= 'w',ls='--')
                
                for j in range(0,len(self.list_of_blocks[i].coordinates)):
                    coordinate_pairs.append(self.list_of_blocks[i].coordinates[j])
                    levels.append(self.list_of_blocks[i].level)
            
            coordinate_pairs = np.array(coordinate_pairs)
            cycle_colors = rcParams['axes.prop_cycle'].by_key()['color']
            
            for i in range(0,len(coordinate_pairs)):

                plot(coordinate_pairs[i,0],coordinate_pairs[i,1],
                         'o',ms=7,markeredgecolor='k', color=cycle_colors[levels[i]])
            
            self.coordinate_pairs = np.unique(coordinate_pairs, axis=1)
            return coordinate_pairs
    
    def get_coordinates_and_values(self, image):
        x_values = []
        y_values = []
        map_values = []
        for i in range(0, len(self.coordinate_pairs)):
            x_values.append(self.coordinate_pairs[i][0])
            y_values.append(self.coordinate_pairs[i][1])
            #print(x_values[-1],y_values[-1])
            map_values.append(image[self.coordinate_pairs[i][0]-1,
                              self.coordinate_pairs[i][1]-1])
            
        return x_values, y_values, map_values
    
class image:
    """
    A Class to contain images
    """
    
    def __init__(self, path,lims=False):
        self.path = path
        self.data_input()
        
        self.lims=lims
        
        if lims:
            self.ROI_data = self.data[lims[0][0]:lims[0][1],lims[1][0]:lims[1][1]]
            self.ROI_data_gray = self.data_gray[lims[0][0]:lims[0][1],lims[1][0]:lims[1][1]]
        else:
            self.ROI_data = self.data
            self.ROI_data_gray = self.data_gray
            
        self.x_min = 0
        self.y_min = 0
        self.x_max = np.shape(self.ROI_data_gray)[0]
        self.y_max = np.shape(self.ROI_data_gray)[1]
        self.compute_criteria()
    
    
    def data_input(self):
        self.data = np.transpose(io.imread(self.path),[1,0,2])
        self.data_gray = np.transpose(io.imread(self.path, as_gray=True))
    
    
    def plot_all(self, gray=False):
        title('Original Image')
        if not gray:
            imshow(np.transpose(self.data,[1,0,2]))
        else:
            imshow(np.transpose(self.data_gray),cmap='Greys')
        
        if self.lims:
            
            x0 = self.lims[0][0]
            x1 = self.lims[0][1]
            y0 = self.lims[1][0]
            y1 = self.lims[1][1]
            
            plot([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0],ls='--',lw=3,color='k')
            fill_between([x0,x1], [y0,y0],[y1,y1], alpha=0.3,color='k')
    def plot_ROI(self, gray=False):
        title("ROI")
        if not gray:
            imshow(np.transpose(self.ROI_data,[1,0,2]))
        else:
            imshow(np.transpose(self.ROI_data_gray),cmap='Greys')
    
    
    def compute_criteria(self,lims=False):
        if lims:
            criteria = filters.sobel(self.ROI_data_gray[lims[0][0]:lims[0][1],
                                                         lims[1][0]:lims[1][1]]) 
            
            value = np.mean(criteria)
            
            max_value = np.max(self.ROI_data_gray[lims[0][0]:lims[0][1], 
                               lims[1][0]:lims[1][1]])
            
            min_value = np.min (self.ROI_data_gray[lims[0][0]:lims[0][1], 
                               lims[1][0]:lims[1][1]])
            
            #value = max_value - min_value 

        else:
            criteria = filters.sobel(self.ROI_data_gray)
            value = 0
        
        return criteria, value
        
    def plot_criteria(self):
        title("Criteria")
        criteria =filters.sobel(self.ROI_data_gray)
        imshow(np.transpose(criteria))
        colorbar()
            
if __name__ == '__main__':
    path='image.png'
    rock_image=image(path,lims=[[1300,1700],[400,800]])
    my_mesh = Mesh(rock_image)
    subplots(figsize=[20,10])
    
    subplot(131)
    rock_image.plot_all()
    
    subplot(132)
    rock_image.plot_ROI(gray=True)
    print(my_mesh.list_of_blocks)
    threshold = 0.5#0.75#0.05
    
    my_mesh.adaptative_mesh(threshold)
    print(my_mesh.list_of_blocks)
    coordinate = my_mesh.plot()
    
    subplot(133)
    coordinate = my_mesh.plot()
    rock_image.plot_criteria()
    
    x_values, y_values, map_values = my_mesh.get_coordinates_and_values(rock_image.ROI_data_gray)
    
    from scipy.interpolate import *
    
    f = interp2d(x_values,y_values,map_values)
    subplots()
    x_new = np.arange(0,400)
    y_new = np.arange(0,400)
    imshow(f(x_new, y_new))