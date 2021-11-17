# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:57:49 2021

@author: nunoa
"""


import os
from matplotlib.pyplot import *
import numpy as np
from skimage import io, filters



class Block:
    
    """
    Instance is a block of a given mesh
    """
    
    def __init__(self, coordinates,level=0):
        
        self.coordinates = np.array(coordinates)
        self.is_checked = False
        self.level = level
        
        ##connections to plot lines of reference
        x0 = coordinates[0][0]
        x1 = coordinates[3][0]
        y0 = coordinates[0][1]
        y1 = coordinates[1][1]
        self.block_connections = [[x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0]]
    
    def check_block(self,threshold,image):
        
        x_min = int(self.coordinates[0][0])
        x_max = int(self.coordinates[3][0])
                                        
        y_min = int(self.coordinates[0][1])
        y_max = int(self.coordinates[1][1])
                    
        criteria = image.compute_criteria([[x_min,x_max],
                                           [y_min,y_max]])
        
        value = np.mean(criteria)
        
        self.is_checked = True
        
        value>threshold
        
        return value>threshold
    
    def __repr__(self):
        return "Block \n"+str(self.coordinates)+",\n level " + str(self.level)
    

class Mesh:
    
    """
    A mesh is a set of Blocks
    """
    
    def __init__(self, image):
        
        self.list_of_blocks = []
        
        self.list_of_blocks.append(Block([[image.x_min, image.y_min],
                                    [image.x_min, image.y_max],
                                    [image.x_max, image.y_max],
                                    [image.x_max, image.y_min]],level=0))
        
        self.image = image
        self.coordinate_pairs=[]
        
    def update_blocks(self,threshold):
        
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
        
    def adaptative_mesh(self,threshold, max_num_divisions=3):
        self.state = 1
        self.max_level = 0
        while self.max_level <= max_num_divisions and self.state == 1:
            print('aqui')

                self.update_blocks(threshold)
            
        
        
    def plot(self):
            
            levels = []
            #print(self.list_of_blocks)
            for i in range(0,len(self.list_of_blocks)):
                plot(self.list_of_blocks[i].block_connections[0],
                     self.list_of_blocks[i].block_connections[1], lw = .5, color= 'w',ls='--')
                
                for j in range(0,len(self.list_of_blocks[i].coordinates)):
                    coordinate_pairs.append(self.list_of_blocks[i].coordinates[j])
                    levels.append(self.list_of_blocks[i].level)
            
            self.coordinate_pairs = np.array(self.coordinate_pairs)
            cycle_colors = rcParams['axes.prop_cycle'].by_key()['color']
            
            for i in range(0,len(self.coordinate_pairs)):

                plot(self.coordinate_pairs[i,0],self.coordinate_pairs[i,1],
                         'o',ms=7,markeredgecolor='k', color=cycle_colors[levels[i]])
        
            return self.coordinate_pairs
        

        
      
        
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
            
        else:
            criteria = filters.sobel(self.ROI_data_gray)
        
        return criteria
        
    def plot_criteria(self):
        title("Criteria")
        criteria =filters.sobel(self.ROI_data_gray)
        imshow(np.transpose(criteria))
        colorbar()


path=r'C:/Users/Diana/Desktop/INESC TEC 2021/Adaptative_Meshing/moscov_ini.jpg'
rock_image=image(path,lims=[[1500,1700],[400,800]])
my_mesh = Mesh(rock_image)

subplots(figsize=[20,10])
subplot(131)
rock_image.plot_all()
subplot(132)
rock_image.plot_ROI(gray=True)
print(my_mesh.list_of_blocks)
threshold = .025

my_mesh.adaptative_mesh(threshold)


#print(my_mesh.list_of_blocks)
coordinate = my_mesh.plot()
print(coordinate)

subplot(133)

coordinate = my_mesh.plot()
rock_image.plot_criteria()

        