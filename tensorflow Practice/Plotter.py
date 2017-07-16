'''
Created on Jul 13, 2017

This file contains methods for plotting a single, classified set of data for this practice example
and for plotting two sets in the same window

@author: Chris Newby
'''

from matplotlib import pyplot as plt

def Plot_Data(dat_in, dat_class, title = 'Sample Data'):
    '''
    Function for plotting the data
    specific to the data I'm generating in Making_Data
    '''
    x_list = []
    y_list = []
    color_list = []
    
    for elem in dat_in:
        x_list.append(elem[0])
        y_list.append(elem[1])
        
    for elem in dat_class:
        if elem[0] == 1:
            color_list.append("r")
        else:
            color_list.append('b')
    
    plt.scatter(x = x_list, y = y_list, c=color_list)
    plt.xlabel('x')
    plt.xlim([-1,1])
    plt.ylabel('y')
    plt.ylim([-1,1])
    plt.title(title)
    plt.axes().set_aspect('equal')
    plt.show()


def Plot_two_data(dat_1,class_1,dat_2,class_2, name_1 = 'Sample Data',name_2 = 'Classified Data'):
    '''
    Function for plotting two sets of data together
    '''
    
    #setting up first data set
    x1_list = []
    y1_list = []
    color1_list = []
    
    for elem in dat_1:
        x1_list.append(elem[0])
        y1_list.append(elem[1])
        
    for elem in class_1:
        if elem[0] == 1:
            color1_list.append("r")
        else:
            color1_list.append('b')
            
    #setting up second data set
    x2_list = []
    y2_list = []
    color2_list = []
    
    for elem in dat_2:
        x2_list.append(elem[0])
        y2_list.append(elem[1])
        
    for elem in class_2:
        if elem[0] == 1:
            color2_list.append("r")
        else:
            color2_list.append('b')
    
    plt.figure(1)
    plt.subplot(121, aspect = 'equal')        
    plt.scatter(x = x1_list, y = y1_list, c=color1_list)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title(name_1)
    
    plt.subplot(122, aspect = 'equal')        
    plt.scatter(x = x2_list, y = y2_list, c=color2_list)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title(name_2)
    
    plt.show()


