'''
Created on Jul 5, 2017

This file contains methods for creating and retrievign the data sample
I use for practicing with tensorflow.

The data describes a circle that is classified differently than other data points.

@author: Chris Newby
'''

import random   # for the random functions
random.seed()   # no arg is current time


def CreateData(data_dir, data_filename, n_max_trials):
    
    try:
        if(type(data_filename) != str or type(data_dir) != str):
            raise ValueError("Not a valid file name!")
        if(type(n_max_trials) != int):
            raise ValueError("Non-integer trials value!")

        n_trials = 0
        
        square = lambda x : x**2
        
        rad_inner = .4   # max radius of inner circle
        rad_outter = .5  # min radius of outer circle
        max_range = 1  # max range of x and y parameters
        
        with open(data_dir + '/' + data_filename, 'w') as f:
            while n_max_trials > n_trials :
                x = random.uniform(-max_range,max_range)
                y = random.uniform(-max_range,max_range)
                
                xsq = square(x)
                ysq = square(y)
                insq = square(rad_inner)
                outsq = square(rad_outter)
                
                str_output = ''
                send_output = False
                
                if (xsq + ysq <= insq) | (xsq + ysq >= outsq) :
                    str_output += str(x) + '\t' + str(y) + '\t'     # tab spacing for gnuplot
                    send_output = True
                
                if xsq + ysq <= insq :
                    str_output += str(1) + '\t' + str(0)
                elif xsq + ysq >= outsq :
                    str_output += str(0) + '\t' + str(1)
                    
                if send_output :
                    f.write(str_output + '\n')
                    n_trials += 1
            
        print("Done creating the sample data file.\n")
    except ValueError as ve:
        print(ve)
        

def Getting_Data(data_dir, data_filename):   
    with open(data_dir + '/' + data_filename, 'r') as f:
        
        input_data = []
        output_data = []
        
        for line in f:
            record = [float(field) for field in line.split()]
            
            # this way is just the data
            xh = record[0]
            yh = record[1]
            
            #this way squares the inputs
            #xh = record[0]**2
            #yh = record[1]**2
            
            input_data.append([xh,yh])
            output_data.append([record[2],record[3]])
            
        print('Done retrieving sample data.\n')
        return input_data,output_data
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            