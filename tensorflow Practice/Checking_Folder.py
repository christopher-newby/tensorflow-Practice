'''
Created on Jul 15, 2017

This file checks to see if the directory where I store data exists

@author: Chris Newby
'''


import os


def Folder_Check(data_dir = 'Data'):
    '''
    Checks whether data_dir is a folder that exists in the current working space
    if it does, then this program does nothing,
    if it doesn't then it makes the directory
    '''
    try:
        if(type(data_filename) != str):
            raise ValueError("Not a valid file name!")

        # this sets up the test_data folder
        if data_filename not in os.listdir():
            os.mkdir(data_filename)
            
        print('Done with files setup.\n')
    except ValueError as ve:
        print(ve)