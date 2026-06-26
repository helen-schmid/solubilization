'''
Common utilities for protien design/dry lab work

Not associated with any particular software

Minimal functions
'''

import os
import csv
import glob
import shutil
from csv import DictWriter

def make_directory(directory:str):
    '''
    Function to make a directory if it does not already exist

    directory - path to the directory to make

    no returns
    '''
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f'Made directory: {directory}')


def initialise_checkpoint_file(file):

    '''
    Initialise a simple text file for checkpointing progress

    Checks whether the file exists, if not, creates a new checkpoint file

    no returns
    '''
        
    if os.path.isfile(file) == True:
        print('Checkpoint file found')

    elif os.path.isfile(file) == False:
        print('No checkpoint file found, making checkpoint file')
        f = open(file, 'w', newline='', encoding='utf-8')
        f.write('Completed:\n')
        f.close()

def initialise_csv_file(file:str, columns:list):

    '''
    initialise a CSV file to write scores to progressively.

    Writes the columns as headers for the data

    Columns must be a list of headers that matches the data to be input

    if the file already exists, takes no actions
    '''
      
    #setting up a csv file for the scores
    try:
        csvfile = open(file, 'x', newline='', encoding='utf-8')
        c = csv.writer(csvfile)
        c.writerow(columns)
        csvfile.close()
    except FileExistsError:
        print(f'The csv file {file} already exists, appending new scores to file')
        pass

def write_to_csv(file:str, columns:list, dictionary:dict):
    '''
    appends a dictionary of scores to an existing csv file

    the file must already exist, and be initialised with a list of headers (columns)

    write to the file using the dictionary

    '''
    with open(file, 'a') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=columns)
        dictwriter_object.writerow(dictionary)
        f_object.close()

def update_checkpoint_file(file, id):
    '''
    add a line to the checkpoint file

    id - the file name or sample id to be tracked for checkpointing
    '''
    f = open(file, 'a', newline='', encoding='utf-8')
    f.write(id+'\n')
    f.close()