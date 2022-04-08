#!/usr/bin/env python

# A system for logging the current status of the analysis

import os

def main():
    # alphabet_string = string.ascii_lowercase
    
    print('Running main')
#    ##Retreive local data path
#    #with open('paths.yaml','r') as file:
#    #    paths_list = yaml.safe_load(file)
#    #    PATH = os.path.abspath(paths_list['PATH'])
#    #my_dir=os.path.dirname(__file__)

def append_status(status,file_name)
    #Write status message, change this anytime this part of the analysis changes
    status='radio-path-imm'
    file_path=os.path.join(os.path.dirname(__file__),'analysis_status',prefix + 'merge.txt')
    with open(file_path,'w') as out_file:
        out_file.write(status)

if __name__ == "__main__":
    main()

