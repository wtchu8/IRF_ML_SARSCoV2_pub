#!/usr/bin/env python3

# runs ML analysis on kaggle COVID-19 vulnerability data

import os
import argparse
from pathlib import Path
import pandas as pd

import ml_classic

def mrmr_learning_curve(ml1,out_basepath):
    # Takes cleaned ml1 object and runs comparison analysis
    # Inputs
    #   ml1) ml_classic object
    #   out_basepath) output basepath

    # ----- Feature Selection -----
    # Run mrmr feature selection with a matching number of features
    if not Path(out_basepath + '_mrmr.csv').is_file():
        max_rel,mrmr_result = ml1.fs_mrmr(n=len(ml1.features))
        mrmr_result.to_csv(out_basepath + '_mrmr.csv')
        print(mrmr_result)
    else:
        mrmr_result = pd.read_csv(out_basepath + '_mrmr.csv')
        print('Skipping mrmr permutation calculation')
        print('    Loading data from:',out_basepath + '_mrmr.csv')


    # ----- Model training/testing -----
    if not Path(out_basepath + '_learning_curve.csv').is_file():
        
        log = ml1.learning_curve(mrmr_result.Name.tolist())

        # Save results
        print(log)
        log.to_csv(out_basepath + '_learning_curve.csv')
    else:
        print('Skipping ML')
        print('    Results at:',out_basepath + '_learning_curve.csv')

    #parser = argparse.ArgumentParser(description='Feature Selection Pipeline')
 
    #parser.add_argument('-d', '--data', required=True, \
    #    help='.csv or .mat of input data, cols are fields, rows are entries')

    #parser.add_argument('-o', '--output', required=True, \
    #    help="output basepath")

    #parser.add_argument('-l', '--label', default='Class', \
    #    help="output directory")

    #args = parser.parse_args()
    #data_file = args.data
    #out_basepath = args.output
    #in_label = args.label

    #data_file = "/hpcdata/irf-image/chuwt/classification_data/peng_data/test_lung_s3.csv"
    #out_basepath = "/hpcdata/irf-image/chuwt/classification_data/peng_data/output2/lung"
    #in_label = "class"


    # Get data
    #data_df = ml_classic.ml_classic(data_file)

    # Clean data
    #data_df = clean(data_df)




#if __name__ == "__main__":
#    main()


