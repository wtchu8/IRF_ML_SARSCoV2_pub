#!/usr/bin/env python3

# runs ML analysis on kaggle COVID-19 vulnerability data

import os
import argparse
from pathlib import Path
import pandas as pd

import ml_classic

def mrmr_permute_compare(ml1,out_basepath,alpha=0.05):
    # Takes cleaned ml1 object and runs comparison analysis
    # Inputs
    #   ml1) ml_classic object
    #   out_basepath) output basepath

    # ----- Feature Selection -----
    # Run mrmr_permute feature selection
    if not Path(out_basepath + '_mrmr_permute.csv').is_file():
        mrmr_permute_result = ml1.fs_mrmr_permute(alpha=alpha)
        mrmr_permute_result.to_csv(out_basepath + '_mrmr_permute.csv')
        print(mrmr_permute_result)
    else:
        mrmr_permute_result = pd.read_csv(out_basepath + '_mrmr_permute.csv')
        print('Skipping mrmr permute calculation')
        print('    Loading data from:',out_basepath + '_mrmr_permute.csv')

    # Run mrmr feature selection with a matching number of features
    if not Path(out_basepath + '_mrmr.csv').is_file():
        max_rel,mrmr_result = ml1.fs_mrmr(n=len(mrmr_permute_result))
        mrmr_result.to_csv(out_basepath + '_mrmr.csv')
        print(mrmr_result)
    else:
        mrmr_result = pd.read_csv(out_basepath + '_mrmr.csv')
        print('Skipping mrmr permutation calculation')
        print('    Loading data from:',out_basepath + '_mrmr.csv')


    # ----- Model training/testing -----
    if not Path(out_basepath + '_ml_results.csv').is_file():

        # Run classification with all features
        ml1.note = ml1.note+'allVar;'
        log = ml1.ml_all_models()

        # Run classification with mrmr features
        ml_mrmr = ml_classic.ml_classic(ml1.data)
        ml_mrmr.set_label(ml1.label)
        # Get features
        if ml1.mrmr_features:
            mrmr_features = ml1.mrmr_features
        else:
            mrmr_features = pd.read_csv(out_basepath + '_mrmr.csv').Name.tolist()
        ml_mrmr.set_features(mrmr_features)
        # Run classification
        ml_mrmr.note = ml_mrmr.note+'mrmr,k='+str(len(mrmr_features))+';'
        log = log.append(ml_mrmr.ml_all_models())
        del ml_mrmr

        # Run classification with mrmr_permute features
        ml_mrmr_permute = ml_classic.ml_classic(ml1.data)
        ml_mrmr_permute.set_label(ml1.label)
        # Get features
        if ml1.mrmr_permute_features:
            mrmr_permute_features = ml1.mrmr_permute_features
        else:
            mrmr_permute_features = pd.read_csv(out_basepath + '_mrmr_permute.csv').Name.tolist()
        ml_mrmr_permute.set_features(mrmr_permute_features)
        # Run classification
        ml_mrmr_permute.note = ml_mrmr_permute.note+'mrmr_permute,k='+str(len(mrmr_permute_features))+';'
        log = log.append(ml_mrmr_permute.ml_all_models())
        del ml_mrmr_permute

        # Save results
        print(log)
        log.to_csv(out_basepath + '_ml_results.csv')
    else:
        print('Skipping ML')
        print('    Results at:',out_basepath + '_ml_results.csv')

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


