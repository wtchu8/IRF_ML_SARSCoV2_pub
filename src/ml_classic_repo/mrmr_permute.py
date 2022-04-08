#!/usr/bin/env python3

# Script runs mrmr permutation analysis

import pandas as pd
import numpy as np
import argparse
import os
from typing import List
from time import process_time

from joblib import Parallel, delayed
import multiprocessing

from . import mrmr_wrapper

def mrmr_permute(data_znorm: "DataFrame",var_cols: List[str], label: str = 'Class', a_thresh: float = 0.05, par_bool: bool = False ) -> "DataFrame":
    # Descrete implementation, may want to do a continuous implementation later

    if not isinstance(a_thresh, float):
        raise TypeError

    #label = 'Class'
    mrmr_method = 'MID'
    #a_thresh = 0.005
    min_top_n = 25
    n_permutations = 1000
    top_n=int(n_permutations*a_thresh)

    t1_start = process_time()

    # Increase permutations depending on alpha threshold
    #   increase the number of permutations so that the top_n
    #   is greater than min_top_n
    if top_n < min_top_n:
        n_permutations = int(min_top_n/a_thresh)
        top_n=int(n_permutations*a_thresh)

    # Calculate minimum number of permutations needed for
    #   bonferonni correction to result in at least one permute
    #   min_permutations = len(var_cols)/0.05
    #   
    #   if min_permutations < 1000:
    #       n_permutations = 1000
    #   else:
    #       n_permutations = int(min_permutations + 10)

    #   a_thresh = 0.05/len(var_cols)
    #   top_n=int(n_permutations*a_thresh)
    print('alpha threshold = ' + str(a_thresh))
    print('permutations = ' + str(n_permutations))

    if top_n < 1:
        print('ERROR: alpha threshold is too high')
        return None

    # Calculate MRMR scores on data
    max_rel_df,mRMR_df = mrmr_wrapper.mrmr(data_znorm[[label]+var_cols].set_index([label]),mrmr_method, len(var_cols))
    mRMR_df = mRMR_df.set_index('Name')    

    # Permute labels and calculate MRMR scores
    null_scores = pd.DataFrame()
    data_tmp = data_znorm[[label]+var_cols].copy()
    
    def calc_null_mrmr():
        # Calls data_tmp, var_cols, and label
        # Returns dataframe with null scores        

        #print(str(i)+'.',end='')
        # Shuffle the class labels
        data_tmp[label] = np.random.permutation(data_tmp[label].values)
        #print(data_tmp[['Class','VoxelVolume']])
        # Calculate MRMR
        mrmr_tmp = mrmr_wrapper.mrmr(data_tmp.set_index(label),mrmr_method, len(var_cols))[1]
        # Save mrmr scores to table
        #print(mrmr_tmp.set_index('Name'))
        return mrmr_tmp[['Name','Score']].set_index('Name')['Score']

    permute_cols=range(0,n_permutations)
    if par_bool:
        # Using multiprocessing
        raise ValueError('Parallel processing is not working yet')
        #pool_obj = multiprocessing.Pool()
        #pool_obj.map(calc_null_mrmr,permute_cols)

        num_cores = multiprocessing.cpu_count()
        
    else:
        for i_tmp in permute_cols:
            null_scores[i_tmp] = calc_null_mrmr()

    def calc_thresh(row):
        # Sort values, take the top n and output the min of the top n
        return null_scores[permute_cols].transpose()[row.name].sort_values(ascending=False).head(top_n).min()
    
    # Calculate threshold for each row (field)
    mRMR_df['Score_thresh'] = null_scores.apply(calc_thresh, axis=1)

    def threshold_score(row):
        if float(row.Score) > float(row.Score_thresh):
            return row.Score
        else:
            return None
    
    # Apply threshold
    mRMR_df['corr_Score'] = mRMR_df.apply(threshold_score, axis=1)
    
    t1_stop = process_time()

    print('Done (',t1_stop-t1_start,'min )')
    return mRMR_df.reset_index().dropna()[['Name','Score_thresh','Score']]

def main():
    parser = argparse.ArgumentParser(description='Run mRMR')
 
    parser.add_argument('-d', '--data', required=True, \
        help="csv of input data, cols are fields, rows are entries, label col is called 'Class'")

    parser.add_argument('-o', '--output', required=True, \
        help="output csv of mRMR_permute feature table")

    parser.add_argument('-l', '--label', default='Class',\
        help="specify the name of the label column, default: 'Class'")

    parser.add_argument('-a', '--alpha', default=0.05,\
        help='p-value threshold')
 
    #   parser.add_argument('-m', '--method', required=True, \
    #       help="'MID' or 'MIQ'")

    #   parser.add_argument('-n', '--nfeatures', required=True, \
    #       help='integer, number of features')
 
    args = parser.parse_args()

    data_csv = os.path.abspath(args.data)

    df = pd.read_csv(data_csv)

    if not args.label in df.columns.tolist():
        print("777ERROR: No column called:" + args.label)
        return 1

    # Calculate mrmr permute
    out_df = mrmr_permute(df,df.loc[:,df.columns != args.label].columns.tolist(), label=args.label, a_thresh=float(args.alpha))

    # Send output to a CSV
    out_df.set_index('Name').to_csv(os.path.abspath(args.output))
    print('Results written to:' + os.path.abspath(args.output) )
    print(out_df)
    return 0

if __name__ == "__main__":
    main()

