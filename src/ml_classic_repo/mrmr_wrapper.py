#!/usr/bin/env python3

# Assumes Peng's mRMR executable is in the same directory

import os
import subprocess
import argparse
import pandas as pd
import tempfile

#def __init__(self):
#    self.max_rel_df = pd.DataFrame()
#    self.mRMR_df = pd.DataFrame()
#
#def get_max_rel_df(self):
#    return self.max_rel_df
#
#def get_mRMR_df(self):
#    return self.mRMR_df

def _call_mrmr(data_csv, method, n):
    mrmr_exe_path = os.path.join(os.path.dirname(__file__),'mrmr_peng_wc')
    return subprocess.run([mrmr_exe_path,'-i', data_csv, '-m', method, '-n', str(n)],stdout=subprocess.PIPE).stdout.decode('utf-8')

def mrmr_csv(data_csv, method='MID', n=50):
    # Runs MRMR on a csv
    n = int(n)
    #if n > 5000:
    #    raise ValueError("Requested more than 5000 features, contact WC")
    
    out = _call_mrmr(data_csv, method, n)
    #out = subprocess.run([os.path.join(os.path.dirname(__file__),'mrmr_peng_wc'),
    #            '-i', data_csv, '-m', method, '-n', str(n)],
    #        stdout=subprocess.PIPE).stdout.decode('utf-8')

    #print('<<< START >>>')
    #print(out)

    save_lines = 0
    max_rel_bool = False
    mRMR_bool = False
    df_line = []
    
    for line in out.splitlines():
        #print('<<'+line+'>>'+str(save_lines)+'-'+str(max_rel_bool)+'-'+str(mRMR_bool))
        if line == '*** MaxRel features ***':
            save_lines = n + 1
            i = 0
            max_rel_bool = True
        elif line == '*** mRMR features *** ':
            save_lines = n + 1
            i = 0
            mRMR_bool = True
        elif save_lines > 0:
            # Split by tabs remove leading and trailing spaces
            line_list=line.split('\t')
            df_line.append([s.strip() for s in line_list])
            i = i + 1
            # if it's the last line save the results
            if save_lines == 1:
                if max_rel_bool == True:
                    #max_rel_df = pd.DataFrame(data=df_line[1:],columns=df_line[0]).set_index('Order')
                    max_rel_df = pd.DataFrame(data=df_line[1:],columns=df_line[0]).astype({'Score':'float'})
                    df_line=[]
                    max_rel_bool = False
                if mRMR_bool == True:
                    #mRMR_df = pd.DataFrame(data=df_line[1:],columns=df_line[0]).set_index('Order')
                    mRMR_df = pd.DataFrame(data=df_line[1:],columns=df_line[0]).astype({'Score':'float'})
                    df_line=[]
                    mRMR_bool = False
            save_lines = save_lines - 1
    
    return [max_rel_df,mRMR_df]


def mrmr(df, method='MID', n=50):
    # Callable python function for running mrmr on a dataframe
    
    # Check that the number of unique labels is = 2
    n_labels=len(df.index.unique())
    if n_labels != 2:
        print('777WARNING: Number of unique labels: '+str(n_labels))
        print('Make sure your labels are set as the index (df.set_index())')

    if n > len(df.columns):
        print('777ERROR: Max number of features: '+str(len(df.columns)))
    # Converts to a csv and calls mrmr_csv
    with tempfile.NamedTemporaryFile(suffix='.csv') as temp:
        # Important that labels are set as index here
        df.to_csv(temp.name)

        #view=pd.read_csv(temp.name)
        try:
            out_tables = mrmr_csv(temp.name, method, n)
        except:
            out = _call_mrmr(temp.name, method, n)
            print('-- Direct MRMR output --')
            print(out)

    return out_tables

def main():
    parser = argparse.ArgumentParser(description='Run mRMR')
 
    parser.add_argument('-d', '--data', required=True, \
        help='csv of input data, cols are fields, rows are entries, first col is label')

    parser.add_argument('-m', '--method', required=True, \
        help="'MID' or 'MIQ'")

    parser.add_argument('-n', '--nfeatures', required=True, \
        help='integer, number of features')
 
    args = parser.parse_args()

    data_csv = os.path.abspath(args.data)

    max_rel_df,mRMR_df = mrmr_csv(data_csv, args.method, args.nfeatures)

    print('*** MaxRel features ***')
    print(max_rel_df)

    print('*** mRMR features *** ')
    print(mRMR_df)


if __name__ == "__main__":
    main()

