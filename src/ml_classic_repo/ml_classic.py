#!/usr/bin/env python3

# Useful tools for classical machine learning 

import os
import argparse
import pandas as pd

from scipy.io import loadmat
import numpy as np
from datetime import datetime

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
#from sklearn.metrics import mean_squared_error, r2_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost.sklearn import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import mrmr_permute
from . import mrmr_wrapper

class ml_classic:

    def __init__(self, data_df):
        if isinstance(data_df, str):
            # if it's a path
            print(data_df)
            self.data = self._load_data(data_df)
        else:
            self.data = data_df
        self.features = []
        self.label = ''
        self.note = ''
        self.randn = 8
        self.mrmr_features = []
        self.mrmr_permute_features = []

    # Functions to output status
    def print_data(self):
        print(self.data)

    def print_columns(self):
        i = 1
        for col in self.data.columns:
            print(str(i)+') >'+str(col)+'<')
            i = i + 1
    
    def _count_na(self):
        return self.data.isnull().sum().sum()

    def summary(self):
        print(self.data.info())
        print(self.data.describe())
        print('NA values: '+str(self._count_na()))
        print('----- Features -----')
        print(self.features)
        print('Label: '+self.label)
        if self.label:
            print(self.data.groupby(self.label).describe())

    # Helper functions
    def set_label(self,new_label):
        if new_label in self.data.columns.tolist():
            self.label = new_label
            self._reset_features()
        else:
            raise ValueError('label is not in data')

    def set_features(self,feature_list):
        self.data = self.data[feature_list+[self.label]]
        self._reset_features()

    def _reset_features(self):
        #set features based on current data and label
        self.features=self.data.columns.tolist()
        self.features.remove(self.label)

    # General analysis functions
    def znorm_single(self,feature):
        # Run z-normalization on a feature, cannot be undone
        if feature in self.features:
            self.data[feature] = (self.data[feature]-self.data[feature].mean())/self.data[feature].std()
        else:
            raise ValueError("feature '"+feature+"' is not in feature set")

    def znorm_all(self):
        # Run z-normalization on all features
        for col in self.features:
            self.znorm_single(feature=col)
        self.note = self.note + 'znorm;'

    def fs_mrmr(self,n=None):
        if n is None:
            n = len(self.features)
        out = mrmr_wrapper.mrmr(df=self.data.set_index(self.label),n=n)
        self.mrmr_features = out[1].Name.tolist()
        return out

    def fs_mrmr_permute(self,alpha=0.05):
        out = mrmr_permute.mrmr_permute(self.data, self.features, label=self.label, a_thresh=alpha)
        self.mrmr_permute_features = out.Name.tolist()
        return out

    # ---------- Binary classification ----------
    def ml_cv_kfold(self,model,k=5):

        # Make pipeline
        pipe = Pipeline([('ml_model',model)])
        #pipe = Pipeline([('z-norm',StandardScaler()), ('ml_model',model)])

        # Test-train split scheme
        gss = KFold(n_splits=k, shuffle=True, random_state=self.randn) 
        #gss = KFold(n_splits=k, train_size=0.7, random_state=randn) 

        # Run cross validation ml train/test
        X = self.data[self.features]
        y = self.data[self.label]
        CVscores = cross_validate(pipe, X, y, cv=gss.split(X), scoring=['accuracy','f1'])
        # AUC not compatible with LOOCV
        #CVscores = cross_validate(pipe, X, y, cv=gss.split(X), scoring=['accuracy','f1','roc_auc'])

        #print(CVscores)
        # Print results
        print("Cross-validation Accuracy: %0.2f (+/- %0.2f)" % (CVscores['test_accuracy'].mean(), CVscores['test_accuracy'].std() * 2))
        print("Cross-validation F1: %0.2f (+/- %0.2f)" % (CVscores['test_f1'].mean(), CVscores['test_f1'].std() * 2))
        #print("Cross-validation ROC AUC: %0.2f (+/- %0.2f)" % (CVscores['test_roc_auc'].mean(), CVscores['test_roc_auc'].std() * 2))

        return [CVscores['test_accuracy'].mean(),CVscores['test_f1'].mean()]
        #return [CVscores['test_accuracy'].mean(),CVscores['test_f1'].mean(),CVscores['test_roc_auc'].mean()]
        
    def ml_all_models(self):
        log = pd.DataFrame(columns=['Date','Model','CVaccuracy','CVf1','Note'])
        #log = pd.DataFrame(columns=['Date','Model','CVaccuracy','CVf1','CVauc','Note'])

        model_list = [LogisticRegression(),LogisticRegression(solver='liblinear', multi_class='ovr'),
             SVC(probability=True),SVC(kernel='linear', probability=True),
             DecisionTreeClassifier(),RandomForestClassifier(),
             KNeighborsClassifier(),GaussianNB(),
             LinearDiscriminantAnalysis(),XGBClassifier(use_label_encoder=False,eval_metric='logloss')
            ]
        
        # For each model type, train and evaluate the results
        for model_i in model_list:
            # Train and test model (LOOCV)
            tmp_acc,tmp_f1 = self.ml_cv_kfold(model=model_i,k=len(self.data))
            #tmp_acc,tmp_f1,tmp_auc = self.ml_cv_kfold(model=model_i,k=len(self.data))

            # Log the results
            log_tmp = pd.DataFrame(data={'Date':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         'Model':str(model_i),
                                         'CVaccuracy':tmp_acc,
                                         'CVf1':tmp_f1,
                                         'Note':self.note},index=[0])
                                         #'CVauc':tmp_auc,
            log = log.append(log_tmp,ignore_index=True)

        return log

    def learning_curve(self,feature_list):
        hold_features = self.features

        log = pd.DataFrame(columns=['Date','Model','CVaccuracy','CVf1','Note'])
        tmp_list = []
        n = 0
        model_i = LogisticRegression(solver='lbfgs')

        for f in feature_list:
            n += 1
            tmp_list.append(f)
            self.features = tmp_list
            tmp_acc,tmp_f1 = self.ml_cv_kfold(model=model_i,k=10)
            # Log the results
            log_tmp = pd.DataFrame(data={'Date':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         'Model':str(model_i),
                                         'CVaccuracy':tmp_acc,
                                         'CVf1':tmp_f1,
                                         'Note':n},index=[0])
                                         #'CVauc':tmp_auc,
            log = log.append(log_tmp,ignore_index=True)

        self.features = hold_features
        return log

    # ----------- MULTICLASS classification --------------
    def ml_cv_kfold_multiclass(self,model,k=5):

        # Make pipeline
        pipe = Pipeline([('ml_model',model)])
        #pipe = Pipeline([('z-norm',StandardScaler()), ('ml_model',model)])

        # Test-train split scheme
        gss = KFold(n_splits=k, shuffle=True, random_state=self.randn) 
        #gss = KFold(n_splits=k, train_size=0.7, random_state=randn) 

        # Run cross validation ml train/test
        X = self.data[self.features]
        y = self.data[self.label]
        #CVscores = cross_validate(pipe, X, y, cv=gss.split(X), scoring=['accuracy','f1','roc_auc'])
        CVscores = cross_validate(pipe, X, y, cv=gss.split(X), scoring=['accuracy','f1_micro'])

        #print(CVscores)
        # Print results
        print("Cross-validation Accuracy: %0.2f (+/- %0.2f)" % (CVscores['test_accuracy'].mean(), CVscores['test_accuracy'].std() * 2))
        print("Cross-validation F1: %0.2f (+/- %0.2f)" % (CVscores['test_f1_micro'].mean(), CVscores['test_f1_micro'].std() * 2))
        #print("Cross-validation F1: %0.2f (+/- %0.2f)" % (CVscores['test_f1'].mean(), CVscores['test_f1'].std() * 2))
        #print("Cross-validation ROC AUC: %0.2f (+/- %0.2f)" % (CVscores['test_roc_auc'].mean(), CVscores['test_roc_auc'].std() * 2))

        #return [CVscores['test_accuracy'].mean(),CVscores['test_f1'].mean(),CVscores['test_roc_auc'].mean()]
        return [CVscores['test_accuracy'].mean(),CVscores['test_f1_micro'].mean()]
        
    def ml_all_models_multiclass(self):
        log = pd.DataFrame(columns=['Date','Model','CVaccuracy','CVf1','Note'])
        #log = pd.DataFrame(columns=['Date','Model','CVaccuracy','CVf1','CVauc','Note'])

        model_list = [LogisticRegression(solver='lbfgs',multi_class='auto'),LogisticRegression(solver='liblinear', multi_class='ovr'),
            SVC(probability=True,gamma='scale'),SVC(kernel='linear', probability=True),
            DecisionTreeClassifier(),RandomForestClassifier(),
            KNeighborsClassifier(),GaussianNB(),
            LinearDiscriminantAnalysis(),XGBClassifier(use_label_encoder=False,eval_metric='logloss')
            ]

        #        model_list = [LogisticRegression(),LogisticRegression(solver='liblinear', multi_class='ovr'),
        #             SVC(probability=True),SVC(kernel='linear', probability=True),
        #             DecisionTreeClassifier(),RandomForestClassifier(),
        #             KNeighborsClassifier(),GaussianNB(),
        #             LinearDiscriminantAnalysis(),XGBClassifier(use_label_encoder=False,eval_metric='logloss')
        #            ]
        
        # For each model type, train and evaluate the results
        for model_i in model_list:
            # Train and test model (LOOCV)
            #tmp_acc,tmp_f1,tmp_auc = self.ml_cv_kfold_multiclass(model=model_i,k=len(self.data))
            tmp_acc,tmp_f1 = self.ml_cv_kfold_multiclass(model=model_i,k=len(self.data))

            # Log the results
            log_tmp = pd.DataFrame(data={'Date':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         'Model':str(model_i),
                                         'CVaccuracy':tmp_acc,
                                         'CVf1':tmp_f1,
                                         'Note':self.note},index=[0])
            #                             'CVauc':tmp_auc,
            log = log.append(log_tmp,ignore_index=True)

        return log

    ## Functions to extract data to a dataframe
        # Problem is there is no standard mat file format
    def _load_data(self,file_path):
        # Get data from csv or mat file
        data_file = os.path.abspath(file_path)
        data_suffix = os.path.splitext(data_file)[1]
    
        if data_suffix == '.csv':
            data_df = pd.read_csv(data_file)
    
        elif data_suffix == '.mat':
            mat = loadmat(data_file)
            # get keys that arn't header information
            data_keys = [ idx for idx in mat.keys() if idx[0] != '_']
            # Initialize dataframe
            data_df = pd.DataFrame()
            # for each key, create columns of data
            for dk in data_keys:
                if len(mat[dk][0]) > 1:
                    # If a key is associated with more than one column of data
                    #   make the column name the key name + column number
                    col_names=[ dk + str(col) for col in range(0,len(mat[dk][0])) ]
                    # Add columns to dataframe
                    data_df = pd.concat([data_df, pd.DataFrame(mat[dk], columns=col_names)], axis=1)
                else:
                    # Otherwise just use the key name
                    data_df[dk]=mat[dk]
        elif data_suffix == '.data':
            data_df = pd.read_csv(data_file,sep=',')
        else:
            print('777ERROR: invalid data file suffix: '+data_suffix)
            return 0
        return data_df

