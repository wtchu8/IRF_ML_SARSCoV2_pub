# Functions used by multiple jupyter notebooks

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

def clean_log(log, label_list, note_list, metric='CVaccuracy', ascending=None, transpose=False):
    # clean up log file for plotting
    # set ascending to None to not rank the results

    model_name_dict={"SVC(probability=True)":'SVM',
                       'LogisticRegression()':'Logistic regression',
                       'DecisionTreeClassifier()':'Decision tree',
                       'RandomForestClassifier()':'Random forest',
                       'KNeighborsClassifier()':'KNN',
                       'GaussianNB()':'Gaussian NB',
                       'LinearDiscriminantAnalysis()':'LDA',
                       'XGBClassifier()':'XGBoost'}
    
    if len(label_list) != len(note_list):
        raise ValueError('777ERROR: '+str(len(label_list))+' labels but '+str(len(note_list))+' notes')
    else:
        n = len(label_list)
        
    if log.duplicated(subset=['Note','Model']).any():
        raise ValueError('777ERROR: duplicate Model and Note combinations')

    tmp_log = log.loc[log['Note'].isin(note_list) ,['Model','Note',metric]]

    # Rename columns to something more friendly
    if 'model_name_dict' in locals():

        tmp_log = tmp_log.replace(to_replace=re.compile('^XGBClassifier.*', flags=re.DOTALL), value='XGBClassifier()', regex=True)
        tmp_log = tmp_log.replace(model_name_dict)

        # remove columns not in dictionary
        tmp_log = tmp_log[tmp_log.Model.isin(model_name_dict.values())]
        #print('Ignoring models:',tmp_log.loc[~tmp_log.Model.isin(model_name_dict.values()),'Model'].unique().tolist())

    # Replace note with label
    for i in range(n):
        tmp_log.loc[tmp_log.Note == note_list[i],'Note'] = label_list[i]

    # Get original model order
    if 'model_name_dict' in locals():
        # default ordering according to dictionary
        model_list = list(model_name_dict.values())
    else:
        # default ordering according to log
        model_list = set([ m[:30] for m in tmp_log.Model.tolist() ])

    # Reorganize dataframe for plotting
    tmp_log = tmp_log.set_index(['Model','Note']).unstack()[metric]
    tmp_log.columns.name = None

    # Shorten long model names
    tmp_log.index = tmp_log.index.str.slice(0,30)

    # Sort models (rows)
    if ascending != None:
        # Re-arrange the models to order by max performance
        tmp_log['sort'] = tmp_log.max(axis = 1, numeric_only = True)
        tmp_log = tmp_log.sort_values(by = 'sort', ascending=ascending)
        tmp_log = tmp_log.drop(columns='sort')
        

    # Get the transpose (columns = models, rows = FS)
    if transpose:
        #model_list = tmp_log.index.tolist()
        tmp_log = tmp_log.transpose()

        # Sort feature selection methods
        if ascending != None:
            # Re-arrange the transposed indicies to order by average performance
            tmp_log['sort'] = tmp_log.max(axis = 1, numeric_only = True)
            tmp_log = tmp_log.sort_values(by = 'sort', ascending=ascending)
            tmp_log = tmp_log.drop(columns='sort')
        else:
            # Set the FS order to match the input label list
            tmp_log = tmp_log.reindex(label_list)
            # Set the model order to match the original model
            tmp_log = tmp_log[model_list]

    return tmp_log

def plot_mlperformance(log, label_list, note_list, title_prefix='', metric='CVaccuracy'):
    # Create bar plot comparing different models

    # Clean up log file for plotting
    tmp_log = clean_log(log, label_list, note_list, metric)

    # Make the plot title
    title = title_prefix + label_list[0]
    for label in label_list[1:]:
        title = title + ' vs. ' + label
    tag = 'ML_comparison'+title.replace(' ','_')

    # If there are many series, only use a subset of the total models
    if len(label_list) > 4 and len(tmp_log) > len(label_list):
        print('Trincating plot to top ' + str(len(tmp_log)- len(label_list)) + ' models only')
        tmp_log = tmp_log.tail(len(tmp_log)- len(label_list))

    # Plot results
    #fig = plt.figure(facecolor='white', figsize=[6.4,4.8*(len(label_list)-2)], dpi=100)
    fig = plt.figure(facecolor='white', dpi=100)
    axs = fig.add_subplot()

    tmp_log.plot.barh(y=label_list, ax=axs)
    
    axs.set_axisbelow(True)
    axs.xaxis.grid()
    axs.set_title(title)

    axs.set_xlim([0.5,1])
    axs.set_xlabel(metric)
    axs.legend(loc="center", ncol=len(label_list),bbox_to_anchor=(0.5, -0.2))
    #fig.savefig(os.path.join(PATH,'figures',tag+'.png'), bbox_inches="tight",dpi=300)
    #plt.show()
    plt.close(fig)
    return fig,tag

def plot_ml_matrix(log, label_list, note_list, title='', metric='CVaccuracy', ascending=None):
    # Display ML result in a FSxModel matrix
    tmp_log = clean_log(log, label_list, note_list, metric, ascending=ascending, transpose=True)

    # Filter and clean up labels
    tmp_log

    # Plot results
    fig = plt.figure(facecolor='white', dpi=100)
    axs = fig.add_subplot()
    #display(tmp_log)
    sns.heatmap(tmp_log, annot=tmp_log.to_numpy(), vmin=0.4, vmax=1,cmap='magma')
    #sns.heatmap(tmp_log, annot=tmp_log.to_numpy(),cmap='magma')
    axs.set_ylabel('Feature selection method [number of features]')
    axs.set_xlabel('Model')
    plt.close(fig)
    return fig

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

def plot_comparison(log, title_prefix, labelA, noteA, labelB, noteB, metric='CVaccuracy'):
    # DEPRICATED, REMAINS HERE FOR LEGACY COMPATIBILITY    

    #   model_name_dict={"SVC(kernel='linear', probability=True)":'SVM',
    #                       'LogisticRegression()':'Logistic Regression',
    #                       'DecisionTreeClassifier()':'Decision Tree',
    #                       'RandomForestClassifier()':'Random Forest',
    #                       'KNeighborsClassifier()':'KNN',
    #                       'GaussianNB()':'Gaussian Naive Bayes',
    #                       'LinearDiscriminantAnalysis()':'LDA',
    #                       'XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,\
    #             colsample_bynode=None, colsample_bytree=None,\
    #             eval_metric=\'logloss\', gamma=None, gpu_id=None,\
    #             importance_type=\'gain\', interaction_constraints=None,\
    #             learning_rate=None, max_delta_step=None, max_depth=None,\
    #             min_child_weight=None, missing=nan, monotone_constraints=None,\
    #             n_estimators=100, n_jobs=None, num_parallel_tree=None,\
    #             random_state=None, reg_alpha=None, reg_lambda=None,\
    #             scale_pos_weight=None, subsample=None, tree_method=None,\
    #             use_label_encoder=False, validate_parameters=None,\
    #             verbosity=None)':'XGBoost'}
    label_list=[labelA,labelB]
    note_list=[noteA,noteB]

    title = title_prefix + ' ' + label_list[0]
    for label in label_list[1:]:
        title = title + ' vs. ' + label
    tag = 'ML_comparison'+title.replace(' ','_')

    if log.duplicated(subset=['Note','Model']).any():
        raise ValueError('777ERROR: duplicate Model and Note combinations')
      
    tmp_log = log.loc[log['Note'].isin([noteA,noteB]) ,['Model','Note',metric]]
    
    # Rename columns to something more friendly
    if 'model_name_dict' in locals():
        tmp_log = tmp_log.replace(model_name_dict)

    # Replace note with label
    tmp_log.loc[tmp_log.Note == noteA,'Note'] = labelA
    tmp_log.loc[tmp_log.Note == noteB,'Note'] = labelB
    
    # Reorganize dataframe for plotting
    tmp_log = tmp_log.set_index(['Model','Note']).unstack()[metric]
    tmp_log.columns.name = None

    #display(tmp_log)

    # Shorten long model names
    tmp_log.index = tmp_log.index.str.slice(0,30)
  
    # Plot results
    fig = plt.figure(facecolor='white')
    axs = fig.add_subplot()

    tmp_log.plot.barh(y=[labelA,labelB], ax=axs)
    
    axs.set_axisbelow(True)
    axs.xaxis.grid()
    axs.set_title(title)

    axs.set_xlim([0.5,1])
    axs.set_xlabel(metric)
    #fig.savefig(os.path.join(PATH,'figures',tag+'.png'), bbox_inches="tight",dpi=300)
    #plt.show()
    plt.close(fig)
    return fig,tag

