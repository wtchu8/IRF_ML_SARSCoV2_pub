# This script is currently not functional

def assess_MLmodel(model,X,y,note_ml='tmp',title_plot="",export_plot=False):
    
    print('----- %s -----' % (model))
    var_cols_tmp = X.columns[X.columns != 'Subject']
    
    # ---- split into train/test sets ----
    # Traditional train/test split
    #trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=8)
    # Grouped train/test split (keep from splitting subjects across train/test groups)
    gss = GroupShuffleSplit(n_splits=2, train_size=0.7, random_state=randn)
    #gss = GroupShuffleSplit(n_splits=2, train_size=0.7)
    train_ix, test_ix = next(gss.split(X, y, groups=X.Subject))
    trainX = X.iloc[train_ix][var_cols_tmp]
    testX = X.iloc[test_ix][var_cols_tmp]
    trainy = y.iloc[train_ix].Class.astype(int)
    testy = y.iloc[test_ix].Class.astype(int)
    
    print(X.iloc[train_ix].Subject.unique())
    print(X.iloc[test_ix].Subject.unique())
    print('Training samples:',len(trainX), '(', trainy.to_numpy().sum(),' infected)')
    print('Testing sample:',len(testX), '(', testy.to_numpy().sum(),' infected)')
    print('Unique training subjects:',len(set(X.iloc[train_ix].Subject)))
    print('Unique testing subjects:',len(set(X.iloc[test_ix].Subject)))
    print('Features:',len(trainX.columns))
    # Remove the subject ID for later use 
    X = X[var_cols_tmp]
    y = y.Class.astype(int)
    
    # ---- Train Machine Learner ----
    # Make pipeline (to add scaling before training and testing)
    #    Scale too mean of 0 and STD of 1 then apply the model
    #    Scalling is essential for calculating feature importance
    pipe = Pipeline([('z-norm',StandardScaler()), ('ml_model',model)])
    
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]
    
    # fit a model
    #model = LogisticRegression(solver='lbfgs')
    fitted_model = pipe.fit(trainX, trainy)
    
    # predict probabilities
    lr_probs = pipe.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print('ROC AUC=%.3f' % (lr_auc))
    
    # ---- Cross-validation ----
    folds=5
    CVscores = cross_validate(pipe, X, y, cv=folds, scoring=['accuracy','f1','roc_auc'])
        
    print("Cross-validation Accuracy: %0.2f (+/- %0.2f)" % (CVscores['test_accuracy'].mean(), CVscores['test_accuracy'].std() * 2))
    print("Cross-validation F1: %0.2f (+/- %0.2f)" % (CVscores['test_f1'].mean(), CVscores['test_f1'].std() * 2))
    print("Cross-validation ROC AUC: %0.2f (+/- %0.2f)" % (CVscores['test_roc_auc'].mean(), CVscores['test_roc_auc'].std() * 2))
    # run with shuffle
    #     cv = ShuffleSplit(n_splits=folds, test_size=0.2, random_state=7)
    #     SSCVscores = cross_val_score(pipe, X, y, cv=cv)
    #     print("Shuffle CV Accuracy: %0.2f (+/- %0.2f)" % (SSCVscores.mean(), SSCVscores.std() * 2))
    
    # ---- Save performance ----
    log = pd.DataFrame(data={'Date':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             'Model':str(model),
                             'AUC':lr_auc,
                             'CVaccuracy':CVscores['test_accuracy'].mean(),
                             'CVf1':CVscores['test_f1'].mean(),
                             'Note':note_ml},index=[0])
    
    # ---- Calculate/plot roc curves ----
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    
    # plot the roc curve for the model
    fig = plt.figure()
    axs = fig.add_subplot()
    axs.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    axs.plot(lr_fpr, lr_tpr, marker='.', label='Model')
    # axis labels
    axs.set_xlabel('False Positive Rate')
    axs.set_ylabel('True Positive Rate')
    if title_plot:
        axs.set_title(title_plot)
    # show the legend
    axs.legend()
    # save the plot
        # As written, only the last of each model type is saved (ex. one logistic regression plot)
    if export_plot == True:
        fig.savefig(os.path.join(PATH,'figures','Output_mlplot_'+re.split('\(|\)',type(model).__name__)[0]+'.png'),dpi=600,bbox_inches='tight')
    # show the plot
    plt.show()
    
    # ---- Output feature ranking ----
    #print(abs(fitted_model['ml_model'].coef_[0]))

    feature_imp = pd.DataFrame({'feature':X.columns.tolist()})

    try:
        feature_imp['score_val'] = abs(fitted_model['ml_model'].coef_[0])
        print('Ranking based on coefficient')
    except:
        print('Not a valid model for feature ranking based on coefficient')

        try:
            feature_imp['score_val'] = fitted_model['ml_model'].feature_importances_
            print('Ranking based on feature importance')
        except:
            print('Not a valid model for feature ranking based on feature importance')
            return [log,None]

    feature_imp = feature_imp.sort_values(by='score_val',ascending=False).round(2).reset_index().drop('index',axis='columns')

    #     # Calculate a score threshold for more than 34 features
    #     n_feature_count=34
    #     if len(X.columns) > n_feature_count:
    #         score_val_thresh = feature_imp.score_val.iloc[n_feature_count]
    #         feature_imp = feature_imp.loc[feature_imp.score_val >= score_val_thresh]
    #         print('Score threshold for n =',n_feature_count,':',score_val_thresh)
    return [log,feature_imp]


