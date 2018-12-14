import numpy as np, scipy.stats as st,pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV




'''
Functions for data preprocessing
'''
def subsampling(data,composers):
    listOfComposers = composers.copy()
    datamin = data[data["composer"].isin(listOfComposers)]["composer"].value_counts().min()
    composermin = data[data["composer"].isin(listOfComposers)]["composer"].value_counts().idxmin()
    listOfComposers.remove(composermin)
    data_selected = data[data["composer"]==composermin]
    for c in listOfComposers:
        data_c = data[data["composer"]==c]
        data_c = data_c.sample(datamin)
        data_selected = pd.concat([data_selected,data_c])
    return data_selected

def splitDataset(data): 
    data = data.drop("song_name",axis=1)
    X = data.drop("composer",axis=1)
    y = data["composer"]

    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,stratify = y)
    print(y_test.shape)
    
    return X_train,X_test,y_train,y_test







def logisticRegression(data,composer_list,subsample=True,normalisation=True):

    
    if subsample : 
        X_train,X_test,y_train,y_test = splitDataset(subsampling(data,composer_list))
    else :
        X_train,X_test,y_train,y_test = splitDataset(data[data["composer"].isin(composer_list)])
        
    if normalisation: 
        pipe = Pipeline(steps = [
        ('scale', StandardScaler()),
        ('regLog', LogisticRegression())])
    else :
         pipe = Pipeline(steps= [
        ('regLog', LogisticRegression())])
            

    hyperparameters = {
        'regLog__penalty' : ['l1', 'l2'],
        'regLog__C' : np.logspace(-3, 5, 20)
    }
    
    
    gcLogReg = GridSearchCV(estimator=pipe,param_grid=hyperparameters,n_jobs=-1)
    gcLogReg.fit(X_train, y_train)
    
    predictions = np.round(gcLogReg.predict(X_test))
    acc = accuracy_score(y_test,predictions)
    return gcLogReg.best_estimator_,acc

def svmRBF(data,composer_list,subsample=True,normalisation=True):


    if subsample : 
        X_train,X_test,y_train,y_test = splitDataset(subsampling(data,composer_list))
    else :
        X_train,X_test,y_train,y_test = splitDataset(data[data["composer"].isin(composer_list)])

    if normalisation: 
        pipe = Pipeline(steps = [
        ('scale', StandardScaler()),
        ('svm', SVC('rbf'))])
    else :
         pipe = Pipeline(steps= [
        ('svm', SVC('rbf'))])


    hyperparameters = {
        'svm__gamma' : np.logspace(-5, 5, 200),
        'svm__C' : np.logspace(-5, 5, 200)
    }


    rscvSVM = RandomizedSearchCV(estimator=pipe,param_distributions=hyperparameters,cv=3,n_iter=100,n_jobs=-1)
    rscvSVM.fit(X_train, y_train)
    randomParameters = rscvSVM.best_params_
    
    hyperparameters = {
        'svm__gamma' : np.linspace(randomParameters.get('svm__gamma')*0.9, randomParameters.get('svm__gamma')*1.1, 50),
        'svm__C' : np.linspace(randomParameters.get('svm__C')*0.9, randomParameters.get('svm__C')*1.1, 50)
    }
    gcSVM = GridSearchCV(estimator=pipe, param_grid=hyperparameters,n_jobs=-1)
    gcSVM.fit(X_train,y_train)

    predictions = np.round(gcSVM.predict(X_test))
    acc = accuracy_score(y_test,predictions)
    return gcSVM.best_estimator_,acc

def randomForest(data,composer_list,subsample=True,normalisation=True):

    
    if subsample : 
        X_train,X_test,y_train,y_test = splitDataset(subsampling(data,composer_list))
    else :
        X_train,X_test,y_train,y_test = splitDataset(data[data["composer"].isin(composer_list)])
    #print(X_train)
    #print("Normalisation") 
    
    if normalisation: 
        pipe = Pipeline(steps = [
        ('scale', StandardScaler()),
        ('rdmf', RandomForestClassifier('rbf'))])
    else :
         pipe = Pipeline(steps= [
        ('rdmf', RandomForestClassifier('rbf'))])
            
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    hyperparameters = {'rdmf__n_estimators': n_estimators,
                   'rdmf__max_features': max_features,
                   'rdmf__max_depth': max_depth,
                   'rdmf__min_samples_split': min_samples_split,
                   'rdmf__min_samples_leaf': min_samples_leaf,
                   'rdmf__bootstrap': bootstrap}
    
    
    
    rscvRdmF = RandomizedSearchCV(estimator = pipe, param_distributions = hyperparameters, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rscvRdmF.fit(X_train, y_train)
    
    predictions = np.round(rscvRdmF.predict(X_test))
    acc = accuracy_score(y_test,predictions)
    return rscvRdmF.best_estimator_,acc

def randomForestRandomizedGrid(data,composer_list,subsample=True,normalisation=True):

    
    if subsample : 
        X_train,X_test,y_train,y_test = splitDataset(subsampling(data,composer_list))
    else :
        X_train,X_test,y_train,y_test = splitDataset(data[data["composer"].isin(composer_list)])
    #print(X_train)
    #print("Normalisation") 
    
    if normalisation: 
        pipe = Pipeline(steps = [
        ('scale', StandardScaler()),
        ('rdmf', RandomForestClassifier('rbf'))])
    else :
         pipe = Pipeline(steps= [
        ('rdmf', RandomForestClassifier('rbf'))])
            
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    hyperparameters = {'rdmf__n_estimators': n_estimators,
                   'rdmf__max_features': max_features,
                   'rdmf__max_depth': max_depth,
                   'rdmf__min_samples_split': min_samples_split,
                   'rdmf__min_samples_leaf': min_samples_leaf,
                   'rdmf__bootstrap': bootstrap}
    
    
    
    rscvRdmF = RandomizedSearchCV(estimator = pipe, param_distributions = hyperparameters, n_iter = 1, cv = 3, verbose=2, n_jobs = -1)
    rscvRdmF.fit(X_train, y_train)
    
    
    randomParameters = rscvRdmF.best_params_

    n_estimators = [int(x) for x in np.linspace(randomParameters.get('rdmf__n_estimators')*0.9, randomParameters.get('rdmf__n_estimators')*1.1,num= 5)]
    max_depth = np.linspace(randomParameters.get('rdmf__max_depth')*0.9, randomParameters.get('rdmf__max_depth')*1.1, num = 5)
    
    min_samples_leaf = randomParameters.get('rdmf__min_samples_leaf')
    min_samples_split =  randomParameters.get('rdmf__min_samples_split')
    bootstrap = randomParameters.get("rdmf__bootstrap")
    max_features = randomParameters.get('rdmf__max_features')
    
    
    hyperparameters = {'rdmf__n_estimators': n_estimators,
                   'rdmf__max_depth': max_depth}
    
    if normalisation: 
        pipe = Pipeline(steps = [
        ('scale', StandardScaler()),
        ('rdmf', RandomForestClassifier('rbf',min_samples_leaf=min_samples_leaf,bootstrap=bootstrap,min_samples_split=min_samples_split,max_features=max_features))])
    else :
         pipe = Pipeline(steps= [
        ('rdmf', RandomForestClassifier('rbf',min_samples_leaf=min_samples_leaf,bootstrap=bootstrap,min_samples_split=min_samples_split,max_features=max_features))])
    
    gcRdmF = GridSearchCV(estimator=pipe, param_grid=hyperparameters,n_jobs=-1)
    gcRdmF.fit(X_train,y_train)
    
    
    
    predictions = np.round(gcRdmF.predict(X_test))
    acc = accuracy_score(y_test,predictions)
    return gcRdmF.best_estimator_,acc


def modelPerformance(model,n,composer_list,subsample=False):
    acc = []
    confusionMatrix = pd.DataFrame(0,columns=le.inverse_transform(composer_list),index = le.inverse_transform(composer_list))
    for i in range(n):
        if subsample : 
            X_train,X_test,y_train,y_test = splitDataset(subsampling(data,composer_list))
        else :
            data_selected=data[data["composer"].isin(composer_list)]
            X_train,X_test,y_train,y_test = splitDataset(data_selected)
                               
        model.fit(X_train, y_train)
        predictions = np.round(model.predict(X_test))
        acc.append(accuracy_score(y_test,predictions))
        confusionMatrix = confusionMatrix + pd.DataFrame(confusion_matrix(y_test,predictions),columns=le.inverse_transform(composer_list),index = le.inverse_transform(composer_list))
    meanAccuracy = np.mean(acc)
    confidenceIntervalAccuracy = st.t.interval(0.95, len(acc)-1, loc=np.mean(acc), scale=st.sem(acc))
    
    return(meanAccuracy,confidenceIntervalAccuracy,confusionMatrix)

def modelCombination(models,data,composer_list,subsample=False):
    if subsample : 
        X_train,X_test,y_train,y_test = splitDataset(subsampling(data,composer_list))
    else :
        X_train,X_test,y_train,y_test = splitDataset(data[data["composer"].isin(composer_list)])
        
    predictions = pd.DataFrame(columns = ["Model0","Model1","Model2"], index = y_test.index)
    for k,model in enumerate(models):
        name = "Model"+str(k)
        model.fit(X_train, y_train)
        predictionsModel = np.round(model.predict(X_test))
        predictions[name] = predictionsModel
    accuracy_score(predictions.mode(axis=1)[0], y_test)
    return predictions, acc

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ## Import des fonctions "maison"
    #import functions

    data = pd.read_csv("Features_2018-12-07.csv",sep=',',index_col=0)
    le = LabelEncoder().fit(data["composer"])
    data["composer"]=le.transform(data["composer"])
    
    uniqLabels, counts = np.unique(data["composer"], return_counts=True)
    composersList = []
    for i, lab in enumerate (uniqLabels):
        print("Classe {} contient {} points".format(lab, counts[i]))
        if counts[i]>50:
            composersList.append(lab)


    #Utiliser la ligne suivante pour retirer un compositeur de votre choix
    composersList.remove(le.transform(['schumann']))

    log,acc1 = logisticRegression(data, composersList, subsample=False, normalisation=True)
    svm,acc2 = svmRBF(data,composersList,subsample=False,normalisation=True)
    rdmf,acc3 = randomForest(data,composersList,subsample=False,normalisation=True)


    accLog,perfLog,confusionMatrixLog = modelPerformance(log,20,composersList,subsample=False)
    perfSVM = modelPerformance(svm,20,composersList,subsample=False)
    perfRDMF = modelPerformance(rdmf,20,composersList,subsample=False)
