'''
Functions for data preprocessing
'''

def subsampling(data,composer_list):
    datamin = data[data["composer"].isin(composer_list)]["composer"].value_counts().min()
    composermin = data[data["composer"].isin(composer_list)]["composer"].value_counts().idxmin()
    composer_list.remove(composermin)
    data_selected = data[data["composer"]==composermin]
    print (composer_list)
    for c in composer_list:
        data_c = data[data["composer"]==c]
        data_c = data_c.sample(datamin)
        data_selected = pd.concat([data_selected,data_c])
    return data_selected

def splitDataset(data): 
    data = data.drop("song_name",axis=1)
    X = data.drop("composer",axis=1)
    y = data["composer"]

    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state= 42,stratify = y)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.3,random_state=42,stratify = y_train)

    return X_train,X_test,y_train,y_test

