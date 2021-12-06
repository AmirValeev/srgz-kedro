from typing import Any, Dict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler

import pandas as pd
import numpy as np


def data_split(X, y, c=10000, test_size = 0.8):

    X1_train, X1_test, y1_train, y1_test = train_test_split(X[y==1], y[y==1], test_size=test_size, random_state = 43)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X[y==2], y[y==2], test_size=test_size, random_state = 43)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X[y==3], y[y==3], test_size=test_size, random_state = 43)
    
    count = c
    count1 = c

    X_train, X_test = np.concatenate((X1_train[:count], X2_train[:count], X3_train[:count])), np.concatenate((X1_test[:count1], X2_test[:count1], X3_test[:count1]))
    y_train, y_test = np.concatenate((y1_train[:count], y2_train[:count], y3_train[:count])), np.concatenate((y1_test[:count1], y2_test[:count1], y3_test[:count1]))

    data_train = np.concatenate((X_train, y_train.reshape((len(y_train), 1))), axis=1)
    np.random.shuffle(data_train)

    data_test = np.concatenate((X_test, y_test.reshape((len(y_test), 1))), axis=1)
    np.random.shuffle(data_test)
    

    return data_train, data_test


def feature_prep(features_raw: pd.DataFrame) -> Dict[str, Any]:
    column = 'sdssdr16_r_cmodel'
    
    sdss = [i for i in features_raw if 'sdss' in i and 'decals' not in i and column not in i] 
    decals = [i for i in features_raw if 'decals' in i and 'sdss' not in i and 'psdr' not in i and column not in i] 
    wise = [i for i in decals if 'Lw' in i and column not in i] 
    ps = [i for i in features_raw if 'psdr' in i and 'decals' not in i and column not in i] 
    return {
        "sdssdr16+wise_decals8tr": sdss+wise,
        "psdr2+wise_decals8tr": ps+wise,
        "sdssdr16+all_decals8tr": sdss+decals,
        "psdr2+all_decals8tr": ps+decals,
        "decals8tr": decals,
        "sdssdr16+psdr2+wise_decals8tr": sdss+ps+wise,
        "sdssdr16+psdr2+all_decals8tr": sdss+ps+decals
            }


def data_preparation(features: Dict, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
    overview = 'sdssdr16+wise_decals8tr'
    X1, y1 = df1[features[overview]].values, df1['class'].values 
    X2, y2 = df2[features[overview]].values, df2['class'].values

    data, datat = data_split(np.concatenate((X1, X2)), np.concatenate((y1, y2)), test_size=0.5, c=5000000)
    #ставишь 'c' меньше - получаешь выборку меньше и более сбалансированную по количеству
    X1, y1 = data[:, :-1], data[:, -1].astype('int')
    X2, y2 = datat[:, :-1], datat[:, -1].astype('int')

    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))
    print(X.shape, y.shape)
    #нормализуем данные
    robust = RobustScaler()
    X_train_norm = robust.fit_transform(X)
    print(X_train_norm.shape)
    X_test_norm = robust.transform(X)
    y_train = y
    y_test = y
    
    return {
        'train_x': X_train_norm,
        'test_x': X_test_norm,
        'train_y': y_train,
        'test_y': y_test
    }








