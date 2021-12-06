#from src.srgz.io.hpopt import HPOpt
from typing import Any, Dict

import pandas as pd
import logging

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import joblib


from hyperopt import hp
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

import lightgbm as lgb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

from time import time
import torch

from srgz.io.hpopt import HPOpt


def params_optimization(X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
    lgb_reg_params = {
    'min_child_samples':hp.randint('min_child_samples', 50)+1,
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.9),
    'num_leaves' :      hp.randint('num_leaves', 100)+10,
    'min_child_weight': hp.uniform('min_child_weight', 0.001, 0.99),
        'n_estimators':     1000
    }
    lgb_fit_params = {
        'early_stopping_rounds': 50,
        'verbose': False
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params
    lgb_para['score'] = lambda y, pred: -accuracy_score(y, pred)


    rf_reg_params = {
        'min_samples_leaf': hp.randint('min_samples_leaf', 20)+1,
        'min_samples_split':hp.uniform('min_samples_split', 0.001, 0.1),
        #'max_features':     hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
        #'learning_rate':    hp.uniform('learning_rate', 0.001, 0.1),
        'n_estimators':     hp.randint('n_estimators', 800)+100
    }
    rf_fit_params = {
    }
    rf_para = dict()
    rf_para['reg_params'] = rf_reg_params
    rf_para['fit_params'] = rf_fit_params
    rf_para['score'] = lambda y, pred: -accuracy_score(y, pred)

    tabnet_reg_params = {
        'n_d' :              8,
        'n_a' :              8,
        'n_shared':          hp.randint('n_shared', 3)+1,
        'n_independent':     hp.randint('n_independent', 3)+1,
        'n_steps' :          hp.randint('n_steps', 3)+3,
        'gamma' :            hp.uniform('gamma', 1.0, 3.0),
        'lambda_sparse' :    hp.uniform('lambda_sparse', 0.0, 0.01),
        'optimizer_params' : dict(lr=2e-2),
        'scheduler_params' : {"step_size":200, "gamma":0.95},
        'scheduler_fn' :     torch.optim.lr_scheduler.StepLR,
        'mask_type' :       'entmax'
    }

    tabnet_fit_params = {
        'max_epochs' : 2000, 
        'patience' : 20,
        'batch_size' : 1024,
        'virtual_batch_size' : 128,
        'num_workers' : 0,
        'weights' : 1,
        'drop_last' : False,
        #'from_unsupervised' : unsupervised_model
    }

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    tabnet_para = dict()
    tabnet_para['reg_params'] = tabnet_reg_params
    tabnet_para['fit_params'] = tabnet_fit_params
    tabnet_para['score'] = lambda y, pred: -accuracy_score(y, pred)
    
    obj = HPOpt(X_train, y_train, cv=2)
    lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals = 10)
    print("HEREEEE: ")
    print(type(lgb_opt))
    print("--- FINISH ---")
    return lgb_opt


def model_fitting(lgb_opt: Dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> LGBMClassifier:
    lgb_fit_params = {
        'early_stopping_rounds': 50,
        'verbose': False
    }
    gb = lgb.LGBMClassifier( 
                            **{'colsample_bytree': lgb_opt[0]['colsample_bytree'],
                              'min_child_samples': lgb_opt[0]['min_child_samples']+1,
                              'min_child_weight': lgb_opt[0]['min_child_weight'],
                              'num_leaves': lgb_opt[0]['num_leaves']+10,
                              'n_estimators': 1000
                              }
                          )
    t = time()
    gb.fit(X_train[:-500], y_train[:-500], eval_set=[(X_train[:-500], y_train[:-500]), (X_train[-500:], y_train[-500:])],  **lgb_fit_params)
    print("TIME: ")
    print(time()-t)
    print()
    #gb_test_acc = accuracy_score(y_test[-500:], gb.predict(X_test[-500:]))
    #print("ACCURACY - "+str(gb_test_acc))
    return(gb)


def model_testing(X_test: pd.DataFrame, y_test: pd.Series, gb: lgb.LGBMClassifier) -> float:
    gb_test_acc = accuracy_score(y_test[-500:], gb.predict(X_test[-500:]))
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test dataset: " + str(gb_test_acc))
    return(gb_test_acc)