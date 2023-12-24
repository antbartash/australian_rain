# CatBoost installation on Google Colab
import subprocess
subprocess.run(['pip', 'install', 'catboost'], check=True)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from catboost import CatBoostClassifier
import nni
import logging
import json



LOG = logging.getLogger('nni_catboost')

def load_data():
    data = pd.read_csv(
        'https://raw.githubusercontent.com/antbartash/australian_rain/main/data/data_transformed.csv',
        index_col=0
    )
    X, y = data.drop(columns=['RainTomorrow', 'RainToday']), data['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for column in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']:
        X_train[column] = X_train[column].astype(np.float32).fillna(-1).apply(lambda x: str(x))
        X_test[column] = X_test[column].astype(np.float32).fillna(-1).apply(lambda x: str(x))
    return X_train, X_test, y_train, y_test


def get_default_parameters():
    params = {
        'n_estimators': 100,
        'learning_rate': 0.3,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_strength': 1.0, # CPU only
        'bagging_temperature': 1.0,
        'grow_policy': 'SymmetricTree',
        'scale_pos_weight': 1.0,
    }
    return params

    
def get_model(PARAMS):
    model = CatBoostClassifier(
        cat_features=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],
        custom_metric='AUC:hints=skip_train~false',
        random_state=42, verbose=False#, task_type='GPU'
    )
    model.set_params(**PARAMS)
    return model


def run(X_train, y_train, model):
    model.fit(X_train, y_train)
    score = np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc'))

    catboost_info = json.load(open('catboost_info/catboost_training.json'))
    iterations = pd.DataFrame(catboost_info['iterations'])
    iterations = iterations[iterations['iteration'] % 100 == 0].reset_index(drop=True)
    for rowid in iterations.index:
        metrics = {
            'default': iterations.loc[rowid, 'learn'][0],
            'AUC': iterations.loc[rowid, 'learn'][1]
        }
        LOG.debug('mertics: ', metrics)
        nni.report_intermediate_result(metrics)

    LOG.debug('score: %s', score)
    nni.report_final_result(score)


if __name__ == '__main__':
    X_train, _, y_train, _ = load_data()
    try:
        RECIEVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECIEVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECIEVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, y_train, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
