import pandas as pd
import numpy as np
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir
import datetime as dt
import xgboost as xgb
import random

NFOLD = 10
NJOBS = 6


def parse_xgb_cv_result(result):
    result_df = []
    print result
    for row in result:
        print row
        row_tab_sep = row.split('\t')
        iteration = int(row_tab_sep[0][1:-1])
        test_logloss = np.float(row_tab_sep[1].split(':')[1].split('+')[0])
        test_logloss_std = np.float(row_tab_sep[1].split(':')[1].split('+')[1])
        result_df.append([iteration, test_logloss, test_logloss_std])
    result_df = pd.DataFrame(result_df, columns=['i', 'mlogloss', 'std'])
    return result_df


def random_xgb_parameters():
    MCW = [3]
    ETA = [0.04]
    CS = [0.25]
    MD = [8]
    SS = [0.9]
    LAMBDA = [0, 0.1, 0.5]
    parameters = {'min_child_weight': random.choice(MCW),
                  'eta': random.choice(ETA),
                  'colsample_bytree': random.choice(CS),
                  'max_depth': random.choice(MD),
                  'subsample': random.choice(SS),
                  'lambda': random.choice(LAMBDA),
                  'nthread': NJOBS, 'objective': 'multi:softprob', 'silent': 1, 'num_class': 3}
    return parameters


base_path = getcwd()
log_path = join(base_path, 'log')
if not exists(log_path):
    makedirs(log_path)
feature_path = join(base_path, 'extracted_features')
feature_importance_path = join(base_path, 'feature_importance')
prediction_path = join(base_path, 'prediction')
if not exists(prediction_path):
    makedirs(prediction_path)

parameter_cv_result = []

time0 = dt.datetime.now()

feature_file_names = listdir(feature_path)
feature_file_name = random.choice(feature_file_names)
print feature_file_name
features = pd.read_csv(join(feature_path, feature_file_name))
print 'features', features.shape
feature_importance_file_name = join(feature_importance_path,
                                    feature_file_name.replace('features', 'feature_importance'))
feature_importance = pd.read_csv(feature_importance_file_name)
feature_importance = feature_importance.fillna(0)

drop_cols = list(feature_importance[feature_importance['imp'] < 1]['feature_name'])
print 'drop cols', len(drop_cols)

train = features[features['fault_severity'] >= 0].copy()
test = features[features['fault_severity'] < 0].copy()
print train.shape, test.shape

feature_names = list(train.columns)
feature_names.remove('id')
feature_names.remove('fault_severity')
feature_names.remove('location_id')
feature_names.remove('order')
feature_names = list(set(feature_names) - set(drop_cols))
print 'features', len(feature_names)

parameters = random_xgb_parameters()
dtrain = xgb.DMatrix(train[feature_names].values, label=train['fault_severity'].values, missing=-9999)
print parameters
result = xgb.cv(parameters, dtrain, 500, nfold=NFOLD, metrics={'mlogloss'}, seed=0)
result_df = parse_xgb_cv_result(result)
result_df = result_df.sort_values(by='mlogloss')
best_iter = result_df.iloc[0]
NUM_ROUND = int(best_iter['i'])
LOGLOSS = best_iter['mlogloss']
parameter_cv_result.append([feature_file_name] + parameters.values() + list(best_iter))
parameter_cv_result_df = pd.DataFrame(parameter_cv_result,
                                      columns=['feature_file_name'] + parameters.keys() + list(best_iter.index))
parameter_cv_result_df.to_csv(join(log_path, 'prediction_cv_results.csv'), index=False)

# ---------------------------------------------------------------------------------
# Create OOF prediction and submission
# ---------------------------------------------------------------------------------
train['cv'] = np.random.randint(0, NFOLD, len(train))
train_predictions = []
for cv in range(NFOLD):
    train_train = train[train['cv'] != cv].copy()
    train_test = train[train['cv'] == cv].copy()
    print train_train.shape, train_test.shape
    dtrain_train = xgb.DMatrix(train_train[feature_names].values, label=train_train['fault_severity'].values,
                               missing=-9999)
    dtrain_test = xgb.DMatrix(train_test[feature_names].values, missing=-9999)
    dtest = xgb.DMatrix(test[feature_names].values, missing=-9999)
    model = xgb.train(parameters, dtrain_train, NUM_ROUND)
    train_prediction = pd.DataFrame(model.predict(dtrain_test), columns=['predict_0', 'predict_1', 'predict_2'])
    train_prediction['id'] = train_test['id'].values
    train_prediction = train_prediction[['id', 'predict_0', 'predict_1', 'predict_2']]
    train_predictions.append(train_prediction)
    test_prediction = pd.DataFrame(model.predict(dtest), columns=['predict_0', 'predict_1', 'predict_2'])
    if cv == 0:
        test_predictions = test_prediction / NFOLD
    else:
        test_predictions = test_predictions + test_prediction / NFOLD

test_predictions['id'] = test['id'].values
test_predictions = test_predictions[['id', 'predict_0', 'predict_1', 'predict_2']]
prediction_id = len(listdir(prediction_path)) // 2
test_predictions.to_csv(join(prediction_path,
                             'submission_mlogloss%i_%i.csv' % (int(100000 * LOGLOSS), prediction_id)), index=False)
train_prediction = pd.concat(train_predictions)
train_prediction.to_csv(join(prediction_path,
                             'train_prediction_mlogloss%i_%i.csv' % (int(100000 * LOGLOSS), prediction_id)), index=False)
print test_prediction.shape, train_prediction.shape
time1 = dt.datetime.now()
print 'total:', (time1-time0).seconds, 'sec'
