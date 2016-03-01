import pandas as pd
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir
import datetime as dt
import xgboost as xgb

base_path = getcwd()
feature_path = join(base_path, 'extracted_features')
feature_importance_path = join(base_path, 'feature_importance')
if not exists(feature_importance_path):
    makedirs(feature_importance_path)

NJOBS = 6
NUM_ROUND = 1000
time0 = dt.datetime.now()

for file_name in listdir(feature_path):
    fimp_file_name = join(feature_importance_path, file_name.replace('features', 'feature_importance'))
    if not exists(fimp_file_name):
        print file_name
        features = pd.read_csv(join(feature_path, file_name))
        train = features[features['fault_severity'] >= 0].copy()
        print train.shape
        feature_names = list(train.columns)
        feature_names.remove('id')
        feature_names.remove('fault_severity')
        feature_names.remove('location_id')
        feature_names.remove('order')
        print len(feature_names)
        parameters = {'min_child_weight': 3, 'eta': 0.05, 'colsample_bytree': 0.4,
                      'max_depth': 10, 'subsample': 0.9, 'lambda': 0.5, 'nthread': NJOBS,
                      'objective': 'multi:softprob', 'silent': 0, 'num_class': 3}
        fs = ['f%i' % i for i in range(len(feature_names))]
        dtrain = xgb.DMatrix(train[feature_names].values, label=train['fault_severity'].values,
                             missing=-9999, feature_names=fs)
        model = xgb.train(parameters, dtrain, NUM_ROUND)
        feature_imp = model.get_fscore()
        f1 = pd.DataFrame({'f': feature_imp.keys(), 'imp': feature_imp.values()})
        f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
        feature_impoertance = pd.merge(f1, f2, how='right', on='f')
        feature_impoertance.to_csv(fimp_file_name, index=False)

time1 = dt.datetime.now()
print 'total:', (time1-time0).seconds, 'sec'
