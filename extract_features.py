import pandas as pd
import numpy as np
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir
import datetime as dt


MIN_FREQ, LOCATION_CAT, FEATURE_CAT, FAULT_LOOKBACK, SHIFT = 20, 4, 5, 10, 1
print MIN_FREQ, LOCATION_CAT, FEATURE_CAT, FAULT_LOOKBACK, SHIFT
base_path = getcwd()
data_path = join(base_path, 'data')
feature_path = join(base_path, 'extracted_features')
if not exists(feature_path):
    makedirs(feature_path)
time0 = dt.datetime.now()

# ---------------------------------------------------------------------------------
# append train & test
# ---------------------------------------------------------------------------------
train = pd.read_csv(join(data_path, 'train.csv'))
train['location_id'] = train.location.apply(lambda x: int(x.split('location ')[1]))
test = pd.read_csv(join(data_path, 'test.csv'))
test['fault_severity'] = -1
test['location_id'] = test.location.apply(lambda x: int(x.split('location ')[1]))
print 'train', train.shape, 'test', test.shape
features = train.append(test)
features = features.drop('location', axis=1)
print features.shape

# ---------------------------------------------------------------------------------
# order ~ time
# ---------------------------------------------------------------------------------
severity_type = pd.read_csv(join(data_path, 'severity_type.csv'))
severity_type_order = severity_type[['id']].drop_duplicates()
severity_type_order['order'] = 1. * np.arange(len(severity_type_order)) / len(severity_type_order)
features = pd.merge(features, severity_type_order, how='inner', on='id')
print features.shape
print features[:3]

# ---------------------------------------------------------------------------------
# location count
# ---------------------------------------------------------------------------------
location_count = features.groupby('location_id').count()[['id']]
location_count.columns = ['location_count']
features = pd.merge(features, location_count, how='inner', left_on='location_id', right_index=True)
print features.shape

# ---------------------------------------------------------------------------------
# binarize frequent locations
# ---------------------------------------------------------------------------------
frequent_locations = location_count[location_count['location_count'] > MIN_FREQ]
frequent_location_records = features[features['location_id'].isin(frequent_locations.index)].copy()
frequent_location_records['value'] = 1
location_features = frequent_location_records.pivot(index='id', columns='location_id', values='value')
location_features.columns = ['location_%i' % c for c in location_features.columns]
print 'location_features', location_features.shape
features = pd.merge(features, location_features, how='left', left_on='id', right_index=True)
features = features.fillna(0)
print features.shape

# ---------------------------------------------------------------------------------
# event type ['id', 'event_type'] (31170, 2)
# ---------------------------------------------------------------------------------
event_type = pd.read_csv(join(data_path, 'event_type.csv'))
event_count = event_type.groupby('id').count()[['event_type']]
event_count.columns = ['event_type_count']
features = pd.merge(features, event_count, how='inner', left_on='id', right_index=True)
print features.shape
event_type_count = event_type.groupby('event_type').count()[['id']].sort_values(by='id', ascending=False)
frequent_event_types = event_type_count[event_type_count['id'] > MIN_FREQ]
frequent_event_records = event_type[event_type['event_type'].isin(frequent_event_types.index)].copy()
frequent_event_records['value'] = 1
event_features = frequent_event_records.pivot(index='id', columns='event_type', values='value')
event_features.columns = map(lambda x: x.replace(' ', '_'), event_features.columns)
print 'event features', event_features.shape
features = pd.merge(features, event_features, how='left', left_on='id', right_index=True)
print features.shape
rare_event_types = event_type_count[event_type_count['id'] <= MIN_FREQ]
rare_event_records = event_type[event_type['event_type'].isin(rare_event_types.index)].copy()
rare_event_records['value'] = 1
rare_event_feature = rare_event_records.groupby('id').max()[['value']]
rare_event_feature.columns = ['rare_event_type']
features = pd.merge(features, rare_event_feature, how='left', left_on='id', right_index=True)
print features.shape
event_type['event_id'] = event_type.event_type.apply(lambda x: int(x.split('event_type ')[1]))
max_event_cat = event_type.groupby('id').max()[['event_id']] // 3
max_event_cat.columns = ['max_event_type_cat']
min_event_cat = event_type.groupby('id').min()[['event_id']] // 3
min_event_cat.columns = ['min_event_type_cat']
features = pd.merge(features, max_event_cat, how='left', left_on='id', right_index=True)
features = pd.merge(features, min_event_cat, how='left', left_on='id', right_index=True)
print features.shape
features = features.fillna(0)

# ---------------------------------------------------------------------------------
# log_feature ['id', 'log_feature', 'volume'] (58671, 3)
# ---------------------------------------------------------------------------------
log_feature = pd.read_csv(join(data_path, 'log_feature.csv'))
log_feature_count = log_feature.groupby('id').count()[['log_feature']]
log_feature_count.columns = ['log_feature_count']
features = pd.merge(features, log_feature_count, how='inner', left_on='id', right_index=True)
print features.shape
log_feature_count = log_feature.groupby('log_feature').count()[['id']].sort_values(by='id', ascending=False)
frequent_log_features = log_feature_count[log_feature_count['id'] > MIN_FREQ]
frequent_log_feature_records = log_feature[log_feature['log_feature'].isin(frequent_log_features.index)].copy()
log_feature_features = frequent_log_feature_records.pivot(index='id', columns='log_feature', values='volume')
log_feature_features.columns = map(lambda x: x.replace(' ', '_'), log_feature_features.columns)
log_feature_features.columns = map(lambda x: x.replace('feature', 'log_feature'), log_feature_features.columns)
print 'log_feature_features', log_feature_features.shape
features = pd.merge(features, log_feature_features, how='left', left_on='id', right_index=True)
print features.shape
rare_log_features = log_feature_count[log_feature_count['id'] <= MIN_FREQ]
rare_log_feature_records = log_feature[log_feature['log_feature'].isin(rare_log_features.index)].copy()
rare_log_feature_records['value'] = 1
rare_log_feature_feature = rare_log_feature_records.groupby('id').max()[['value']]
rare_log_feature_feature.columns = ['rare_log_feature']
features = pd.merge(features, rare_log_feature_feature, how='left', left_on='id', right_index=True)
print features.shape
log_feature['log_feature_id'] = log_feature.log_feature.apply(lambda x: int(x.split('feature ')[1]))
max_log_feature_cat = log_feature.groupby('id').max()[['log_feature_id']] // FEATURE_CAT
max_log_feature_cat.columns = ['max_log_feature_cat']
median_log_feature_cat = log_feature.groupby('id').median()[['log_feature_id']] // FEATURE_CAT
median_log_feature_cat.columns = ['median_log_feature_cat']
min_log_feature_cat = log_feature.groupby('id').min()[['log_feature_id']] // FEATURE_CAT
min_log_feature_cat.columns = ['min_log_feature_cat']
features = pd.merge(features, max_log_feature_cat, how='left', left_on='id', right_index=True)
features = pd.merge(features, median_log_feature_cat, how='left', left_on='id', right_index=True)
features = pd.merge(features, min_log_feature_cat, how='left', left_on='id', right_index=True)
print features.shape
log_feature['log_feature_id_cat'] = log_feature['log_feature_id'] // FEATURE_CAT
log_feature_cat = log_feature.groupby(['id', 'log_feature_id_cat']).sum()['volume']
log_feature_cat = log_feature_cat.reset_index()
log_feature_cat_feature = log_feature_cat.pivot(index='id', columns='log_feature_id_cat', values='volume')
log_feature_cat_feature.columns = ['log_feature_cat_%i' % c for c in log_feature_cat_feature.columns]
features = pd.merge(features, log_feature_cat_feature, how='left', left_on='id', right_index=True)
print 'log_feature_cat_feature', log_feature_cat_feature.shape
log_feature.loc[log_feature['volume'] > 49, 'volume'] = 50
volume_counts = log_feature.groupby(['id', 'volume']).count()[['log_feature']].reset_index()
volume_features = volume_counts.pivot(index='id', columns='volume', values='log_feature')
volume_features.columns = ['volume_%i' % c for c in volume_features.columns]
print 'volume_features', volume_features.shape
features = pd.merge(features, volume_features, how='left', left_on='id', right_index=True)
print features.shape
features = features.fillna(0)

# ---------------------------------------------------------------------------------
# resource_type ['id', 'resource_type'] (21076, 2)
# ---------------------------------------------------------------------------------
resource_type = pd.read_csv(join(data_path, 'resource_type.csv'))
resource_type['value'] = 1
resource_type_count = resource_type.groupby('id').count()[['value']]
resource_type_count.columns = ['resource_type_count']
features = pd.merge(features, resource_type_count, how='left', left_on='id', right_index=True)
resource_type_features = resource_type.pivot(index='id', columns='resource_type', values='value')
resource_type_features.columns = [c.replace(' ', '_') for c in resource_type_features.columns]
resource_type_features = resource_type_features[['resource_type_1', 'resource_type_10', 'resource_type_2',
                                                 'resource_type_3', 'resource_type_4', 'resource_type_6',
                                                 'resource_type_7', 'resource_type_8', 'resource_type_9']]
print 'resource_type_features', resource_type_features.shape
features = pd.merge(features, resource_type_features, how='left', left_on='id', right_index=True)
print features.shape

# ---------------------------------------------------------------------------------
# severity_type ['id', 'severity_type'] (18552, 2)
# ---------------------------------------------------------------------------------
severity_type = pd.read_csv(join(data_path, 'severity_type.csv'))
severity_type['value'] = 1
severity_type_features = severity_type.pivot(index='id', columns='severity_type', values='value')
severity_type_features.columns = [c.replace(' ', '_') for c in severity_type_features.columns]
severity_type_features = severity_type_features.fillna(0)
severity_type_features = severity_type_features[['severity_type_1', 'severity_type_2', 'severity_type_4', 'severity_type_5']]
print 'severity_type_features', severity_type_features.shape
features = pd.merge(features, severity_type_features, how='left', left_on='id', right_index=True)
print features.shape
features = features.fillna(0)
features['location_cat'] = features['location_id'] // LOCATION_CAT
features['location_cat2'] = (features['location_id'] + LOCATION_CAT//2) // LOCATION_CAT
features = features.sort_values(by='order')
feature_names = list(features.columns)
feature_names.remove('id')
feature_names.remove('fault_severity')
feature_names.remove('location_id')
feature_names.remove('order')

# ---------------------------------------------------------------------------------
# Before features
# ---------------------------------------------------------------------------------
ids = features['id'].values
location = features['location_id'].values
for shift in range(1, SHIFT + 1):
    before_dt = features[feature_names].values
    before_dt = before_dt[shift:, :] - before_dt[:-shift, :]
    location_mask = 1. * (location[shift:] == location[:-shift])
    location_mask[location_mask == 0] = np.nan
    before_cols = [c + '_diff_before_%i' % shift for c in feature_names]
    before_dt_df = pd.DataFrame(before_dt, columns=before_cols)
    useful_cols = []
    for c in before_cols:
        before_dt_df[c] = before_dt_df[c] * location_mask
        non_zero_count = np.sum(1*(before_dt_df[c].fillna(0) != 0))
        if non_zero_count > MIN_FREQ:
            useful_cols.append(c)
    before_dt_df = before_dt_df[useful_cols].copy()
    before_dt_df['id'] = ids[shift:]
    features = pd.merge(features, before_dt_df, how='left', on='id')
    print 'before', shift, features.shape

# ---------------------------------------------------------------------------------
# After features
# ---------------------------------------------------------------------------------
ids = features['id'].values
location = features['location_id'].values
for shift in range(1, SHIFT + 1):
    after_dt = features[feature_names].values
    after_dt = after_dt[:-shift, :] - after_dt[shift:, :]
    location_mask = 1. * (location[:-shift] == location[shift:])
    location_mask[location_mask == 0] = np.nan
    after_cols = [c + '_diff_after_%i' % shift for c in feature_names]
    after_dt_df = pd.DataFrame(after_dt, columns=after_cols)
    useful_cols = []
    for c in after_cols:
        after_dt_df[c] = after_dt_df[c] * location_mask
        non_zero_count = np.sum(1*(after_dt_df[c].fillna(0) != 0))
        if non_zero_count > MIN_FREQ:
            useful_cols.append(c)
    after_dt_df = after_dt_df[useful_cols].copy()
    after_dt_df['id'] = ids[:-shift]
    features = pd.merge(features, after_dt_df, how='left', on='id')
    print 'after', shift, features.shape
features = features.fillna(-9999)

# ---------------------------------------------------------------------------------
# before fault_severity
# ---------------------------------------------------------------------------------
ids = features['id'].values
location = features['location_id'].values
fault_severity = features['fault_severity'].values
for diff in range(1, FAULT_LOOKBACK + 1):
    before_fault_severity = fault_severity[:-diff]
    location_mask = 1. * (location[:-diff] == location[diff:])
    location_mask[location_mask == 0] = np.nan
    before_fault_severity_df = pd.DataFrame({'before_fs_%i' % diff: before_fault_severity})
    before_fault_severity_df['before_fs_%i' % diff] = location_mask * before_fault_severity_df['before_fs_%i' % diff]
    before_fault_severity_df['id'] = ids[diff:]
    features = pd.merge(features, before_fault_severity_df, how='left', on='id')
before = features[['before_fs_%i' % d for d in range(1, FAULT_LOOKBACK+1)]]
before = before.replace(-1, np.nan)
before_values = before.values
for diff in range(3, FAULT_LOOKBACK + 1):
    features['before_fs__mean_%i' % diff] = np.nanmean(before_values[:, :diff], axis=1)
    features['before_fs_sum_%i' % diff] = np.nansum(before_values[:, :diff], axis=1)
before = before.replace(0, 1)
before = before.replace(2, 1)
before_values = before.values
for diff in range(3, FAULT_LOOKBACK + 1):
    features['before_fs_count_%i' % diff] = np.nansum(before_values[:, :diff], axis=1)

# ---------------------------------------------------------------------------------
# after fault_severity
# ---------------------------------------------------------------------------------
ids = features['id'].values
location = features['location_id'].values
fault_severity = features['fault_severity'].values
for diff in range(1, FAULT_LOOKBACK+1):
    after_fault_severity = fault_severity[diff:]
    location_mask = 1. * (location[:-diff] == location[diff:])
    location_mask[location_mask == 0] = np.nan
    after_fault_severity_df = pd.DataFrame({'after_fs_%i' % diff: after_fault_severity})
    after_fault_severity_df['after_fs_%i' % diff] = location_mask * after_fault_severity_df['after_fs_%i' % diff]
    after_fault_severity_df['id'] = ids[:-diff]
    features = pd.merge(features, after_fault_severity_df, how='left', on='id')
after = features[['after_fs_%i' % d for d in range(1, FAULT_LOOKBACK+1)]]
after = after.replace(-1, np.nan)
after_values = after.values
for diff in range(3, FAULT_LOOKBACK + 1):
    features['after_fs__mean_%i' % diff] = np.nanmean(after_values[:, :diff], axis=1)
    features['after_fs_sum_%i' % diff] = np.nansum(after_values[:, :diff], axis=1)
after = after.replace(0, 1)
after = after.replace(2, 1)
after_values = after.values
for diff in range(3, FAULT_LOOKBACK + 1):
    features['after_fs_count_%i' % diff] = np.nansum(after_values[:, :diff], axis=1)
features = features.fillna(-9999)

# ---------------------------------------------------------------------------------
# rank features
# ---------------------------------------------------------------------------------
features['location_rank_asc'] = features.groupby('location_id')[['order']].rank()
features['location_rank_desc'] = features.groupby('location_id')[['order']].rank(ascending=False)
features['location_rank_rel'] = 1. * features['location_rank_asc'] / features['location_count']
features['location_rank_rel'] = np.round(features['location_rank_rel'], 2)

# ---------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------
feature_file_name = 'features_mf%i_lc%i_fc%i_fl%i_sh%i.csv' % (MIN_FREQ, LOCATION_CAT, FEATURE_CAT,
                                                               FAULT_LOOKBACK, SHIFT)
features.to_csv(join(feature_path, feature_file_name), index=False)
print 'final features', features.shape
time1 = dt.datetime.now()
print 'total:', (time1-time0).seconds, 'sec'
