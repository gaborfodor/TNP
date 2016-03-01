# Telstra Network Disruptions - Predict service faults on Australia's largest telecommunications network

This is my simplified single model solution for the Telstra kaggle competition https://www.kaggle.com/c/telstra-recruiting-network. With the hard coded parameters one should be able reach around 0.41 on Private LB (~17th). With some additional tuning and luck it should be possible to reach 0.405 (~13th). Model averaging helped me to rank 7th out of the thousand participants.

More information about the solution and other approaches could be found in this thread: https://www.kaggle.com/c/telstra-recruiting-network/forums/t/19239/it-s-been-fun-post-your-code-github-links-here-after-the-competition/109851#post109851


# Execute
1. mkdir data
2. copy the dataset from https://www.kaggle.com/c/telstra-recruiting-network/data into the data folder
3. python extract_features.py
4. python feature_importance.py
5. python create_submission.py

# Requirements
    python 2.7.9
    pandas (0.17.0)
    numpy (1.10.4)
    xgboost 0.4
