import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import load, dump


def get_date_dummies(X, delete_date=True):
    X['Month'] = pd.to_datetime(X['Date']).dt.month
    X["DayOfWeek"] = pd.to_datetime(X['Date']).dt.dayofweek
    X['Hour'] = pd.to_datetime(X['Date']).dt.hour
    for i in range(1, 13):
        X[f'Month_{i}'] = np.where(X['Month'] == i, 1, 0)

    for j in range(7):
        X[f'Day_{j}_in_week'] = np.where(X['DayOfWeek'] == j, 1, 0)

    for k in range(24):
        X[f'Hour_{k}'] = np.where(X['Hour'] == k, 1, 0)

    X = X.drop(['Month', 'DayOfWeek','Hour'], axis=1)
    if delete_date:
        X = X.drop(['Date'], axis=1)
    return X


def preprocess(X, y=None, part_a=True):
    if y is not None:
        y = y.replace(['BATTERY', 'THEFT', 'CRIMINAL DAMAGE', 'DECEPTIVE PRACTICE', 'ASSAULT'], [0, 1, 2, 3, 4])

    # rename columns heads with spaces
    X = X.rename(columns={'Case Number': 'CaseNumber', 'Primary Type': 'PrimaryType',
                          'Location Description': 'LocationDescription', 'Community Area': 'CommunityArea',
                          'Updated On': 'UpdatedOn'})

    # delete correlated columns
    X = X.iloc[:, 1:]
    if (part_a):
        X = X.drop(['IUCR', 'Description', 'FBI Code'], axis=1)
    X = X.drop(['ID', 'X Coordinate', 'Y Coordinate', 'Year', 'Location'],
               axis=1)
    X = X.drop(['Block', 'CaseNumber','UpdatedOn'], axis=1)

    # fill Nan values with mode
    X = X.fillna(X.mode().iloc[0])

    # dummies + convert True\False to 1\-1
    X["Arrest"] = np.where(X["Arrest"] == True, 1, -1)
    X["Domestic"] = np.where(X["Domestic"] == True, 1, -1)
    X["Apartment"] = np.where(X["LocationDescription"] == "APARTMENT", 1, 0)
    X["Residence"] = np.where(X["LocationDescription"] == "RESIDENCE", 1, 0)
    X["Street"] = np.where(X["LocationDescription"] == "STREET", 1, 0)
    X["SideWalk"] = np.where(X["LocationDescription"] == "SIDEWALK", 1, 0)

    X = X.drop(['LocationDescription'], axis=1)

    # convert datetime to date
    X = get_date_dummies(X, part_a)

    agc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    X['Geo'] = agc.fit_predict(X[['Latitude', 'Longitude']])
    if part_a:
        X = X.drop(['Latitude', 'Longitude'], axis=1)

    return X, y

