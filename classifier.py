from main import *
from joblib import load

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def predict(X):
    X = pd.read_csv(X)
    X = preprocess(X)[0]
    model = load("forest_trained_model600.pkl")
    return model.predict(X)


def send_police_cars(X):
    kmeans0 = load("part_b_model0.pkl")
    kmeans1 = load("part_b_model1.pkl")
    kmeans2 = load("part_b_model2.pkl")
    kmeans3 = load("part_b_model3.pkl")
    kmeans4 = load("part_b_model4.pkl")
    kmeans5 = load("part_b_model5.pkl")
    kmeans6 = load("part_b_model6.pkl")
    models = {0: kmeans0.cluster_centers_, 1: kmeans1.cluster_centers_, 2: kmeans2.cluster_centers_,
              3: kmeans3.cluster_centers_, 4: kmeans4.cluster_centers_, 5: kmeans5.cluster_centers_,
              6: kmeans6.cluster_centers_}

    for i in range(7):
        nump = np.asarray(models[i])
        time = nump[:, [2]]
        time = time / 60
        hour = time.astype(np.int).flatten()
        minute = ((time - time.astype(np.int)) * 60).astype(np.int).flatten()
        final_string_time = np.full(shape=30, fill_value='xxxxx')
        for j in range(30):
            h = f'{hour[j]}'
            if hour[j] < 10:
                h = '0' + h
            m = f'{minute[j]}'
            if minute[j] < 10:
                m = '0' + m
            final_string_time[j] = h + ':' + m
        final = pd.DataFrame({'time_as_string': final_string_time})
        time_stamp = pd.to_datetime(final['time_as_string']).dt.time
        Latitude = nump[:, [0]].flatten()
        Longitude = nump[:, [1]].flatten()
        result = pd.DataFrame({'Latitude':Latitude, 'Longitude': Longitude})
        result = pd.concat([result, time_stamp],axis=1)
        mat = result.to_numpy()
        lst = []
        for row in range(30):
            lst.append(tuple(mat[row,:]))
        models[i] = lst

    X = pd.Series(X)
    days = pd.to_datetime(X).dt.dayofweek
    # X = X[1:]
    result = []
    for i in range(len(days)):
        result.append(models[days[i]])
    return result

