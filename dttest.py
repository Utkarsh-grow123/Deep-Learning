import datetime
import numpy as np
import pandas as pd
import pytz


#
#
# def logic(index):
#     if index % 2 == 0:
#         return True
#     return False

def dateparse(time_in_secs):
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))


def load_and_clean(file):
    df = pd.read_csv(file, parse_dates=[0], date_parser=dateparse)
    # print(df['Timestamp'].head(5))
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    # print(df['Timestamp'].head(5))
    df['Timestamp'] = df['Timestamp'].values.astype(float)
    # print(df['Timestamp'].head(5))
    df['Open'].fillna(method='bfill', inplace=True)
    df['High'].fillna(method='bfill', inplace=True)
    df['Low'].fillna(method='bfill', inplace=True)
    df['Close'].fillna(method='bfill', inplace=True)
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)
    df['Weighted_Price'].fillna(value=0, inplace=True)
    # print(df['Open'].head(500))
    X = df.iloc[:, 0].copy()
    training_set = X.values
    X = np.reshape(training_set, (len(training_set), 1))
    Y = df.iloc[:, 7].values
    return X, Y


def split_data(X, Y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=56745, test_size=0.8)
    return X_train, X_test, Y_train, Y_test


def build_LR_model(X_train, Y_train):
    from sklearn.linear_model import LinearRegression
    LRregressor = LinearRegression().fit(X_train, Y_train)
    return LRregressor


def cross_validation_regressor(algorithm, regressor, X_test, Y_test):
    # evaluate the model
    y_pred = regressor.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from math import sqrt
    mse = sqrt(mean_squared_error(Y_test, y_pred))
    mae = sqrt(mean_absolute_error(Y_test, y_pred))
    # print(f"mse of LR:{mse}")
    # print(f"mae of LR:{mae}")
    # print(f"Predicted price of Bitcoin is : {y_pred}")
    return mse, mae


def build_RF_model(X_train, Y_train):
    from sklearn.ensemble import RandomForestRegressor
    RFregressor = RandomForestRegressor(n_estimators=100, bootstrap=True, max_depth=8, max_features='auto',
                                        min_samples_leaf=10, min_samples_split=10)
    # The minimum number of samples required to split an internal node:
    # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

    RFregressor = RFregressor.fit(X_train, Y_train)
    xtree = RFregressor.estimators_[0]
    from sklearn.tree import export_graphviz
    export_graphviz(xtree, out_file='xyz100.dot')

    return RFregressor


def cross_validation_RFregressor(algorithm, regressor, X_test, Y_test):
    # evaluate the model
    y_pred = regressor.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from math import sqrt
    mse = sqrt(mean_squared_error(Y_test, y_pred))
    mae = sqrt(mean_absolute_error(Y_test, y_pred))
    print(f"mse of LR:{mse}")
    print(f"mae of LR:{mae}")
    print(f"Predicted price of Bitcoin is : {y_pred}")
    # return mse, mae


def build_XGB_model(X_train, Y_train, X_Test, Y_Test):
    import xgboost as xgb
    model = xgb.XGBRegressor(objective='reg:squarederror', eval_matric='rmse', min_child_weight=10, booster='dart',
                             colsample_bytree=0.3,
                             learning_rate=0.02, max_depth=2, alpha=5, n_estimators=100)
    XGBregressor = model.fit(X_train, Y_train,
                             verbose=False)
    return XGBregressor
    # LR 0.01 => bitcoin_price: 3781.8284
    # LR 0.02 =>


def cross_validation_XGBregressor(algorithm, regressor, X_test, Y_test):
    # evaluate the model
    y_pred = regressor.predict(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from math import sqrt
    mse = np.sqrt(mean_squared_error(y_pred, Y_test))
    mae = np.sqrt(mean_absolute_error(Y_test, y_pred))
    print(f"Predicted price of Bitcoin is : {y_pred}")
    # print(f"mse of XGBoost:{mse}")
    # print(f"mae of XGBoost:{mae}")
    return mse, mae


def build_rnn_model(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam
    import keras.backend as K
    K.clear_session()
    model = Sequential()
    model.add(Dense(1, input_shape=(X_test.shape[1],), activation='relu',
                    kernel_initializer='lecun_uniform'))  # 1 hidden layer with 1 newron
    model.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu'))
    model.add(Dense(100, input_shape=(X_test.shape[1],), activation='relu'))
    model.add(Dense(
        1))  # 2 Hidden layers with 50 neurons each and ReLu activation function  op : R-Squared: -0.390489 R-squared is : -0.390489 3996.299364621151 46.40323901395093
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    model.fit(X_train, Y_train, batch_size=16, epochs=5, verbose=1)
    return model


def cross_validation_rnnregressor(algorithm, regressor, X_test, Y_test):
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    def adj_r2_score(r2, n, k):
        return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

    y_pred = regressor.predict(X_test)
    plt.plot(Y_test)
    plt.plot(y_pred)
    print('R-Squared: %f' % (r2_score(Y_test, y_pred)))

    r2_test = r2_score(Y_test, y_pred)
    r2_test = r2_score(Y_test, y_pred)
    print('R-squared is : %f' % r2_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from math import sqrt
    mse = sqrt(mean_squared_error(Y_test, y_pred))
    mae = sqrt(mean_absolute_error(Y_test, y_pred))
    # print(mse)
    # print(mae)
    return mse, mae


X, Y = load_and_clean(
    '/home/sunbeam/Desktop/project/prodata3.csv')
X_train, X_test, Y_train, Y_test = split_data(X, Y)
regressor_LR = build_LR_model(X_train, Y_train)
validation = cross_validation_regressor('Linear Regression', regressor_LR, X_test, Y_test)
regressor_XGB = build_XGB_model(X_train, Y_train, X_test, Y_test)
validation2 = cross_validation_XGBregressor('XG boost', regressor_XGB, X_test, Y_test)
regressor_RF = build_RF_model(X_train, Y_train)
validation3 = cross_validation_RFregressor('Random Forest', regressor_RF, X_test, Y_test)
# regressor_rnn = build_rnn_model(X_train, Y_train, X_test, Y_test)
# validation4 = cross_validation_RFregressor('RNN', regressor_rnn, X_test, Y_test)
