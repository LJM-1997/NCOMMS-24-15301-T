import pandas as pd
import numpy as np


import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#from sklearn.model_selection import train_test_split
#from sklearn import svm
#from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, ShuffleSplit
#from sklearn.tree import ExtraTreeRegressor
#from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor 



def PCA_mode_1(df, n_components = 4):
    X_40 = df.values[:,8:48].astype(float)  # non-steady-state data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_40 = scaler.fit_transform(X_40)  # Before performing PCA on the data, perform normalization
    pca = PCA(n_components, random_state = 42) 
    X_p = pca.fit(X_40).transform(X_40)  # Data after PCA dimensionality reduction
    # Delete the original 40 non-steady-state data and replace it with the dimensionally reduced data
    index = np.arange(8, 48)
    df.drop(df.columns[index], axis=1, inplace=True)
    df.insert(loc=8, column='P1', value = X_p[:,0])

    return df


def model_fit(train_x, train_y, method = 'xgb'):

    if method == 'xgb':
        model = XGBRegressor(n_estimators=500) # ,verbosity=1
        model.fit(train_x, train_y)

    return model  



if __name__ == "__main__":

    # Load dataset
    # Please replace the file path here with your own path
    df = pd.read_excel("./dataset.xlsx", sheet_name = 5) 

    # Data preprocessing
    df = PCA_mode_1(df) 
    X = df.drop('Steady-state absorbance', axis = 1).values 
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    X = scaler_1.fit_transform(X)
    y = df['Steady-state absorbance'].values  

    # Perform 10-fold cross validation
    kf = KFold(n_splits=10) 
    X, y = sklearn.utils.shuffle(X, y, random_state=42) 

    MAE_list = []
    MSEP_list = []
    RMSEP_list = []
    R2_list = []

    for train_index,test_index in kf.split(X):
        print("Train Index:",test_index,",Test Index:",train_index)
        X_test, X_train = X[train_index],X[test_index]
        y_test, y_train = y[train_index],y[test_index]

        model = model_fit(X_train, y_train, method = 'xgb')  
        y_test_pred = model.predict(X_test)  

        MAE = mean_absolute_error(y_test, y_test_pred)
        MSE = mean_squared_error(y_test, y_test_pred)
        RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))
        R2 = r2_score(y_test, y_test_pred)
        print("Test set mean absolute error(MAE)：", MAE, '\n')
        print("Test set mean square error(MSE)：", MSE, '\n')
        print("Test set root mean square error(RMSE)：", RMSE, '\n')
        print("Coefficient of determination(R^2)：", R2, '\n')

        MAE_list.append(MAE)
        MSEP_list.append(MSE)
        RMSEP_list.append(RMSE)
        R2_list.append(R2)

    
    MAE_mean = sum(MAE_list) / len(MAE_list)
    MSEP_mean = sum(MSEP_list) / len(MSEP_list)
    RMSEP_mean = sum(RMSEP_list) / len(RMSEP_list)
    R2_mean = sum(R2_list) / len(R2_list)

    MAE_std_dev = np.std(np.array(MAE_list), ddof=1)
    MSEP_std_dev = np.std(np.array(MSEP_list), ddof=1)
    RMSEP_std_dev = np.std(np.array(RMSEP_list), ddof=1)
    R2_std_dev = np.std(np.array(R2_list), ddof=1)

    MAE_CV = MAE_std_dev/MAE_mean
    MSEP_CV = MSEP_std_dev/MSEP_mean
    RMSEP_CV = RMSEP_std_dev/RMSEP_mean
    R2_CV = R2_std_dev/R2_mean

    print('MAE_mean: ' + str(MAE_mean))
    print('MSE_mean: ' + str(MSEP_mean))
    print('RMSE_mean: ' + str(RMSEP_mean))
    print('R2_mean: ' + str(R2_mean))

    print('MAE_std_dev: ' + str(MAE_std_dev))
    print('MSE_std_dev: ' + str(MSEP_std_dev))
    print('RMSE_std_dev: ' + str(RMSEP_std_dev))
    print('R2_std_dev: ' + str(R2_std_dev))

    print('MAE_CV: ' + str(MAE_CV))
    print('MSE_CV: ' + str(MSEP_CV))
    print('RMSE_CV: ' + str(RMSEP_CV))
    print('R2_CV: ' + str(R2_CV))



