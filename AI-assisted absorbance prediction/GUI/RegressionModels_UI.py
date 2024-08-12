import os
import sys
import ast
import time
import threading
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from RegressionModels_form import Ui_Form

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from deepforest import CascadeForestRegressor




class RegressionModels_UI(QtWidgets.QWidget, Ui_Form):
    
    def __init__(self):
        super(RegressionModels_UI, self).__init__()
        self.setupUi(self)


        self.load_dataset.clicked.connect(self.LoadDataset)
        self.model_select.currentTextChanged.connect(self.UpdateModelParam)
        self.pushButton_2.clicked.connect(self.StartThread)
        self.pushButton_3.clicked.connect(self.SelectSaveFolder)

    # Load dataset
    def LoadDataset(self):
        path = os.getcwd() 
        xlsx_file, ok = QtWidgets.QFileDialog.getOpenFileName(self,"Please select dataset", path, "Excel Files (*.xlsx)" ) 
        if ok:
            self.xlsx_file_path = xlsx_file
            folder_path, xlsx_file_name = os.path.split(xlsx_file)
            self.xlsx_file_name = xlsx_file_name
            self.folder_path = folder_path
            self.label_4.setText('Excel file has selected: ' + xlsx_file_name)
            self.pushButton_2.setEnabled(True)

    # Select the path to save the result
    def SelectSaveFolder(self):
        path = os.getcwd() 
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self,"Please select save folder", path, QtWidgets.QFileDialog.ShowDirsOnly ) 
        if dir_name:
            self.label_12.setText('Save folder has selected: ' + dir_name.split('/')[-1])
            self.save_folder = dir_name

            
    # When the user selects a model, update the model's GridSearch parameters into the lineEdit control
    def UpdateModelParam(self):
        if self.model_select.currentText() == 'XGBRegressor':
            self.lineEdit.setText("{'n_estimators': [200, 300, 500]}")
            
        elif self.model_select.currentText() == 'RandomForestRegressor':
            self.lineEdit.setText("{'n_estimators': [250, 350, 500], 'max_depth':[10, 30, 50], 'max_features': [4, 6]}")
            
        elif self.model_select.currentText() == 'SVR':
            self.lineEdit.setText("{'C': [0.1, 1, 5, 10]}")
            
        elif self.model_select.currentText() == 'PLSRegression':
            self.lineEdit.setText("{'n_components': [1, 2, 5, 9]}")
            
        elif self.model_select.currentText() == 'BaggingRegressor':
            self.lineEdit.setText("{'n_estimators': [100, 200, 300, 500]}")
            
        elif self.model_select.currentText() == 'LinearRegression':
            self.lineEdit.setText("No model parameters need to screen")
            
        elif self.model_select.currentText() == 'AdaBoostRegressor':
            self.lineEdit.setText("{'n_estimators': [50, 100, 200], 'learning_rate':[0.1, 0.5, 1]}")
            
        elif self.model_select.currentText() == 'KNeighborsRegressor':
            self.lineEdit.setText("{'weights':['distance'], 'n_neighbors': [5, 10, 30], 'p':[1, 2, 4]}")
            
        elif self.model_select.currentText() == 'MLPRegressor':
            self.lineEdit.setText("{'solver':['lbfgs'], 'activation':['relu'], 'learning_rate_init':[0.001], 'hidden_layer_sizes':[(50,50), (50,20), (50,50,20)], 'max_iter': [100, 200], 'alpha':[0.001, 0.01]}")
            
        elif self.model_select.currentText() == 'ExtraTreeRegressor':
            self.lineEdit.setText("{'max_depth': [3, 10, 20], 'min_samples_leaf':[5, 10, 20], 'min_samples_split':[10, 30, 50]}")
            
        elif self.model_select.currentText() == 'DecisionTreeRegressor':
            self.lineEdit.setText("{'max_depth': [3, 10, 20], 'min_samples_leaf':[5, 10, 20], 'min_samples_split':[10, 30, 50]}")
            
        elif self.model_select.currentText() == 'CascadeForestRegressor':
            self.lineEdit.setText("{'max_layers': [20, 30], 'n_estimators':[2, 10], 'n_trees':[100, 150]}")
            
        else:
            pass


    def StartThread(self):
        self.thread_1 = threading.Thread(target = self.ModelFitAndPredict)
        self.thread_1.setDaemon(True)
        self.thread_1.start()


    def ModelFitAndPredict(self):

        self.pushButton_2.setEnabled(False)
        model_name = self.model_select.currentText()
        print('Model: ' + model_name + '\n')


        ratio_list = self.comboBox.currentText()
        a = ratio_list[1:-1].split(',')

        train_set_ratio = []
        for i in a:
            train_set_ratio.append(float(i))


        print('Loading Dataset...\n')
        df = pd.read_excel(self.xlsx_file_path, sheet_name = int(self.lineEdit_5.text()))

        print('Starting dataset pre-treatment...\n')
        print('Using PCA mode ' + str(self.comboBox_3.currentIndex()+1))
        if self.comboBox_3.currentIndex() == 0:
            df = self.PCA_plan1(df)  
        elif self.comboBox_3.currentIndex() == 1:
            df = self.PCA_plan2(df)
        elif self.comboBox_3.currentIndex() == 2:
            df = self.PCA_plan3(df)

        #self.df_info(df)  # Display information about the dataset

        for ratio in train_set_ratio:

            df2 = df

            print('Starting split dataset...\n')
            train_x, train_y, test_x, test_y, all_X, all_Y = self.data_set_split_and_pretreat(df2, test_size=1-ratio)

            print('Training model(It may takes a long time)...\n')

            if model_name != 'LinearRegression':
                param_grid = ast.literal_eval(self.lineEdit.text())
            else:
                param_grid = ''
            model = self.model_fit(train_x, train_y, param_grid, method = model_name)  # Model training

            y_train_pred = model.predict(train_x)  # Predict using training set
            y_test_pred = model.predict(test_x)  # Predict using test set
            #y_pred = model.predict(all_X)  # Predict using all data

            print('\n>>>>>> Result <<<<<< \n')
            print('Model: ' + model_name + '，Training set proportion: ' + str(ratio) + '，PCA mode: ' + str(self.comboBox_3.currentIndex()+1) + '\n')
            #print("Training set mean square error(MSE)：", mean_squared_error(train_y, y_train_pred), '\n')
            #print("Training set root mean square error(RMSE)：", np.sqrt(mean_squared_error(train_y, y_train_pred)), '\n')
            MAE = mean_absolute_error(test_y, y_test_pred)
            MSE = mean_squared_error(test_y, y_test_pred)
            RMSE = np.sqrt(mean_squared_error(test_y, y_test_pred))
            R2 = r2_score(test_y, y_test_pred)
            print("Test set mean absolute error(MAE)：", MAE, '\n')
            print("Test set mean square error(MSE)：", MSE, '\n')
            print("Test set root mean square error(RMSE)：", RMSE, '\n')
            print("Coefficient of determination(R^2)：", R2, '\n')

            # Save the predicted values ​​and true values ​​of the test set to excel document
            df3 = pd.DataFrame(columns=['Data index', 'Predicted value', 'True value'])
            df3['Data index'] = test_y.index.array + 2
            df3['Predicted value'] = y_test_pred
            df3['True value'] = test_y.values
            df3.to_excel(self.save_folder+'\{0} (Training set proportion {1}, PCA mode {2}, MAE {3}, RMSE {4}, R2 {5}) - Predicted and true values ​​of the test set.xlsx'.format(model_name, ratio, self.comboBox_3.currentIndex()+1, format(MAE, '.6f'), format(RMSE, '.6f'), format(R2, '.6f')), index=False)
            print('Save xlsx file: {0} (Training set proportion {1}, PCA mode {2}, MAE {3}, RMSE {4}, R2 {5}) - Predicted and true values ​​of the test set.xlsx'.format(model_name, ratio, self.comboBox_3.currentIndex()+1, format(MAE, '.6f'), format(RMSE, '.6f'), format(R2, '.6f')))

            # Save the prediction results of all data to excel document
            #df4 = pd.DataFrame(columns=['Data index', 'Predicted value', 'True value'])
            #df4['Data index'] = all_Y.index.array + 1
            #df4['Predicted value'] = y_pred
            #df4['True value'] = all_Y.values
            #df4.to_excel(self.save_folder+'\{0}（Training set proportion {1}, PCA mode {2}） - Predicted and true values ​​of all data.xlsx'.format(model_name, ratio, self.comboBox_3.currentIndex()+1), index=False)
            #print('Save file: {0}（Training set proportion {1}, PCA mode {2}） - Predicted and true values ​​of all data.xlsx'.format(model_name, ratio, self.comboBox_3.currentIndex()+1))

            print('\n\n')

        print('\n\n')

        self.pushButton_2.setEnabled(True)



    # PCA method 1, only perform PCA processing on 40 non-steady-state data
    def PCA_plan1(self, df, n_components = 4):
        X_40 = df.values[:,int(self.lineEdit_3.text()):(int(self.lineEdit_3.text()) + int(self.lineEdit_4.text()))].astype(float)  # non-steady-state data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_40 = scaler.fit_transform(X_40)  # Before performing PCA on the data, perform normalization
        pca = PCA(n_components, random_state = 42) 
        X_p = pca.fit(X_40).transform(X_40)  # Data after PCA dimensionality reduction
        #print('explained_variance_ratio_: ', pca.explained_variance_ratio_, '\n') 
        # Delete the original 40 non-steady-state data and replace it with the dimensionally reduced data
        index = np.arange(int(self.lineEdit_3.text()), (int(self.lineEdit_3.text()) + int(self.lineEdit_4.text())))
        df.drop(df.columns[index], axis=1, inplace=True)
        df.insert(loc=int(self.lineEdit_3.text()), column='P1', value = X_p[:,0])

        return df

    # PCA method 2, Perform PCA processing on all features
    def PCA_plan2(self, df, n_components = 9):
        X = df.values[:,0:(int(self.lineEdit_3.text()) + int(self.lineEdit_4.text()))].astype(float)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)  
        pca = PCA(n_components, random_state = 42) 
        X_p = pca.fit(X).transform(X) 
        #print('explained_variance_ratio_: ',pca.explained_variance_ratio_, '\n') 
        # Keep the first 9 principal components
        # Delete the original data and replace it with the dimensionally reduced data
        index = np.arange(0, (int(self.lineEdit_3.text()) + int(self.lineEdit_4.text())))
        df.drop(df.columns[index], axis=1, inplace=True)
        df.insert(loc=0, column='P1', value = X_p[:,0])
        df.insert(loc=1, column='P2', value = X_p[:,1])
        df.insert(loc=2, column='P3', value = X_p[:,2])
        df.insert(loc=3, column='P4', value = X_p[:,3])
        df.insert(loc=4, column='P5', value = X_p[:,4])
        df.insert(loc=5, column='P6', value = X_p[:,5])
        df.insert(loc=6, column='P7', value = X_p[:,6])
        df.insert(loc=7, column='P8', value = X_p[:,7])
        df.insert(loc=8, column='P9', value = X_p[:,8])

        return df

    # Method 3, No PCA processing
    def PCA_plan3(self, df):

        return df


    # Data set information statistics
    def df_info(self, df):
        print(df.head(), '\n') 
        print(df.describe(), '\n')  


    # Dataset partitioning and preprocessing
    def data_set_split_and_pretreat(self, df, test_size):
        train_set,test_set = train_test_split(df, test_size=test_size, random_state=int(self.lineEdit_7.text()))  # Division of training set and test set

        train_x = train_set.drop('Steady-state absorbance', axis = 1)  # training set，X
        train_y = train_set['Steady-state absorbance'].copy()  # training set，y

        test_x = test_set.drop('Steady-state absorbance', axis = 1)  # test set，X
        test_y = test_set['Steady-state absorbance'].copy()  # test set，y

        all_x = df.drop('Steady-state absorbance', axis = 1) # all data，X
        all_y = df['Steady-state absorbance'].copy()  # all data，y

        # Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_x = scaler.fit_transform(train_x) 
        test_x = scaler.fit_transform(test_x)
        all_x = scaler.fit_transform(all_x)

        return train_x, train_y, test_x, test_y, all_x, all_y


    # Model training
    def model_fit(self, train_x, train_y, param_grid, method = 'RandomForestRegressor'):

        if method == 'SVR':

            model = svm.SVR()
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model


        elif method == 'PLSRegression':

            model = PLSRegression()
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'BaggingRegressor':

            model = BaggingRegressor()
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'LinearRegression':

            # No need to optimize model hyperparameters
            model = LinearRegression()
            model.fit(train_x, train_y)
            return model
            

        elif method == 'KNeighborsRegressor':

            model = KNeighborsRegressor()
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'MLPRegressor':

            model = MLPRegressor(random_state=42)
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'XGBRegressor':

            model = XGBRegressor(verbosity=1)
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'AdaBoostRegressor':

            model = AdaBoostRegressor(random_state=42)
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'ExtraTreeRegressor':

            model = ExtraTreeRegressor(random_state=42)
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'DecisionTreeRegressor':

            model = DecisionTreeRegressor(random_state=42)
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'CascadeForestRegressor':

            model = CascadeForestRegressor(random_state=42)
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model

        elif method == 'RandomForestRegressor':

            forest_reg = RandomForestRegressor(random_state=42)
            # Grid search + k-fold cross validation
            grid_search  = GridSearchCV(forest_reg, param_grid, cv = int(self.lineEdit_6.text()),
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # best model
            return model





if __name__ == '__main__': 


    app = QtWidgets.QApplication(sys.argv)
    window = RegressionModels_UI()
    window.show()
    app.exec_()

    


