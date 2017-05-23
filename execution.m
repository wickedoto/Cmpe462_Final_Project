%clc; clear all; close all;
if ~exist('Data/UCI HAR Dataset','file')
    downloadSensorData;
end
%%
if ~exist('rawSensorData_train.mat','file') && ~exist('rawSensorData_test.mat','file')
    saveSensorDataAsMATFiles;
end
%%
load rawSensorData_train
plotRawSensorData(total_acc_x_train, total_acc_y_train, ...
    total_acc_z_train,trainActivity,1000)
%%
rawSensorDataTrain = table(...
    total_acc_x_train, total_acc_y_train, total_acc_z_train, ...
    body_gyro_x_train, body_gyro_y_train, body_gyro_z_train);
%%
humanActivityData = varfun(@Wmean,rawSensorDataTrain);
humanActivityData.activity = trainActivity;
%%
classificationLearner
%%
T_mean = varfun(@Wmean, rawSensorDataTrain);
T_stdv = varfun(@Wstd,rawSensorDataTrain);
T_pca  = varfun(@Wpca1,rawSensorDataTrain);

humanActivityData = [T_mean, T_stdv, T_pca];
humanActivityData.activity = trainActivity;
%%
classificationLearner

load rawSensorData_test
plotRawSensorData(total_acc_x_train, total_acc_y_train, ...
    total_acc_z_train,testActivity,1000)
%%
rawSensorDataTest = table(...
    total_acc_x_test, total_acc_y_test, total_acc_z_test, ...
    body_gyro_x_test, body_gyro_y_test, body_gyro_z_test);

% Step 2: Extract features from raw sensor data
T_mean = varfun(@Wmean, rawSensorDataTest);
T_stdv = varfun(@Wstd,rawSensorDataTest);
T_pca  = varfun(@Wpca1,rawSensorDataTest);

humanActivityData = [T_mean, T_stdv, T_pca];
humanActivityData.activity = testActivity;

% Step 3: Use trained model to predict activity on new sensor data
% Make sure that you've exported 'trainedClassifier' from
% ClassificationLearner
plotActivityResults(trainedClassifier,rawSensorDataTest,humanActivityData,0.1)