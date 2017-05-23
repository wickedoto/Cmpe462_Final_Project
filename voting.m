load rawSensorData_train

rawSensorDataTrain = table(...
    total_acc_x_train, total_acc_y_train, total_acc_z_train, ...
    body_gyro_x_train, body_gyro_y_train, body_gyro_z_train);

humanActivityData = varfun(@Wmean,rawSensorDataTrain);
humanActivityData.activity = trainActivity;

T_mean = varfun(@Wmean, rawSensorDataTrain);
T_stdv = varfun(@Wstd,rawSensorDataTrain);
T_pca  = varfun(@Wpca1,rawSensorDataTrain);

humanActivityData = [T_mean, T_stdv, T_pca];
humanActivityData.activity = trainActivity;
%%
[a1,b1] = linearSVM(humanActivityData,1);
[a2,b2] = quadraticSVM(humanActivityData,1);
[a3,b3] = cubicSVM(humanActivityData,1);
[a4,b4] = fgaussianSVM(humanActivityData,1);
[a5,b5] = mgaussianSVM(humanActivityData,1);
[a6,b6] = cgaussianSVM(humanActivityData,1);
[a7,b7] = fine_knn_maho(humanActivityData,1);
[a8,b8] = fine_knn_cos(humanActivityData,1);
%%
clear humanActivityData
load rawSensorData_test
rawSensorDataTest = table(...
    total_acc_x_test, total_acc_y_test, total_acc_z_test, ...
    body_gyro_x_test, body_gyro_y_test, body_gyro_z_test);

T_mean = varfun(@Wmean, rawSensorDataTest);
T_stdv = varfun(@Wstd,rawSensorDataTest);
T_pca  = varfun(@Wpca1,rawSensorDataTest);

humanActivityData = [T_mean, T_stdv, T_pca];
humanActivityData.activity = testActivity;
%%
[a11,b11] = linearSVM(humanActivityData,2);
[a21,b21] = quadraticSVM(humanActivityData,2);
[a31,b31] = cubicSVM(humanActivityData,2);
[a41,b41] = fgaussianSVM(humanActivityData,2);
[a51,b51] = mgaussianSVM(humanActivityData,2);
[a61,b61] = cgaussianSVM(humanActivityData,2);
[a71,b71] = fine_knn_maho(humanActivityData,2);
[a81,b81] = fine_knn_cos(humanActivityData,2);
%%
c1 = a11.ClassificationSVM.predict(humanActivityData);
c2 = a21.ClassificationSVM.predict(humanActivityData);
c3 = a31.ClassificationSVM.predict(humanActivityData);
c4 = a41.ClassificationSVM.predict(humanActivityData);
c5 = a51.ClassificationSVM.predict(humanActivityData);
c6 = a61.ClassificationSVM.predict(humanActivityData);
c7 = a71.ClassificationKNN.predict(humanActivityData);
c8 = a81.ClassificationKNN.predict(humanActivityData);
%%
result = [grp2idx(c2),grp2idx(c3),grp2idx(c5)];
result_c = [c2,c3,c5];
weight_vector = [b2,b3,b5];
weight_vector_normalized = weight_vector/sum(weight_vector);
turn = bsxfun(@minus,result,t_p);
turn(turn == 0) = -1;
turn(turn ~= -1) = 0;
turn(turn == -1) = 1;
voted_results = turn * weight_vector_normalized';
voted_results(voted_results < 0.5) = 0;
accuracy =  sum(voted_results) / length(voted_results) * 100;
modes = mode(result,2);
confusion_prediction = modes;
true_predicts = logical(voted_results == 1);
confusion_prediction(true_predicts) = t_p(true_predicts);
disp('3 SVMs');
disp(accuracy);
disp('confusion matrix is:')
conf_mat = confusionmat(confusion_prediction,t_p);
disp(conf_mat);
%%
result = [grp2idx(c2),grp2idx(c3),grp2idx(c5),grp2idx(c7)];
result_c = [c2,c3,c5,c7];
weight_vector = [b2,b3,b5,b7];
weight_vector_normalized = weight_vector/sum(weight_vector);
turn = bsxfun(@minus,result,t_p);
turn(turn == 0) = -1;
turn(turn ~= -1) = 0;
turn(turn == -1) = 1;
voted_results = turn * weight_vector_normalized';
voted_results(voted_results < 0.5) = 0;
accuracy =  sum(voted_results) / length(voted_results) * 100;
modes = mode(result,2);
confusion_prediction = modes;
true_predicts = logical(voted_results == 1);
confusion_prediction(true_predicts) = t_p(true_predicts);
disp('3 SVM, 1 KNN');
disp(accuracy);
disp('confusion matrix is:')
conf_mat = confusionmat(confusion_prediction,t_p);
disp(conf_mat);
%%
t_p = grp2idx(humanActivityData.activity);
t_p_c = humanActivityData.activity;
%result = [grp2idx(c1),grp2idx(c2),grp2idx(c3),grp2idx(c4),grp2idx(c5),grp2idx(c6)];
result = [grp2idx(c2),grp2idx(c3),grp2idx(c5),grp2idx(c7),grp2idx(c8)];
modes = mode(result);
result_c = [c2,c3,c5,c7,c8];
weight_vector = [b2,b3,b5,b7,b8];
weight_vector_normalized = weight_vector/sum(weight_vector);
turn = bsxfun(@minus,result,t_p);
turn(turn == 0) = -1;
turn(turn ~= -1) = 0;
turn(turn == -1) = 1;
voted_results = turn * weight_vector_normalized';
voted_results(voted_results < 0.5) = 0;
accuracy =  sum(voted_results) / length(voted_results) * 100;
modes = mode(result,2);
confusion_prediction = modes;
true_predicts = logical(voted_results == 1);
confusion_prediction(true_predicts) = t_p(true_predicts);
disp(' 3 SVM ,2 KNN');
disp(accuracy);
disp('confusion matrix is:')
conf_mat = confusionmat(confusion_prediction,t_p);
disp(conf_mat);

