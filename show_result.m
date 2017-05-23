load after_calc;
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
%weight_vector_normalized = [0.2,0.2,0.2,0.2,0.2];
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

