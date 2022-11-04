clear; close all; clc

[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
% reshape each image
for i = 1:length(labels)
    imag(:,i) = reshape(images(:,:,i),784,1);
end
% subtracting the row wise mean
imag1 = double(imag);
for j = 1:size(imag1,1)
    imag1(j,:) = imag1(j,:) - mean(imag1(j,:));
end
%  SVD analysis of the digit images
[U,S,V] = svd(imag1,'econ');
sig1 = diag(S);

%% Plot singular value spectrum
figure(1)
subplot(2,1,1)
plot(diag(S),'ko','Linewidth',1.5)
title("Training-data: First 200 Nonzero Singular Values");
set(gca,'Fontsize',12,'Xlim',[0 200])
xlabel("Singular Values"); ylabel("Energy");
subplot(2,1,2)
plot(cumsum(sig1.^2/sum(sig1.^2)),'ko','Linewidth',1.5)
title("Training-data: Cumulative Energy Captured");
set(gca,'Fontsize',12,'Xlim',[0 800])
xlabel("Singular Values"); ylabel("Cumulative Energy");
hold on
plot([0 800], [0.85 0.85], '--r', 'Linewidth',1.5)
plot([0 800], [0.90 0.90], '--g', 'Linewidth',1.5)
plot([0 800], [0.95 0.95], '--b', 'Linewidth',1.5)
legend('Cum Energy Captured','85% Energy Captured','90% Energy Captured','95% Energy Captured')
 
%% Finding # of modes necessary for good image reconstruction 
% Capture at least 85% Engery
capture_percent = 0.85;
energy1 = 0;
nodes1 = 0;
while energy1 < capture_percent
    nodes1 = nodes1 + 1;
    sig1 = diag(S);  
    % get each singular value
    current_var1 = sig1(nodes1); 
    % energy captured by each nonzero singular value
    each_energy1 = current_var1.^2/sum(sig1.^2);
    % find cumulative energy for first n-modes
    energy1 = energy1 + each_energy1;
end
disp(nodes1)
% Capture at least 90% Engery
capture_percent = 0.90;
energy2 = 0;
nodes2 = 0;
while energy2 < capture_percent
    nodes2 = nodes2 + 1;
    sig2 = diag(S);  
    current_var2 = sig2(nodes2); 
    each_energy2 = current_var2.^2/sum(sig2.^2);
    energy2 = energy2 + each_energy2;
end
disp(nodes2)
% Capture at least 95% Engery
capture_percent = 0.95;
energy3 = 0;
nodes3 = 0;
while energy3 < capture_percent
    nodes3 = nodes3 + 1;
    sig3 = diag(S);  
    % get each singular value
    current_var3 = sig3(nodes3); 
    % energy captured by each nonzero singular value
    each_energy3 = current_var3.^2/sum(sig3.^2);
    % find cumulative energy for first n-modes
    energy3 = energy3 + each_energy3;
end
disp(nodes3)

%% Plot first image Digit 0-9's reconstruction image and with 85,90,95% Energy
figure(2)
for k = 1:10
    % plot original digit 0-9
    subplot(4,10,k)
    label_index = find(labels == k-1,1);
    imshow(images(:,:,label_index))
    sgtitle({'Row1: First Original 10 Digits Images from Train-Images Data',...
        'Row2: Reconstruction Images with 85% Energy Capture',...
        'Row3: Reconstruction Images with 90% Energy Capture',...
        'Row4: Reconstruction Images with 95% Energy Capture'},'fontweight','bold','FontSize',16)   
    % plot original digit 0-9 with 85% energy capture
    imag_est_85 = U(:,1:nodes1)*S(1:nodes1,1:nodes1)*V(:,1:nodes1)';
    subplot(4,10,10+k)
    imag_est_85 = reshape(imag_est_85(:,label_index),28,28);
    imshow(imag_est_85)
    % plot original digit 0-9 with 90% energy capture
    imag_est_90 = U(:,1:nodes2)*S(1:nodes2,1:nodes2)*V(:,1:nodes2)';
    subplot(4,10,20+k)
    imag_est_90 = reshape(imag_est_90(:,label_index),28,28);
    imshow(imag_est_90)
    % plot original digit 0-9 with 95% energy capture
    imag_est_95 = U(:,1:nodes3)*S(1:nodes3,1:nodes3)*V(:,1:nodes3)';
    subplot(4,10,30+k)
    imag_est_95 = reshape(imag_est_95(:,label_index),28,28);
    imshow(imag_est_95)
end
 
%% Interpretation of the U a matrices
% Interpretation of the U
figure(3)
for k = 1:9
   subplot(3,3,k)
   ut1 = reshape(U(:,k),28,28);
   ut2 = rescale(ut1);
   imshow(ut2)
   sgtitle('First Nine Principal Components','fontweight','bold')
end

%%  Projection onto three selected V-modes colored by their digit label
figure(5)
proj = S*V';  %project onto three selected V-modes
for i=1:7
    digit = i-1;
    str ='digit:' + string(digit);
    label_indices = find(labels == digit);
    plot3(proj(2, label_indices), proj(3, label_indices), proj(5, label_indices),'o', ...
    'DisplayName', str, 'Linewidth', 2)
    hold on
end
% Manullay plot last 3 digits due to there's only 7 default color for plot3
label_indices = find(labels == 7);
plot3(proj(2, label_indices), proj(3, label_indices), proj(5, label_indices),...
'o', 'DisplayName', 'digit:7', 'Linewidth', 2, 'Color',[1,1,0])
label_indices = find(labels == 8);
plot3(proj(2, label_indices), proj(3, label_indices), proj(5, label_indices),...
     'o', 'DisplayName', 'digit:8', 'Linewidth', 2, 'Color',[0 0 0])
label_indices = find(labels == 9);
plot3(proj(2, label_indices), proj(3, label_indices), proj(5, label_indices),...
     'o', 'DisplayName', 'digit:9', 'Linewidth', 2, 'Color',[1 0 0])
xlabel('2nd V-Mode'), ylabel('3rd V-Mode'), zlabel('5th V-Mode')
title('Projection onto V-modes 2, 3, 5')
legend
set(gca,'Fontsize', 14)

%% linear classifier (LDA) that identify 2 digits
[labels,I] = sort(labels);
imag1 = imag1(:,I');

%  SVD analysis of the digit images
[U,S,V] = svd(imag1,'econ');
sig1 = diag(S);
feature = nodes3;     % use first 59 nodes capture 85% energy
imag_proj = S(1:feature,1:feature)*V(:,1:feature)';
error_list = [];

% load test data: linear classifier (LDA) that identify 2 digits
[test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
% reshape each image
for i = 1:length(test_labels)
    imag_test(:,i) = reshape(test_images(:,:,i),784,1);
end
% subtracting the row wise mean
imag1_test = double(imag_test);
for j = 1:size(imag1_test,1)
    imag1_test(j,:) = imag1_test(j,:) - mean(imag1_test(j,:));
end
[test_labels,Ind] = sort(test_labels);
imag1_test= imag1_test(:,Ind');

%  SVD analysis of the test digit images
[U_test,S_test,V_test] = svd(imag1_test,'econ');
imag_proj_test = S_test(1:feature,1:feature)*V_test(:,1:feature)';

%% One Example for 2 digits and 3 digits
figure(6)
% 2 Digits Example: digit 0 and 1
label_first_indices = find(labels == 0);  % find all first digit images in train data
digit_first = imag_proj(:,label_first_indices);
label_second_indices = find(labels == 1);  % find all second digit images in train data
digit_second = imag_proj(:,label_second_indices);
num_first = length(label_first_indices);  % # num of first digit images in train data
num_second = length(label_second_indices);  % # num of second digit images in train data
xtrain = [digit_first'; digit_second'];
ctrain = [2*ones(num_first,1); ones(num_second,1)];
test_label_first_indices = find(test_labels == 0);  % find all first digit images in test data
test_digit_first = imag_proj_test(:,test_label_first_indices);
test_label_second_indices = find(test_labels == 1);  % find all second digit images in test data
test_digit_second = imag_proj_test(:,test_label_second_indices); 
xtest = [test_digit_first'; test_digit_second'];
[prediction] = classify(xtest,xtrain,ctrain);
subplot(2,1,1)
bar(prediction)
set(gca,'Fontsize',12)
title('Identification of Digits 0 and 1 in Test Data','fontweight','bold','FontSize',14)

% 3 Digits Example: digit 0, 1 and 2
label_first_indices = find(labels == 0);  % find all first digit images in train data
digit_first = imag_proj(:,label_first_indices);
label_second_indices = find(labels == 1);  % find all second digit images in train data
digit_second = imag_proj(:,label_second_indices);
label_third_indices = find(labels == 2);  % find all second digit images in train data
digit_third = imag_proj(:,label_third_indices);
num_first = length(label_first_indices);  % # num of first digit images in train data
num_second = length(label_second_indices);  % # num of second digit images in train data
num_third = length(label_third_indices);  % # num of second digit images in train data
xtrain = [digit_first'; digit_second'; digit_third'];
ctrain = [2*ones(num_first,1); ones(num_second,1); 4*ones(num_third,1)];
test_label_first_indices = find(test_labels == 0);  % find all first digit images in test data
test_digit_first = imag_proj_test(:,test_label_first_indices);
test_label_second_indices = find(test_labels == 1);  % find all second digit images in test data
test_digit_second = imag_proj_test(:,test_label_second_indices); 
test_label_third_indices = find(test_labels == 2);  % find all second digit images in test data
test_digit_third = imag_proj_test(:,test_label_third_indices); 
xtest = [test_digit_first'; test_digit_second'; test_digit_third'];
[prediction] = classify(xtest,xtrain,ctrain);
subplot(2,1,2)
bar(prediction)
set(gca,'Fontsize',12)
title('Identification of Digits 0, 1, 2 in Test Data','fontweight','bold','FontSize',14)

%% Run through all combination of two digits
comb_2_digits = combntns(0:9,2);
for i = 1:45
    % Train data
    digit_1st = comb_2_digits(i,1);
    digit_2nd = comb_2_digits(i,2);
    label_first_indices = find(labels == digit_1st);  
    digit_first = imag_proj(:,label_first_indices);
    label_second_indices = find(labels == digit_2nd); 
    digit_second = imag_proj(:,label_second_indices);
    num_first = length(label_first_indices); 
    num_second = length(label_second_indices);  
    xtrain = [digit_first'; digit_second'];
    ctrain = [ones(num_first,1); 2*ones(num_second,1)];
    % Test data
    test_label_first_indices = find(test_labels == digit_1st); 
    test_digit_first = imag_proj_test(:,test_label_first_indices);
    test_label_second_indices = find(test_labels == digit_2nd);  
    test_digit_second = imag_proj_test(:,test_label_second_indices); 
    xtest = [test_digit_first'; test_digit_second'];
    [prediction,err] = classify(xtest,xtrain,ctrain); 
    error_list(1,i) = err;
end

%% Plot all All Combinations Error Rate
[Max_Err,I1] = max(error_list); % 0.04275
[Min_Err,I2] = min(error_list); % 0.00173
figure(7)
plot(1:45,100.*error_list,'-ko', 'Linewidth', 2)
ylabel('Misclassification Error Rate(%)')
xlabel('Combination of Two Digits')
hold on
plot(I1,100.*Max_Err,'r*','Linewidth', 4)
plot(I2,100.*Min_Err,'b*','Linewidth', 4)
legend('All Combinations','Most Difficult to Separate(3&5)','Most Easy to Separate (6&7)')
set(gca,'Fontsize',14)
title('Misclassification Error Rate with First 59 Features','fontweight','bold','FontSize',16)

%%  Decision Tree Classifier
data = imag_proj';
tree = fitctree(data,labels);
% Demo of tree plot for 30 MaxNumSplits
tree1=fitctree(data,labels,'MaxNumSplits',30,'CrossVal','on');
view(tree1.Trained{1},'Mode','graph');

%% Decision Tree-Traning Data Accuracy between all ten digits 0.9836 
train_labels_tree = predict(tree,data);
counter = 0;
train_labels = labels;
train_labels_tree = sort(train_labels_tree);
for i = 1:length(train_labels_tree)
    pre = train_labels_tree(i,1);
    actual = train_labels(i,1);
    if pre == actual
        counter = counter + 1;
    end
end

%% most easy + most hard  
easy_digit_6_num = length(find(train_labels_tree == 6));
easy_digit_7_num = length(find(train_labels_tree == 7));
total_num_easy = easy_digit_6_num + easy_digit_7_num;
data_total_num_easy = length(find(labels == 6)) + length(find(labels == 7));
easy_acc = abs(total_num_easy-data_total_num_easy) / data_total_num_easy;
hard_digit_3_num = length(find(train_labels_tree == 3));
hard_digit_5_num = length(find(train_labels_tree == 5));
total_num_hard = hard_digit_3_num + hard_digit_5_num;
data_total_num_hard = length(find(labels == 3)) + length(find(labels == 5));
hard_acc = abs(total_num_hard-data_total_num_hard) / data_total_num_hard;

%% Decision Tree-Testing Data Accuracy   
test_data = imag_proj_test';
test_labels_tree = predict(tree,test_data);
counter_test = 0;
test_labels_tree = sort(test_labels_tree);
for i = 1:length(test_labels_tree)
    pre = test_labels_tree(i,1);
    actual = test_labels(i,1);
    if pre == actual
        counter_test = counter_test + 1;
    end
end

%% SVM Classifier
% keep data small for computing efficienty, by the inverse of the largest singular value.
data_1 = imag_proj' ./ max(diag(S)); 
Mdl = fitcecoc(data_1,labels);
% SVM-Traning Data Accuracy between all ten digits 
train_labels_SVM = predict(Mdl,data_1);
counter = 0;
train_labels = labels;
train_labels_SVM = sort(train_labels_SVM);
for i = 1:length(train_labels_SVM)
    pre = train_labels_SVM(i,1);
    actual = train_labels(i,1);
    if pre == actual
        counter = counter + 1;
    end
end
% svm-Testing Data Accuracy    
test_labels_SVM = predict(Mdl,test_data);
counter_test = 0;
test_labels_SVM = sort(test_labels_SVM);
for i = 1:length(test_labels_SVM)
    pre = test_labels_SVM(i,1);
    actual = test_labels(i,1);
    if pre == actual
        counter_test = counter_test + 1;
    end
end
easy_digit_6_num = length(find(train_labels_SVM == 6));
easy_digit_7_num = length(find(train_labels_SVM == 7));
total_num_easy = easy_digit_6_num + easy_digit_7_num;
data_total_num_easy = length(find(labels == 6)) + length(find(labels == 7));
easy_acc = abs(total_num_easy-data_total_num_easy) / data_total_num_easy;
hard_digit_3_num = length(find(train_labels_SVM == 3));
hard_digit_5_num = length(find(train_labels_SVM == 5));
total_num_hard = hard_digit_3_num + hard_digit_5_num;
data_total_num_hard = length(find(labels == 3)) + length(find(labels == 5));
hard_acc = abs(total_num_hard-data_total_num_hard) / data_total_num_hard;
