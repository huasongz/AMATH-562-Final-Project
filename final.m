clear all;clc
% load files
mainFolder1 = dir('jaffe');
IMG = [];
LABEL = [];
for i=4:length(mainFolder1)
    name = mainFolder1(i).name;
    LABEL = [LABEL;name(4:5)];
    a = ['jaffe','/',name];
    if strcmp(a,'jaffe/README')
        delete jaffe/README
    else
        img = imread(a);
        [a,b] = size(img);
        img = reshape(img,a*b,1);
        IMG = [IMG img];
    end
end
IMG = double(IMG);
% matrix IMG is the matrix containing all the images and each image is 
% stored in a column vector
% matrix LABEL is the labEls for IMG

% seperating training and testing set
[m,n] = size(IMG);
P = 0.7;
idx = randperm(n);
IMG_train = IMG(:,idx(1:round(P*n))); 
IMG_test = IMG(:,idx(round(P*n)+1:end));
LABEL_train = LABEL(idx(1:round(P*n)),:);
LAEBL_test = LABEL(idx(round(P*n)+1:end),:);

%% svm
svm_mod = fitcecoc(IMG_train.',LABEL_train);
svm_label = predict(svm_mod,IMG_test.');
right = 0;
for i = 1:length(LAEBL_test)
    if strcmp(svm_label(i,:),LAEBL_test(i,:))==1
        right = right + 1;
    end
end
svm_accuracy = right/length(LAEBL_test);
svm_A = ['Accuracy for using SVM is ', num2str(svm_accuracy)];
disp(svm_A)

%% PCA dimension reduction
[u,s,v] = svd(IMG_train,'econ');
figure(4)
plot(diag(s)/sum(diag(s)),'ro','LineWidth',[2])
xlabel('Images')
ylabel('Energy')
title('Singular value spectrum for images')
print(gcf,'-dpng','s_plot.png');

e = diag(s)/sum(diag(s));

threshold = [0.5 0.7 0.9];
sample_idx = [2,50,100];
for j = 1:length(threshold)
    th = threshold(j);
    E = 0;
    for i = 1:length(e)
        E = E + e(i);
        if (E > th)
            break
        end
    end
    recon_faces = u*s(:,1:i)*v(:,1:i)';
    recon_train = u(:,1:i)'*IMG_train;
    recon_test = u(:,1:i)'*IMG_test;
    % reconstruction of faces
    figure(j)
    for i = 1:length(sample_idx)
        subplot(3,2,2*i-1)
        imshow(uint8(reshape(IMG_train(:,sample_idx(i)),256,256)));
        title('Original Image')
        subplot(3,2,2*i)
        imshow(uint8(reshape(recon_faces(:,sample_idx(i)),256,256)));
        title('Reconstructed Image')
        t = ['Reconstructed Image with ', num2str(th*100),' percent energy'];
        suptitle(t)
    end
    
    % SVM after dimension reduction
    svm_mod2 = fitcecoc(recon_train.',LABEL_train);
    svm_label2 = predict(svm_mod2,recon_test.');
    right = 0;
    for i = 1:length(LAEBL_test)
        if strcmp(svm_label2(i,:),LAEBL_test(i,:))==1
            right = right + 1;
        end
    end
    svm_pca_accuracy = right/length(LAEBL_test);
    svm_pca_A = ['Accuracy for using SVM after dimension reduction for a threshold of '...
        ,num2str(th),' is ', num2str(svm_pca_accuracy)];
    disp(svm_pca_A)
    
    % LDA
    lda_mod = fitcdiscr(recon_train.',LABEL_train);
    lda_label = predict(lda_mod,recon_test.');
    right = 0;
    for i = 1:length(LAEBL_test)
        if strcmp(lda_label(i,:),LAEBL_test(i,:))==1
            right = right + 1;
        end
    end
    lda_accuracy = right/length(LAEBL_test);
    lda_A = ['Accuracy for using LDA after dimension reduction for a threshold of '...
        ,num2str(th),' is ', num2str(lda_accuracy)];
    disp(lda_A)
end
