clear all;clc
% load files
mainFolder1 = dir('jaffe');
wavelength = [4,8,16];
orientation = 0:30:180-30;
idx = randperm(213);
P = 0.8;
for j = 1:length(wavelength)
    w = wavelength(j);
    for k = 1:length(orientation)
        o = orientation(k);IMG = [];
        LABEL = [];
        MAG = [];
        PHASE = [];
        for i=4:length(mainFolder1)
            name = mainFolder1(i).name;
            LABEL = [LABEL;name(4:5)];
            a = ['jaffe','/',name];
            
            if strcmp(a,'jaffe/README')
                delete jaffe/README
            else
                img = imread(a);
                [mag,phase] = imgaborfilt(img,w,o);
                [a,b] = size(img);
                %img = reshape(img,a*b,1);
                IMG = [IMG reshape(img,a*b,1)];
                MAG = [MAG reshape(mag,a*b,1)];
                PHASE = [PHASE reshape(phase,a*b,1)];
            end
        end
        IMG = double(IMG);
        
        [m,n] = size(IMG);
        
        IMG_train = IMG(:,idx(1:round(P*n)));
        IMG_test = IMG(:,idx(round(P*n)+1:end));
        LABEL_train = LABEL(idx(1:round(P*n)),:);
        LAEBL_test = LABEL(idx(round(P*n)+1:end),:);
        MAG_train = MAG(:,idx(1:round(P*n)));
        MAG_test = MAG(:,idx(round(P*n)+1:end));
        
        svm_mod = fitcecoc(IMG_train.',LABEL_train);
        svm_label = predict(svm_mod,IMG_test.');
        right = 0;
        for i = 1:length(LAEBL_test)
            if strcmp(svm_label(i,:),LAEBL_test(i,:))==1
                right = right + 1;
            end
        end
        svm_accuracy = right/length(LAEBL_test);
        svm_A = ['Accuracy for directly using SVM is ', num2str(svm_accuracy)];
        disp(svm_A)
        
        svm_mod1 = fitcecoc(MAG_train.',LABEL_train);
        svm_label1 = predict(svm_mod1,MAG_test.');
        right = 0;
        for i = 1:length(LAEBL_test)
            if strcmp(svm_label1(i,:),LAEBL_test(i,:))==1
                right = right + 1;
            end
        end
        svm_accuracy = right/length(LAEBL_test);
        svm_A1 = ['Accuracy for using SVM on filtered images with wavelength ',...
            num2str(w),' and orientation ',num2str(o),' is ', num2str(svm_accuracy)];
        disp(svm_A1)
    end
end





% subplot(1,3,1);
% imshow(reshape(img,256,256));
% title('Original Image');
% subplot(1,3,2);
% imshow(mag,[])
% title('Gabor magnitude');
% subplot(1,3,3);
% imshow(phase,[]);
% title('Gabor phase');



% K = [pi/2,pi/4,pi/8]; % wavenumber
% sigma = pi; % filter bandwidth
% k_v = 0:pi/6:pi; % wavevector orientation
% 
% r0 = zeros(length(img),1);
% for i = 1:length(K)
%     k1 = K(i);
%     for j = 1:length(k_v)-1
%         k2 = k_v(j)
%         g_plus = (k^2/sigma^2)*exp(-(k^2)*(norm(r-r0))^2/(2*sigma^2))*cos(k_v*(r-r0)-exp(-(sigma^2)/2));
%         g_minus = (k^2/sigma^2)*exp(-(k^2)*(norm(r-r0))^2/(2*sigma^2))*sin(k_v*(r-r0));
%     end
% end
% 
% g_plus = (k^2/sigma^2)*exp(-(k^2)*(norm(r-r0))^2/(2*sigma^2))*cos(k_v*(r-r0)-exp(-(sigma^2)/2));
% g_minus = (k^2/sigma^2)*exp(-(k^2)*(norm(r-r0))^2/(2*sigma^2))*sin(k_v*(r-r0));