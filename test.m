clear all;clc
% load files
mainFolder1 = dir('jaffe');
w = 4;
o = 30;
IMG = [];
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
        [a,b] = size(img);
        [mag,phase] = imgaborfilt(img,w,o);
        IMG = [IMG reshape(img,a*b,1)];
        MAG = [MAG reshape(mag,a*b,1)];
        PHASE = [PHASE reshape(phase,a*b,1)];
    end
end
IMG = double(IMG);

%%
[m,n] = size(IMG);
P = 0.7;
idx = randperm(n);
IMG_train = IMG(:,idx(1:round(P*n))); 
IMG_test = IMG(:,idx(round(P*n)+1:end));
LABEL_train = LABEL(idx(1:round(P*n)),:);
LABEL_test = LABEL(idx(round(P*n)+1:end),:);
MAG_train = MAG(:,idx(1:round(P*n))); 
MAG_test = MAG(:,idx(round(P*n)+1:end));

%%
L = [];
for i = 1:length(LABEL)
    if strcmp(LABEL(i,:),'AN')
        L(i) = 1;
    elseif strcmp(LABEL(i,:),'SA')
        L(i) = 2;
    elseif strcmp(LABEL(i,:),'HA')
        L(i) = 3;
    elseif strcmp(LABEL(i,:),'DI')
        L(i) = 4;
    elseif strcmp(LABEL(i,:),'FE')
        L(i) = 5;
    elseif strcmp(LABEL(i,:),'SU')
        L(i) = 6;
    elseif strcmp(LABEL(i,:),'NE')
        L(i) = 7;
    end
end
LABEL_train = L(idx(1:round(P*n)));
LABEL_test = L(idx(round(P*n)+1:end));
%%
[U,S,V] = svd(MAG_train,'econ');
D = diag(S);
[U1,S1,V1] = svd(MAG_test,'econ');
D1 = diag(S1);
%%
% net = perceptron;
% net = configure(net,D,LABEL_train');
% y = net(D);
net = feedforwardnet(10);
%options = trainingOptions('ExecutionEnvironment','multi-gpu');
[net,tr] = train(net,D,LABEL_train');
y = round(net(D));

%YPred = predict(net,D1)
%y1 = round(sim(trained_net,D1));

%% 
right = 0;
for i = 1:length(y)
    if y(i) == LABEL_train(i)
        right = right + 1;
    end
end