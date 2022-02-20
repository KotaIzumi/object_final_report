n = 25;  % 上位n枚をポジティブ画像に
% ポジティブ画像を読み込み
m = 0;
training = {};
DIR='reranking/apple_50/';
W = dir(DIR);
for j = 1:size(W)
    if (strfind(W(j).name,'.jpg'))
        fn = strcat(DIR,W(j).name);
	    m = m+1;
	    training = {training{:} fn};
        if(m == n)
            break;
        end
    end
end


%ネガティブ画像を読み込み
DIR='reranking/bgimg/';
W = dir(DIR);
for j = 1:size(W)
    if (strfind(W(j).name,'.jpg'))
        fn = strcat(DIR,W(j).name);
	    m = m+1;
	    training = {training{:} fn};
    end
end


% テスト画像を読み込み
t = 0;
test = {};
DIR='reranking/apple_300/';
W = dir(DIR);
for j = 1:size(W)
    if (strfind(W(j).name,'.jpg'))
        fn = strcat(DIR,W(j).name);
	    t = t+1;
	    test = {test{:} fn};
    end
end


net = vgg16;


% 学習画像からDCNN特徴を抽出
data_train = [];
for i = 1:m
    img = imread(training{i});
    reimg = imresize(img, net.Layers(1).InputSize(1:2));
    IM = cat(4, reimg);
    dcnnf = activations(net, IM, 'fc7');
    dcnnf = squeeze(dcnnf);
    dcnnf = dcnnf/norm(dcnnf);
    dcnnf = dcnnf';
    data_train= [data_train; dcnnf];
end


% テスト画像からDCNN特徴を抽出
data_test = [];
for i = 1:t
    img = imread(test{i});
    reimg = imresize(img, net.Layers(1).InputSize(1:2));
    IM = cat(4, reimg);
    dcnnf = activations(net, IM, 'fc7');
    dcnnf = squeeze(dcnnf);
    dcnnf = dcnnf/norm(dcnnf);
    dcnnf = dcnnf';
    data_test= [data_test; dcnnf];
end


% 再度分類を行う際にロードする
%load('data_25_train.mat');
%load('data_50_train.mat');


% nが25か50かでtrain_pos,train_negを変更する
train_pos = data_train(1:25,:);
%train_pos = data_train(1:50,:);
train_neg = data_train(26:1025,:);
%train_neg = data_train(51:1050,:);

% 線形SVMで学習
train = [train_pos; train_neg];
train_label = [ones(size(train_pos, 1), 1); ones(size(train_neg, 1),1)*(-1)];
model = fitcsvm(train, train_label, 'KernelFunction','linear');

% テスト画像のスコアを出す
[label,score] = predict(model, data_test);

% スコアをソート
[sorted_score,sorted_idx] = sort(score(:,2),'descend');

% スコアとパスを表示
for i=1:numel(sorted_idx)
  fprintf('%s %f\n',test{sorted_idx(i)},sorted_score(i));
end

