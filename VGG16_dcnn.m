%組み合わせを変える場合は，LIST,DIR0を変更する．
%画像は1クラス100枚
n = 0;
list = {};
LIST={'airplane', 'helicopter'};
DIR0='vehicle/';
%LIST={'lion', 'tiger'};
%DIR0='cat/';
for i = 1:length(LIST)
    DIR = strcat(DIR0,LIST(i),'/');
    W = dir(DIR{:});

    for j = 1:size(W)
      if (strfind(W(j).name,'.jpg'))
        fn = strcat(DIR{:},W(j).name);
	    n = n+1;
	    list = {list{:} fn};
      end
    end
end


% 全ての画像の特徴抽出
net = vgg16;
data = [];


for i = 1:200
    img = imread(list{i});
    reimg = imresize(img, net.Layers(1).InputSize(1:2));
    IM = cat(4, reimg);
    dcnnf = activations(net, IM, 'fc7');
    dcnnf = squeeze(dcnnf);
    dcnnf = dcnnf/norm(dcnnf);
    dcnnf = dcnnf';
    data = [data; dcnnf];
end


% 再度分類を行う際にロードする
%load("dcnnf_vehicle.mat");
%load("dcnnf_cat.mat");


% 抽出した特徴をクラスごとに分ける
data_pos = data(1:100,:);
data_neg = data(101:200,:);


n = 100;
cv = 5;
idx = [1:n];
accuracy = [];

% 5-fold cross validation
for i = 1:cv
    train_pos = data_pos(find(mod(idx,cv)~=(i-1)),:);
    eval_pos = data_pos(find(mod(idx,cv)==(i-1)),:);
    train_neg = data_neg(find(mod(idx,cv)~=(i-1)),:);
    eval_neg = data_neg(find(mod(idx,cv)==(i-1)),:);

    train = [train_pos; train_neg];
    eval = [eval_pos; eval_neg];

    train_label = [ones(size(train_pos, 1), 1); ones(size(train_neg, 1),1)*(-1)];
    eval_label = [ones(size(eval_pos, 1), 1); ones(size(eval_neg, 1),1)*(-1)];
    
    % 線形SVMで分類
    model = fitcsvm(train, train_label, 'KernelFunction','linear');
    [plabel, scores] = predict(model, eval);
    ac = numel(find(eval_label==plabel))/numel(eval_label);
    accuracy = [accuracy ac];
end

fprintf('accuracy: %f\n', mean(accuracy))