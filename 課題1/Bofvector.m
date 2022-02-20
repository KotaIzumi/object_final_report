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


% 特徴点抽出
Features=[];
for i = 1:200
    I = rgb2gray(imread(list{i}));
    p = createRandomPoints(I,300);
    [f, p2] = extractFeatures(I, p);
    Features=[Features; f];
end

Features = Features(randperm(size(Features, 1), 50000), :);

% k-means法で代表ベクトルを選ぶ
k = 500;
[idx, CODEBOOK] = kmeans(Features, k);

% Bofベクトルを作成
n = 200;

bof = zeros(n, k);
for j = 1:n
   I = rgb2gray(imread(list{j}));
   p = createRandomPoints(I,1000);
   [f, p2] = extractFeatures(I, p);
   for i = 1:size(p2, 1)
        f2 = repmat(f(i,:), k, 1);
        tmp = (CODEBOOK - f2).^2;
        sum_tmp = sqrt(sum(tmp'));
        [min_ans, index] = min(sum_tmp);
        bof(j, index) = bof(j, index) + 1;
   end
end
bof = bof./sum(bof, 2);


% 再度分類を行う際にロードする
%load('bof_vehicle.mat');
%load('bof_cat.mat');


% 抽出した特徴をクラスごとに分ける
data_pos = bof(1:100,:);
data_neg = bof(101:200,:);


% 5-fold cross validation
n = 100;
cv = 5;
idx = [1:n];
accuracy = [];
for i = 1:cv
    train_pos = data_pos(find(mod(idx,cv)~=(i-1)),:);
    eval_pos = data_pos(find(mod(idx,cv)==(i-1)),:);
    train_neg = data_neg(find(mod(idx,cv)~=(i-1)),:);
    eval_neg = data_neg(find(mod(idx,cv)==(i-1)),:);

    train = [train_pos; train_neg];
    eval = [eval_pos; eval_neg];

    train_label = [ones(size(train_pos, 1), 1); ones(size(train_neg, 1),1)*(-1)];
    eval_label = [ones(size(eval_pos, 1), 1); ones(size(eval_neg, 1),1)*(-1)];

    %非線形SVMで分類
    model=fitcsvm(train, train_label,'KernelFunction','rbf', 'KernelScale','auto');
    [plabel, scores] = predict(model, eval);
    ac = numel(find(eval_label==plabel))/numel(eval_label);
    accuracy = [accuracy ac];

end
fprintf('accuracy: %f\n', mean(accuracy))

