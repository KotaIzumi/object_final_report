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



% 全画像からカラーヒストグラムを作成
database = [];
for i = 1:length(list)
    X = imread(list{i});
    RED = X(:,:,1); GREEN = X(:,:,2); BLUE = X(:,:,3);
    X64 = floor(double(RED)/64) *4*4 + floor(double(GREEN)/64) *4 + floor(double(BLUE)/64);
    X64_vec = reshape(X64,1,numel(X64));
    h = histc(X64_vec,[0:63]);
    h = h / sum(h);
    database = [database; h];
end

% 再度分類を行う際にロードする
%load('colorhist_vehicle.mat');
%load('colorhist_cat.mat');

% 抽出した特徴をクラスごとに分ける
data_pos = database(1:100,:);
data_neg = database(101:200,:);

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

    %線形SVMで分類
    model = fitcsvm(train, train_label, 'KernelFunction','linear');
    [plabel, scores] = predict(model, eval);
    ac = numel(find(eval_label==plabel))/numel(eval_label);
    accuracy = [accuracy ac];

end
fprintf('accuracy: %f\n', mean(accuracy))