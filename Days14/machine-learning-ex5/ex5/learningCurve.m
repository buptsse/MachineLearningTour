function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------






	trainSamplsX = X(1:i,:);
	trainSamplsY = y(1:i);
	%训练对应的theta,这个地方lambda不能为0
	theta = trainLinearReg(trainSamplsX,trainSamplsY,lambda);
	%计算对应的cost,这个地方 lambda 需要设置为0
	%这个地方lambda必须为0,因为是要根据 结果来挑选要哪个lamba,lambda 不能影响到这个决定因子.
	[J, grad] = linearRegCostFunction(trainSamplsX, trainSamplsY, theta, 0);

	error_train(i) = J;

	%计算对应的cost,这个地方 lambda 需要设置为0	
	%这个地方lambda必须为0,因为是要根据 结果来挑选要哪个lamba,lambda 不能影响到这个决定因子.
	[J_val, grad_val] = linearRegCostFunction(Xval, yval, theta, 0);

	error_val(i) = J_val;



% -------------------------------------------------------------

% =========================================================================

end






%for i=1:m,

%	cost = 0;
%	cost2 = 0;

%	for j=1:50,
%		%每次随机取m个样例
%		sampleIndexs = randperm(m,i);
%		trainSamplsX = X(sampleIndexs,:);
%		trainSamplsY = y(sampleIndexs,:);
%		%训练对应的theta,这个地方lambda不能为0
%		theta = trainLinearReg(trainSamplsX,trainSamplsY,lambda);
%		%计算对应的cost,这个地方 lambda 需要设置为0
%		%这个地方lambda必须为0,因为是要根据 结果来挑选要哪个lamba,lambda 不能影响到这个决定因子.
%		[J, grad] = linearRegCostFunction(trainSamplsX, trainSamplsY, theta, 0);
%		cost = cost + J;


%		testIndexs = randperm(size(Xval,1),i);

%			%计算对应的cost,这个地方 lambda 需要设置为0	
%	%这个地方lambda必须为0,因为是要根据 结果来挑选要哪个lamba,lambda 不能影响到这个决定因子.
%	[J_val, grad_val] = linearRegCostFunction(Xval(testIndexs,:), yval(testIndexs,:), theta, 0);
%			cost2 = cost2 + J_val;

%	end
%

	
%	error_train(i) = cost/50;



%	error_val(i) = cost2/50;
%end
