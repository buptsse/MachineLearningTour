function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values

m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1) X];
% for i=1:m
% 	Intwo = (Theta1*transpose(X(i,:)));
% 	Inthree= Theta2*sigmoid([1; Intwo]);
% 	output =transpose(sigmoid(Inthree));
% 	[tt,I]=max(output,[],2);
% 	p(i,1)=I;
% end

p_lay1 = zeros(m,size(Theta2, 2)-1);

for i=1 : m 
	sampleInDifferentKindPredictions1 = X(i,:)*transpose(Theta1);
	p_lay1(i,:) = sigmoid(sampleInDifferentKindPredictions1);
end

X_2 = [ones(m, 1) p_lay1];

for j=1 : m
	sampleInDifferentKindPredictions2 = X_2(j,:)*transpose(Theta2);
	[maxPredict2,position2]=max(sampleInDifferentKindPredictions2);
	p(j) = position2;
end







% =========================================================================


end
