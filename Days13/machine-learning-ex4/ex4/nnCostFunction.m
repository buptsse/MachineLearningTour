function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




%----------------------------------Part 1--------------------------------------------

Layer1_output = [ones(m,1),X];
Layer2_output_exclude_bias =  sigmoid(Layer1_output*transpose(Theta1));
Layer2_output_include_bias =  [ones(m,1),Layer2_output_exclude_bias];
Layer3_output= sigmoid(Layer2_output_include_bias*transpose(Theta2));

J = 0;

for i=1:m,
	% 遍历每一个sample
	sample_y_vetor = zeros(num_labels,1);
    sample_y_vetor(y(i)) = 1;
  	J = J + (   -log(Layer3_output(i,:))*sample_y_vetor - log(1-Layer3_output(i,:))*(1-sample_y_vetor)   );
end 
%J = -y.*log(Layer3_output)-(1-y).*log(1-Layer3_output);
%keyboard();
J = J/m;



Theta1_without_column = Theta1;
Theta1_without_column(:,1)= 0;

Theta1_without_column = Theta1_without_column.*Theta1_without_column;

Theta2_without_column = Theta2;
Theta2_without_column(:,1)= 0;

Theta2_without_column = Theta2_without_column.*Theta2_without_column;

regularizationItems = (sum(Theta1_without_column(:)) + sum(Theta2_without_column(:)))*lambda/(2*m);

J = J + regularizationItems;

%----------------------------------Part 2--------------------------------------------

outputLayer_delta = zeros(num_labels,m);
hiddenLayer_delta = zeros(hidden_layer_size+1,m);
for i=1:m,

  	sample_y_vetor = zeros(num_labels,1);
    sample_y_vetor(y(i)) = 1;
    outputLayer_delta(:,i) = transpose(Layer3_output(i,:))-sample_y_vetor;

    hiddenLayer_delta(:,i) = transpose(Theta2)*outputLayer_delta(:,i).*sigmoidGradient([1;Theta1*transpose(Layer1_output(i,:))]);

    Theta2_grad = Theta2_grad + outputLayer_delta(:,i)*(Layer2_output_include_bias(i,:));
    Theta1_grad = Theta1_grad + (hiddenLayer_delta(2:end,i))*(Layer1_output(i,:));

end

Theta1_grad = Theta1_grad./m + lambda*[zeros(hidden_layer_size,1),Theta1(:,2:end)]/m;
Theta2_grad = Theta2_grad./m + lambda*[zeros(num_labels,1),Theta2(:,2:end)]/m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
