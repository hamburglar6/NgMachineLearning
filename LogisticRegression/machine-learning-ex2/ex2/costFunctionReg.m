function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z =  X * theta;
cost_vector = -1/m * (y .* log(sigmoid(z)) + (ones(size(y)) - y) .* log(1 - sigmoid(z)));
J = sum(cost_vector) + (lambda/(2*m)) * sum(theta(2:length(theta)) .^2);

residual_vector = sigmoid(z) - y;
theta_temp = theta
theta_temp(1) = 0
grad = (1/m) * X' * residual_vector + (lambda/m)*theta_temp;





% =============================================================

end
