function J = computeCostMulti(X, y, theta)
% Initialize some useful values
m = length(y); % number of training examples
J = 0;
J = sum(((theta'*(X')-y').^2))/(2*m);
end