function [theta, J_history] = gradientDescentMulti(X, Y, theta, alpha, num_iters)
% Initialize some useful values
m = length(Y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    %theta �� 3*1 ��������   X�ĵ�һ����1 ǡ�ó���theta0
    % ��1*3 X  3*47  -  1*47��'*
    theta = theta + (alpha/m) *sum((Y-X*theta).*X)'; 
    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, Y, theta);
end
end