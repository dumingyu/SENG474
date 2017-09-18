%read file
f = csvread('regdata.csv');
X = f(:,1:2);
Y = f(:,3:3);
m = length(Y);
%scale the attributes
m1 = mean(X);
max1 = max(X);
min1 = min(X);
m2 = mean(Y);
max2 = max(Y);
min2 = min(Y);
X = (X - m1)./(max1 - min1);
Y = (Y - m2)./(max2 - min2);
%compute the error
X = [ones(m, 1) X];
alpha = 0.01;
num_iters = 3000;
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, Y, theta, alpha, num_iters);
%convergence ： 收敛
% Plot the convergence graph
figure;
%numel  矩阵中元素个数
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
%plot the error vector
%find a good learning rate