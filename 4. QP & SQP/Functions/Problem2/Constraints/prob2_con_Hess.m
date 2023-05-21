function [H_c] = prob2_con_Hess(x)
%PROB2_CON_HESS Summary of this function goes here
%   Detailed explanation goes here

H_c = zeros(length(x), length(x), 3);
H_c(:, :, 1) = 2*eye(length(x));
H_c(2, 3, 2) = 1;
H_c(3, 2, 2) = 1;
H_c(4, 5, 2) = -5;
H_c(5, 4, 2) = -5;

H_c(1, 1, 3) = 6*x(1);
H_c(2, 2, 3) = 6*x(2);

end

