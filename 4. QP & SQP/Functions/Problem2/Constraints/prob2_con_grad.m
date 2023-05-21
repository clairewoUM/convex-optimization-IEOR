function [g_c] = prob2_con_grad(x)
%PROB2_CON_GRAD Summary of this function goes here
%   Detailed explanation goes here

g_c = [2*x, [0; x(3); x(2); -5*x(5); -5*x(4)], [3*x(1)^2; 3*x(2)^2; 0; 0; 0] ];

end

