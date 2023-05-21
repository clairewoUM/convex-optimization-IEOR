function [f] = prob2_func(x)
%PROB2_FUNC Summary of this function goes here
%   Detailed explanation goes here

f = exp(prod(x)) - 0.5*(x(1)^3 + x(2)^3 + 1)^2;

end

