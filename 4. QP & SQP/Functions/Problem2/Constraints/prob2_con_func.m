function [c] = prob2_con_func(x)
%PROBLEM2_CON_FUNC Summary of this function goes here
%   Detailed explanation goes here

c = [x.'*x - 10;
    x(2)*x(3)-5*x(4)*x(5);
    x(1)^3+x(2)^3+1];

end

