function [H] = prob2_Hess(x)
%PROB2_HESS Summary of this function goes here
%   Detailed explanation goes here

prod_ = prod(x);
exp_ = exp(prod_);

nonexp = zeros(length(x));

nonexp(1,1) = -3*(5*x(1)^4 + 2*x(2)^3*x(1) + 2*x(1));
nonexp(1,2) = -9*x(1)^2*x(2)^2;
nonexp(2,1) = nonexp(1,2);
nonexp(2,2) = -3*(5*x(2)^4 + 2*x(1)^3*x(2) + 2*x(2));

H = [(x(2)*x(3)*x(4)*x(5))^2, x(3)*x(4)*x(5)*(1+prod_), x(2)*x(4)*x(5)*(1+prod_), x(2)*x(3)*x(5)*(1+prod_), x(2)*x(3)*x(4)*(1+prod_);
    x(3)*x(4)*x(5)*(1+prod_), (x(1)*x(3)*x(4)*x(5))^2, x(1)*x(4)*x(5)*(1+prod_), x(1)*x(3)*x(5)*(1+prod_), x(1)*x(3)*x(4)*(1+prod_);
    x(2)*x(4)*x(5)*(1+prod_), x(1)*x(4)*x(5)*(1+prod_), (x(1)*x(2)*x(4)*x(5))^2, x(1)*x(2)*x(5)*(1+prod_), x(1)*x(2)*x(4)*(1+prod_);
    x(2)*x(3)*x(5)*(1+prod_), x(1)*x(3)*x(5)*(1+prod_), x(1)*x(2)*x(5)*(1+prod_), (x(1)*x(2)*x(3)*x(5))^2, x(1)*x(2)*x(3)*(1+prod_);
    x(2)*x(3)*x(4)*(1+prod_), x(1)*x(3)*x(4)*(1+prod_), x(1)*x(2)*x(4)*(1+prod_), x(1)*x(2)*x(3)*(1+prod_), (x(1)*x(2)*x(3)*x(4))^2];

 H = exp_*H + nonexp;
 
 
end

