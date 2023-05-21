function [g] = prob2_grad(x)
%PROB2_GRAD Summary of this function goes here
%   Detailed explanation goes here

exp_ = exp(prod(x));
term2 = (x(1)^3 + x(2)^3 +1);

g = [ exp_ * x(2)*x(3)*x(4)*x(5) - term2 * 3 * x(1)^2;
      exp_ * x(1)*x(3)*x(4)*x(5) - term2 * 3 * x(2)^2;
      exp_ * x(1)*x(2)*x(4)*x(5);
      exp_ * x(1)*x(2)*x(3)*x(5);
      exp_ * x(1)*x(2)*x(3)*x(4);];

end

