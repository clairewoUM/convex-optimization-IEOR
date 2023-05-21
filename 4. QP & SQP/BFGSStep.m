% IOE 511/MATH 562, University of Michigan
% Code written by: Seonho Woo (Claire)

% Function that: (1) computes the Newton step; (2) updates the iterate; and, 
%                (3) computes the function and gradient at the new iterate
% 
%           Inputs: x, f, g, H, problem, method, options
%           Outputs: x_new, f_new, g_new, H_new, d, alpha
%
function [x_new, f_new, g_new, H_new, d, alpha] = BFGSStep(x, f, g, H, problem, method, options)

d = -H*g;

% determine step size
switch method.options.step_type
    case 'Backtracking'
        alpha_bar = options.alpha_bar;
        rho = options.rho;
        c1 = options.c1;
        alpha = alpha_bar;
        while problem.compute_f(x+alpha*d) > f + c1*alpha*g'*d
           alpha = alpha*rho;
        end
end
        
    x_new = x + alpha*d;
    f_new = problem.compute_f(x_new);
    g_new = problem.compute_g(x_new);

    s_new = x_new - x;
    y_new = g_new - g;
    
    inner_prod = s_new'*y_new;

    if inner_prod <= options.eps * norm(s_new) * norm(y_new)
        H_new = H;
    else
        I_ = eye(length(x));
        V = (I_ - s_new * y_new'/inner_prod);
        H_new = V * H * V' + s_new * s_new' / inner_prod;
    end
end

