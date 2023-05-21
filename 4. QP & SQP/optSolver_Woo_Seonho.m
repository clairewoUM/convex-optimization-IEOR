% IOE 511/MATH 562, University of Michigan
% Code written by: Seonho Woo (Claire)

% Function that runs a chosen algorithm on a chosen problem
%           Inputs: problem, method, options (structs)
%           Outputs: final iterate (x) and final function value (f)
function [x, f] = optSolver_Woo_Seonho(problem, method, options)

% set problem, method and options
[problem] = setProblem(problem);
[method] = setMethod(method);
[options] = setOptions(options);

% compute initial function/gradient/Hessian
x = problem.x0;
f = problem.compute_f(x);
g = problem.compute_g(x);
H = problem.compute_H(x);

norm_g = norm(g,inf);
term_constant = max(norm_g, 1);

% set initial iteration counter
k = 0;

while k < options.max_iterations && norm_g >= options.term_tol * term_constant
    
    % take step according to a chosen method
    switch method.name
        
        case 'Newton'
            [x_new,f_new,g_new,H_new,d,alpha,modified] = NewtonStep(x,f,g,H,problem,method,options);

        case 'BFGS'
            if k == 0
            
                H = eye(problem.n);
            end
            [x_new,f_new,g_new,H_new,d,alpha, skipped] = BFGSStep(x,f,g,H,problem,method,options);
            H_old = H;
            H = H_new;
        
        otherwise
            error('Method not implemented yet!')            
    end
    
    % update old and new function values
    x_old = x; f_old = f; g_old = g; norm_g_old = norm_g; H_old = H;
    x = x_new; f = f_new; g = g_new; norm_g = norm(g,inf); H = H_new;

    % increment iteration counter
    k = k + 1;
    
end

end