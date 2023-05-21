% IOE 511/MATH 562, University of Michigan
% Code written by: Seonho Woo (Claire)

% Function that runs a chosen algorithm on a chosen problem
%           Inputs: problem, method, options (structs)
%           Outputs: final iterate (x) and final function value (f)
function [x, f, norm_c, cvals, fvals] = optSolverConst_Woo_Seonho(problem,method,options)

% set problem, method and options
[problem] = setProblem(problem);
[method] = setMethod(method);
[options] = setOptions(options);

% compute initial function/gradient/Hessian
x = problem.x0;

f = problem.compute_f(x);
g = problem.compute_g(x);
% H = problem.compute_H(x);

fvals = [];
cvals = [];

c = problem.compute_c(x);
gc = problem.compute_gc(x);
% Hc = problem.compute_Hc(x);

% Initial lambda and Lagrangian values
lam = -gc\g;
gL = g + gc*lam;
norm_gL = norm(gL, inf);
L_term_constant = max(norm_gL, 1);

norm_c = norm(c, inf);
c_term_constant = max(norm_c, 1);

cvals = [cvals, norm_c];
fvals = [fvals, f];

% set initial iteration counter
k = 0;
nu = options.nu0;
ksi = 1/nu;
eps = options.term_tol;
gamma = options.gamma;
num_unc = 0;

while k < options.max_iterations
    
    % take step according to a chosen method
    switch method.name

        case 'QuadraticPenalty'

            % Step 2: solve phi(x, nu)
            phi = @(t)problem.compute_f(t) + 0.5*nu*problem.compute_c(t).'*problem.compute_c(t);
            g_phi = @(t)problem.compute_g(t) + nu*problem.compute_gc(t)*problem.compute_c(t);
            
            subproblem.compute_f = phi;
            subproblem.compute_g = g_phi;
            H_sub = eye(problem.n);
            phi_k = phi(x);
            g_phi_k = g_phi(x);

            % Number of steps of the unconstrained optimization
%             num_unc = 0;

            while norm(g_phi_k, inf) > ksi   
                [x_new, phi_new, g_phi_new, H_sub_new, ~, ~] = BFGSStep(x, phi_k, g_phi_k, H_sub, subproblem, method, options);
                x = x_new; phi_k = phi_new; g_phi_k = g_phi_new; H_sub = H_sub_new;
                num_unc = num_unc + 1;
            end
            
            % Step 3: Check termination condition
            g_new = problem.compute_g(x);
            %H = problem.compute_H(x);

            c_new = problem.compute_c(x);
            gc_new = problem.compute_gc(x);
            %Hc = problem.compute_Hc(x);
            norm_c = norm(c_new, inf);
            
            lam = -gc_new\g_new;
            gL = g_new + gc_new*lam;
            norm_gL = norm(gL, inf);
            
            if norm_gL <= eps * L_term_constant && norm_c <= eps * c_term_constant
                f = problem.compute_f(x);
                break;
            end
            
            % Step 4: Update nu
            nu = gamma * nu;
            ksi = 1/nu;

            % Step 5: Update x_k_bar
            % Already done when setting x = x_new

        case 'SQP'

            % Check termination condition
            gc = problem.compute_gc(x);
            g = problem.compute_g(x);
            gL = g + gc * lam;
            c = problem.compute_c(x);

            norm_gL = norm(gL, inf);
            norm_c = norm(c, inf);
            
            if norm_gL <= eps * L_term_constant && norm_c <= eps * c_term_constant
                f = problem.compute_f(x);
                break;
            end
            
            % Solve linear system
            Hk = problem.compute_H(x);
            Hc = problem.compute_Hc(x);

            for i = 1:length(c)
                Hk = Hk + lam(i)*Hc(:,:,i);
            end
            A = zeros(length(x)+length(c));
            A(1:length(x), 1:length(x)) = Hk;
            A(1:length(x), length(x)+1:end) = gc;
            A(length(x)+1:end, 1:length(x)) = gc';
            b = -[g; c];
            z = A\b;
            d = z(1:length(x));

            % Update lambda and x
            lam = z(length(x)+1:end);
            x = x + d;
        otherwise
            error('Method not implemented yet!')            
    end
    
%     % update old and new function values
%     x_old = x; f_old = f; g_old = g; H_old = H;
%     c_old = c; gc_old = gc; Hc_old = Hc; norm_c_old = norm_c;
%     x = x_new; f = f_new; g = g_new;
%     c = c_new; gc = gc_new;

    % increment iteration counter
    k = k + 1;
    cvals = [cvals, norm_c];
    f = problem.compute_f(x);
    fvals = [fvals, f];
    
end

avg_num_unc = num_unc / k;

end