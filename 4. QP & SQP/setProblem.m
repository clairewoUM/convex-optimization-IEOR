% IOE 511/MATH 562, University of Michigan
% Code written by: Albert S. Berahas

% Function that specifies the problem. Specifically, a way to compute: 
%    (1) function values; (2) gradients; and, (3) Hessians (if necessary).
%
%           Input: problem (struct), required (problem.name)
%           Output: problem (struct)
%
% Error(s): 
%       (1) if problem name not specified;
%       (2) function handles (for function, gradient, Hessian) not found
%
function [problem] = setProblem(problem)

% check is problem name available
if ~isfield(problem,'name')
    error('Problem name not defined!!!')
end

% set function handles according the the selected problem
switch problem.name
        
    case 'Problem1'
        
        problem.compute_f = @prob1_func;
        problem.compute_g = @prob1_grad;
        problem.compute_H = @prob1_Hess;
        
        problem.compute_c = @prob1_con_func;
        problem.compute_gc = @prob1_con_grad;
        problem.compute_Hc = @prob1_con_Hess;
        

    case 'Problem2'
        
        problem.compute_f = @prob2_func;
        problem.compute_g = @prob2_grad;
        problem.compute_H = @prob2_Hess;
        
        problem.compute_c = @prob2_con_func;
        problem.compute_gc = @prob2_con_grad;
        problem.compute_Hc = @prob2_con_Hess;
    
    otherwise
        
        error('Problem not defined!!!')
end