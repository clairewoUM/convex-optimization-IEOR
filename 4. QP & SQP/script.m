% IOE 511/MATH 562, University of Michigan
% Code written by: Albert S. Berahas

% Script to run code

% close all figures, clear all variables from workspace and clear command
% window
close all
clear all
clc

addpath(genpath('./Functions'))

% set problem (minimal requirement: name of problem)
% problem.name = 'Problem1';
% problem.x0 = [2; 2];
% problem.n = length(problem.x0);

problem.name = 'Problem2';
problem.x0 = [-1.8; 1.7; 1.9; -0.8; -0.8];
problem.n = length(problem.x0);


% set method (minimal requirement: name of method)
method.name = 'SQP';
% method.name = "QuadraticPenalty";
method.options.step_type = 'Backtracking';

% set options
options.term_tol = 1e-5;
options.max_iterations = 1e3;
options.gamma = 10;
options.nu0 = 1e-4;
options.alpha_bar = 1;
options.c1 = 1e-4;
options.rho=0.5;
options.eps = 1e-6;

% run method and return x^* and f^*
[x,f,norm_c, cvals, fvals] = optSolverConst_Woo_Seonho(problem,method,options);

figure
plot(1:length(fvals), (fvals), 'LineWidth', 2)
% Various plot options: Axis labels, grid, font size and typeface.
xlabel('Iterations $k$', 'Interpreter', 'latex')
ylabel('$f(x_k)$', 'Interpreter', 'latex')
grid on
hold on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15)

figure
plot(1:length(cvals), (cvals), 'LineWidth', 2)
% Various plot options: Axis labels, grid, font size and typeface.
xlabel('Iterations $k$', 'Interpreter', 'latex')
ylabel('$c(x_k)$', 'Interpreter', 'latex')
grid on
hold on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15)



