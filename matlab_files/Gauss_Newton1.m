function Gauss_Newton1 
%  Solution to a nonlinear least squares problem 
%  y = Asin(wt+a), take x0 = [1; 1; 0]  
% (t1,y1) = (0, 0), (t2,y2) = (pi/3, 1/2), 
% (t3,y3) = (pi/2, 1/sqrt(2)),  (t4,y4) = (pi, 1)
%  x* = (1; 1/2; 0) % exact solution for zero residual (not unique)
format long 
t = [0; pi/3; pi/2; pi]; 
% d = [0; 1/2-.1; 1/sqrt(2)+.1; 1.2]; % nonzero residual
d = [0; 1/2; 1/sqrt(2); 1]; % zero residual
m = length(t); 
epsilon = 1e-10; 
A = 1; w = 1; a = 0;
% n = 3;
[r, J] = fun(m, t, A, w, a); 
for i = 1:20     
    p = -(J'*J)\(J'*(r-d));     
    A = A + p(1);     
    w = w + p(2);  
    a = a + p(3);
    [r, J] = fun(m, t, A, w, a);     
    ftol = norm(J'*(r-d));     
    fprintf('[A w a] = [%12.10f, %12.10f, %12.10f], ftol = %8.6e\n', [A w a], ftol);     
    if  ftol < epsilon          
        fprintf('root is within tolerance %6.2e after %2d iterations\n', epsilon, i); 
        plot(t,d, 'r*','LineWidth',2)
        hold on
        fplot(@(t) A*sin(w*t + a),[0,pi],'k-','LineWidth',2)
        return;      
    end
end
fprintf('root is NOT within tolerance %6.2e after %2d iterations', epsilon, i); 
end

function [r, J] = fun(m, t, A, w, a) 
r = zeros(3,1);  
J = zeros(m,3); 
for i = 1:m     
    r(i,1) = A*sin(w*t(i) + a);     
    J(i,1) = sin(w*t(i) + a);     
    J(i,2) = A*t(i)*cos(w*t(i) + a); 
    J(i,3) = A*cos(w*t(i) + a); 
end
end