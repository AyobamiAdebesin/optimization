function Gradient_Descent1
D = [3/2 2; 0 3/2];
Q = D + D';
b = [-3;1];
c = -22;
x0 = 1e10*eye(2,1);
x1 = zeros(2,1);
% x1 = [-2;2];
% alpha = 1/3;
alpha = 1/5;
 
disp('GRADIENT DESCENT ALGORITHM')
disp('____________________________________________________')
disp('k .......... x ........... y ............ f(xk) ....')
disp('____________________________________________________')

k = 0;
while norm(x1-x0)>1e-6
    x0 = x1;
    x1 = x0 - alpha*(Q*x0-b);
    f = x1'*Q*x1/2 - b'*x1 + c;
    k = k + 1;
    fprintf('%2.0f ... %12.8f ... %12.8f ... %12.8f\n', k,x1(1),x1(2),f);   
end
disp('EXACT SOLUTION')
disp('___________________________________________________')
disp('x ................... y ................ f(xk) ....')
disp('___________________________________________________')
x = Q\b;
fprintf('%12.8f ... %12.8f ... %12.8f\n',x(1), x(2), x'*Q*x/2 - b'*x + c);
end

