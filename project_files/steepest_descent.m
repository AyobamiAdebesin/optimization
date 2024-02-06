function [alpha, C] = exponential_least_squares(data_points)
    rng(0); % Set random seed
    nrows = size(data_points, 1);
    ncols = size(data_points, 2);
    A = ones(nrows, ncols);
    B = zeros(nrows, 1);
    for i = 1:size(A, 1)
        A(i, 1) = data_points(i, 1);
    end
    for i = 1:size(B, 1)
        B(i, 1) = log(data_points(i, 2));
    end
    Q = 2 * A' * A;
    b = 2 * A' * B;
    c = B' * B;
    x_0 = steepest_descent(Q, b, c);
    vec = squeeze(x_0);
    alpha = vec(1);
    C = exp(vec(2));
end

function x_0 = steepest_descent(Q, b, c)
    if ndims(Q) ~= 2 || any(size(Q) ~= [2, 2])
        error('Q must be a 2x2 matrix');
    end
    if ndims(b) ~= 2 || any(size(b) ~= [2, 1])
        error('b must be a 2x1 matrix');
    end
    x_0 = zeros(2, 1);
    fprintf('::::::::Fixed step Gradient Descent Algorithm:::::::::::\n');
    for i = 1:3
        g_k = Q * x_0 - b;
        try
            alpha_k = (g_k' * g_k) / (g_k' * Q * g_k);
        catch
            fprintf('Unable to compute step size alpha for step %d\n', i);
            return;
        end
        alpha_k = squeeze(alpha_k);
        if ~isa(alpha_k, 'double')
            error('step size is not of type double. Check if correct values are passed');
        end
        x_0 = x_0 - alpha_k * g_k;
    end
    fprintf('Approximate minimizer: %f\n', x_0);
    x = inv(Q) * b;
    fprintf('Exact minimizer: %f\n', x);
end

function plot_exp_points(data_points, alpha, C)
    x = zeros(length(data_points), 1);
    y = zeros(length(data_points), 1);
    for i = 1:length(data_points)
        x(i) = data_points(i, 1);
        y(i) = data_points(i, 2);
    end
    scatter(x, y, 'DisplayName', 'Random Points');
    hold on;
    plot(x, C * exp(alpha * x), 'Color', 'red', 'DisplayName', 'Exponential Curve');
    title('Random Points and Exponential Curve');
    xlabel('X Axis');
    ylabel('Y Axis');
    legend;
    hold off;
end