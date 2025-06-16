clear; clc; clf;

% Load CSV
T = readtable('Dataset_1.csv');
t = T.Time_s;

%% Plot All Variables vs Time
vars_to_plot = T.Properties.VariableNames;
vars_to_plot(strcmp(vars_to_plot, 'Time_s')) = [];

plots_per_fig = 2;
num_vars = numel(vars_to_plot);
num_figs = ceil(num_vars / plots_per_fig);

for figIdx = 1:num_figs
    figure(figIdx); clf;
    start_idx = (figIdx - 1) * plots_per_fig + 1;
    end_idx = min(figIdx * plots_per_fig, num_vars);
    for j = start_idx:end_idx
        subplot(2, 1, j - start_idx + 1)
        plot(t, T.(vars_to_plot{j}), 'b');
        xlabel('Time (s)');
        ylabel(strrep(vars_to_plot{j}, '_', '\_'));
        title([strrep(vars_to_plot{j}, '_', '\_') ' vs Time']);
        grid on;
    end
end

%% Extract Variables
beta = T.beta_rad;
p    = T.p_rad_s;
r    = T.r_rad_s;
ail  = T.ail_deg;
rud  = T.rud_deg;
CY   = T.CY;
Cl   = T.Cl;
Cn   = T.Cn;

%% Regression Matrices
X_CY = [ones(size(beta)), beta, p, r, ail, rud];
X_Cl = [ones(size(beta)), beta, p, r, ail];
X_Cn = [ones(size(beta)), beta, p, r, rud];

% Apply Kalman Filter to each output
[theta_CY_kf, P_CY, res_CY_kf, CY_pred_kf, St_CY] = kf(X_CY, CY);
[theta_Cl_kf, P_Cl, res_Cl_kf, Cl_pred_kf, St_Cl] = kf(X_Cl, Cl);
[theta_Cn_kf, P_Cn, res_Cn_kf, Cn_pred_kf, St_Cn] = kf(X_Cn, Cn);

[theta_CY_ols, CY_pred_ols, res_CY_ols, std_CY, R2_CY, s2_CY] = ols(X_CY, CY);
[theta_Cl_ols, Cl_pred_ols, res_Cl_ols, std_Cl, R2_Cl, s2_Cl] = ols(X_Cl, Cl);
[theta_Cn_ols, Cn_pred_ols, res_Cn_ols, std_Cn, R2_Cn, s2_Cn] = ols(X_Cn, Cn);

% Global mean for baseline R² denominator
zbar_CY = mean(CY);
zbar_Cl = mean(Cl);
zbar_Cn = mean(Cn);

% Per-time-step R² and fit error
s2_CY_t = (res_CY_kf).^2;
s2_Cl_t = (res_Cl_kf).^2;
s2_Cn_t = (res_Cn_kf).^2;

%% KF vs OLS 
figure(9); clf;
plot(t, CY, 'b', t, CY_pred_kf, 'g--', t, CY_pred_ols, 'r--', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('C_Y'); legend('Actual','KF','OLS');
set(gca,'fontsize',14); grid on;

figure(10); clf;
plot(t, Cl, 'b',t, Cl_pred_kf, 'g--', t, Cl_pred_ols, 'r--', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('C_l'); legend('Actual','KF','OLS');
set(gca,'fontsize',14); grid on;

figure(11); clf;
plot(t, Cn, 'b',t, Cn_pred_kf, 'g--', t, Cn_pred_ols, 'r--', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('C_n'); legend('Actual','KF','OLS');
set(gca,'fontsize',14); grid on;

%% Residuals Plot
std_res_CY = sqrt(St_CY);
std_res_Cl = sqrt(St_Cl);
std_res_Cn = sqrt(St_Cn);

figure(12); clf;
plot(t, res_CY_ols, 'c', t, res_CY_kf, 'k', 'LineWidth', 1.2); hold on;
plot(t, 2*std_res_CY, 'r--');
plot(t, -2*std_res_CY, 'r--');
ylim([-0.05 0.05]);
yticks(-0.05:0.025:0.05);
xlabel('Time (s)'); ylabel('C_Y (Residual)');
set(gca,'fontsize',14); grid on; legend('OLS','KF');

figure(13); clf;
plot(t, res_Cl_ols, 'c', t, res_Cl_kf, 'k', 'LineWidth', 1.2); hold on;
plot(t, 2*std_res_Cl, 'r--');
plot(t, -2*std_res_Cl, 'r--');
ylim([-0.025 0.025]);
xlabel('Time (s)'); ylabel('C_l (Residual)');
set(gca,'fontsize',14); grid on; legend('OLS','KF');

figure(14); clf;
plot(t, res_Cn_ols, 'c', t, res_Cn_kf, 'k', 'LineWidth', 1.2); hold on;
plot(t, 2*std_res_Cn, 'r--');
plot(t, -2*std_res_Cn, 'r--');
ylim([-0.005 0.005]);
yticks(-0.005:0.0025:0.005);
xlabel('Time (s)'); ylabel('C_n (Residual)');
set(gca,'fontsize',14); grid on; legend('OLS','KF');

%% Derivatives vs Time
figure(15); clf;
labels_CY = {'C_{Y0}','C_{Y\beta}','C_{Yp}','C_{Yr}','C_{Y\delta_a}','C_{Y\delta_r}'};
colors = lines(length(labels_CY));  % Distinct colors for each derivative

for i = 1:length(labels_CY)
    plot(t, theta_CY_kf(i,:), '-', 'Color', colors(i,:), 'DisplayName', labels_CY{i}, 'LineWidth', 2); hold on;
    yline(theta_CY_ols(i), '--', 'Color', colors(i,:), 'HandleVisibility','off', 'LineWidth', 2);
end
ylim([-0.01 0.01]);
set(gca,'fontsize',14);
xlabel('Time (s)'); ylabel('\theta');
legend('Location', 'northeast','NumColumns',3);
grid on;

figure(16); clf;
labels_Cl = {'C_{l0}','C_{l\beta}','C_{lp}','C_{lr}','C_{l\delta_a}'};
colors = lines(length(labels_Cl));

for i = 1:length(labels_Cl)
    plot(t, theta_Cl_kf(i,:), '-', 'Color', colors(i,:), 'DisplayName', labels_Cl{i}, 'LineWidth', 2); hold on;
    yline(theta_Cl_ols(i), '--', 'Color', colors(i,:), 'HandleVisibility','off', 'LineWidth', 2);
end
ylim([-0.025 0.025]);
set(gca,'fontsize',14);
xlabel('Time (s)'); ylabel('\theta');
legend('Location', 'northeast','NumColumns',3);
grid on;

figure(17); clf;
labels_Cn = {'C_{n0}','C_{n\beta}','C_{np}','C_{nr}','C_{n\delta_r}'};
colors = lines(length(labels_Cn));

for i = 1:length(labels_Cn)
    plot(t, theta_Cn_kf(i,:), '-', 'Color', colors(i,:), 'DisplayName', labels_Cn{i}, 'LineWidth', 2); hold on;
    yline(theta_Cn_ols(i), '--', 'Color', colors(i,:), 'HandleVisibility','off', 'LineWidth', 2);
end
ylim([-0.025 0.025]);
set(gca,'fontsize',14);
xlabel('Time (s)'); ylabel('\theta');
legend('Location', 'northeast','NumColumns',3);
grid on;

%% R² Values Plot
R2_CY_t = 1 - (res_CY_kf.^2) ./ (var(CY) * ones(size(res_CY_kf)));
R2_Cl_t = 1 - (res_Cl_kf.^2) ./ (var(Cl) * ones(size(res_Cl_kf)));
R2_Cn_t = 1 - (res_Cn_kf.^2) ./ (var(Cn) * ones(size(res_Cn_kf)));

k = 41;
R2_CY_t = movmean(R2_CY_t, k);
R2_Cl_t = movmean(R2_Cl_t, k);
R2_Cn_t = movmean(R2_Cn_t, k);

figure(18); clf;
plot(t, R2_CY_t, 'b', t, R2_Cl_t, 'r', t, R2_Cn_t, 'g', 'LineWidth', 2);
hold on
yline(R2_CY, 'b--', 'LineWidth', 2);
yline(R2_Cl, 'r--', 'LineWidth', 2);
yline(R2_Cn, 'g--', 'LineWidth', 2);
legend('C_Y (KF)','C_l (KF)','C_n (KF)','C_Y (OLS)','C_l (OLS)','C_n (OLS)','Location','southeast','NumColumns',2);
xlabel('Time (s)');
ylabel('R^2'); 
grid on;
set(gca,'fontsize',14);
ylim([0.98 1]);

%% KF and OLS Function
function [theta_hist, P_hist, residuals, y_pred, St] = kf(X, z)
    N = size(X, 1);        % Number of time steps
    n = size(X, 2);        % Number of parameters

    % Initialization
    theta = zeros(n, 1);   % Initial parameter estimate
    P = 1e3 * eye(n);      % Initial covariance (large uncertainty)
    Q = 1e-6 * eye(n);     % Process noise covariance (tune if needed)
    R = 1e-3;              % Measurement noise variance (scalar)

    % Storage
    theta_hist = zeros(n, N);
    P_hist = zeros(n, n, N);
    residuals = zeros(1, N);
    y_pred = zeros(1, N);
    St = zeros(1, N);      % Innovation covariance per step

    % Kalman filter loop
    for t = 1:N
        H = X(t, :)';            % Regressor column vector
        y = z(t);                % Measurement at time t

        % Prediction
        theta_pred = theta;
        P_pred = P + Q;

        % Innovation
        y_hat = H' * theta_pred;
        S = H' * P_pred * H + R;
        K = (P_pred * H) / S;

        % Update
        residuals(t) = y - y_hat;
        theta = theta_pred + K * residuals(t);
        P = (eye(n) - K * H') * P_pred;

        % Store
        theta_hist(:, t) = theta;
        P_hist(:, :, t) = P;
        y_pred(t) = y_hat;
        St(t) = S;
    end
end

function [theta, z_pred, residuals, stddev, R2, s2] = ols(X, z)
    theta = (X' * X) \ (X' * z);
    z_pred = X * theta;
    residuals = z - z_pred;
    N = length(z);
    p = size(X, 2);
    s2 = sum(residuals.^2) / (N - p);                       % Fit error
    Cov = s2 * inv(X' * X);                                 % Covariance matrix
    stddev = sqrt(diag(Cov));                               % Std dev of coefficients
    zbar = mean(z);
    R2 = ((theta' * (X' * z)) - N*zbar^2) / (z' * z - N*zbar^2); % R-squared
end