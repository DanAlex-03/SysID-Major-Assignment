clear; clc

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

%% OLS Regression
[theta_CY, CY_pred, res_CY, std_CY, R2_CY, s2_CY] = ols(X_CY, CY);
[theta_Cl, Cl_pred, res_Cl, std_Cl, R2_Cl, s2_Cl] = ols(X_Cl, Cl);
[theta_Cn, Cn_pred, res_Cn, std_Cn, R2_Cn, s2_Cn] = ols(X_Cn, Cn);

%% Print Results
fprintf('\nCY Derivatives:\n');
for i = 1:length(theta_CY)
    fprintf('  CY(%d) = %.5f \xB1 %.5f\n', i-1, theta_CY(i), std_CY(i));
end
fprintf('  R^2     = %.4f\n  Fit Var = %.6e\n', R2_CY, s2_CY);

fprintf('\nCl Derivatives:\n');
for i = 1:length(theta_Cl)
    fprintf('  Cl(%d) = %.5f \xB1 %.5f\n', i-1, theta_Cl(i), std_Cl(i));
end
fprintf('  R^2     = %.4f\n  Fit Var = %.6e\n', R2_Cl, s2_Cl);

fprintf('\nCn Derivatives:\n');
for i = 1:length(theta_Cn)
    fprintf('  Cn(%d) = %.5f \xB1 %.5f\n', i-1, theta_Cn(i), std_Cn(i));
end
fprintf('  R^2     = %.4f\n  Fit Var = %.6e\n', R2_Cn, s2_Cn);

%% Actual vs Predicted Plots
figure(9); clf;
plot(t, CY, 'b', t, CY_pred, 'r--', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('C_Y'); legend('Actual','Predicted');
set(gca,'fontsize',14); grid on;

figure(10); clf;
plot(t, Cl, 'b', t, Cl_pred, 'r--', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('C_l'); legend('Actual','Predicted');
set(gca,'fontsize',14); grid on;

figure(11); clf;
plot(t, Cn, 'b', t, Cn_pred, 'r--', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('C_n'); legend('Actual','Predicted');
set(gca,'fontsize',14); grid on;

%% Residuals Plot
figure(12); clf;
scatter(t, res_CY, 'k.', 'LineWidth', 0.6); hold on;
h = yline(2*sqrt(s2_CY), 'r--', 'LineWidth', 2.4); yline(-2*sqrt(s2_CY), 'r--', 'LineWidth', 2.4);
xlabel('Time (s)'); ylabel('C_Y (Residual)');
set(gca,'fontsize',14); grid on;
legend(h,'± 2σ','Location','northeast');

figure(13); clf;
scatter(t, res_Cl, 'k.', 'LineWidth', 0.6); hold on;
h = yline(2*sqrt(s2_Cl), 'r--', 'LineWidth', 2.4); yline(-2*sqrt(s2_Cl), 'r--', 'LineWidth', 2.4);
xlabel('Time (s)'); ylabel('C_l (Residual)');
set(gca,'fontsize',14); grid on;
legend(h,'± 2σ','Location','northeast');

figure(14); clf;
scatter(t, res_Cn, 'k.', 'LineWidth', 0.6); hold on;
h = yline(2*sqrt(s2_Cn), 'r--', 'LineWidth', 2.4); yline(-2*sqrt(s2_Cn), 'r--', 'LineWidth', 2.4);
xlabel('Time (s)'); ylabel('C_n (Residual)');
set(gca,'fontsize',14); grid on;
legend(h,'± 2σ','Location','northeast');

%% Bar Plots for Derivatives with Error Bars
figure(15); clf;
bar(theta_CY); hold on;
errorbar(1:length(theta_CY), theta_CY, 2*std_CY, 'k.', 'LineWidth', 1.2);
set(gca,'fontsize',14); ylabel('\theta'); grid on;
xticks(1:length(theta_CY));
xticklabels({'C_{Y0}','C_{Y\beta}','C_{Yp}','C_{Yr}','C_{Y\delta_a}','C_{Y\delta_r}'});

figure(16); clf;
bar(theta_Cl); hold on;
errorbar(1:length(theta_Cl), theta_Cl, 2*std_Cl, 'k.', 'LineWidth', 1.2);
set(gca,'fontsize',14); ylabel('\theta'); grid on;
xticks(1:length(theta_Cl));
xticklabels({'C_{l0}','C_{l\beta}','C_{lp}','C_{lr}','C_{l\delta_a}'});

figure(17); clf;
bar(theta_Cn); hold on;
errorbar(1:length(theta_Cn), theta_Cn, 2*std_Cn, 'k.', 'LineWidth', 1.2);
set(gca,'fontsize',14); ylabel('\theta'); grid on;
xticks(1:length(theta_Cn));
xticklabels({'C_{n0}','C_{n\beta}','C_{np}','C_{nr}','C_{n\delta_r}'});

%% R² Values Bar Plot
figure(18); clf;
R2_values = [R2_CY, R2_Cl, R2_Cn];
bar(R2_values, 0.5, 'FaceColor', [0.2 0.6 0.8]);  % Light blue bars
ylabel('R^2');
set(gca,'fontsize',14);
ylim([0 1]);
xticks(1:3);
xticklabels({'C_Y', 'C_l', 'C_n'});
grid on;

%% OLS Function
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