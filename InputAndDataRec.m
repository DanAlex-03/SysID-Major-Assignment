clear; clc;

%% X-Plane Connect MATLAB Configuration for Cessna 172P Autopilot
addpath('../')
import XPlaneConnect.*

% Setup Connection to X-Plane
Socket = openUDP();
disp('Connection successful. Collecting data.');
%% Simulation Constants, Parameters, and Settings
rec_time = 60; % Recording Duration (s)
dt = 0.025;    % Sampling Period (40 Hz)
C = 1.5;       % Mean Aerodynamic Chord [m]
S = 16.2;      % Wing Surface Area [m²]
m = 911;       % Aircraft Mass (kg)
g = 9.81;      % Gravity Constant (m/s²)
Ix = 1285.3;   % Inertia X (kg.m²)
Iz = 2667.5;   % Inertia Z (kg.m²)
Iy = 1824.3;   % Inertia Y (kg.m²)
Ixz = -161.5;  % Inertia XZ (kg.m²)
rho0 = 1.225;  % Air Density at Sea Level (kg/m³)
R_air = 287;   % Air Gas Constant [J/(kg·K)]

% Choose Input: 'rudder', 'aileron', or 'both'
selected_input = 'both';  % <-- Change this as needed
fprintf('Selected input: %s\n', selected_input);

% Noise Addition: 'false' or 'true'
add_noise = true;  % <-- Change this as needed
if add_noise
    disp('Noise addition: ENABLED');
else
    disp('Noise addition: DISABLED');
end

% Samples Count
num_samples = rec_time/dt;

% Input Rudder Doublet Excitation
rudder_delay = 5;  % seconds (must be > rudder input window)
rudder_deflection_deg = 10;  % Change as needed
rudder_max_deg = 20;  % Assumed max deflection for C172
rudder_input_norm = rudder_deflection_deg / rudder_max_deg;

% Input Aileron Doublet Excitation
aileron_delay = 5; % seconds (must be > aileron input window)
aileron_deflection_deg = 10;  % Change as needed
aileron_deflection_rad = deg2rad(aileron_deflection_deg);
aileron_max_deg = 20; % Assumed max deflection for C172
aileron_input_norm = aileron_deflection_deg / aileron_max_deg;

% DREFs List
dref_beta = 'sim/flightmodel/position/beta';
dref_p = 'sim/flightmodel/position/P';
dref_r = 'sim/flightmodel/position/R';
dref_CAS      = 'sim/cockpit2/gauges/indicators/calibrated_airspeed_kts_pilot'; 
dref_Pstatic  = 'sim/weather/barometer_current_inhg';    
dref_Tstatic  = 'sim/weather/temperature_ambient_c'; 
dref_aill = 'sim/flightmodel/controls/lail1def';
dref_ailr = 'sim/flightmodel/controls/rail1def';
dref_rudl = 'sim/flightmodel/controls/ldruddef';
dref_rudr = 'sim/flightmodel/controls/rdruddef';
dref_q = 'sim/flightmodel/position/Q'; 
dref_pdot = 'sim/flightmodel/position/P_dot';
dref_rdot = 'sim/flightmodel/position/R_dot';
dref_u = 'sim/flightmodel/forces/vx_acf_axis';
dref_v = 'sim/flightmodel/forces/vy_acf_axis';
dref_w = 'sim/flightmodel/forces/vz_acf_axis';

%% Logs Initialization
rud_log = zeros(num_samples, 1);
ail_log = zeros(num_samples, 1);
aill_log = zeros(num_samples, 1);
ailr_log = zeros(num_samples, 1);
P_log = zeros(num_samples, 1);
R_log = zeros(num_samples, 1);
beta_log = zeros(num_samples, 1);
CY_log = zeros(num_samples, 1);
Cl_log = zeros(num_samples, 1);
Cn_log = zeros(num_samples, 1);
time_log = zeros(num_samples, 1);
Q_log = zeros(num_samples, 1);
pdot_log = zeros(num_samples, 1);
rdot_log = zeros(num_samples, 1);
u_log = zeros(num_samples, 1);
v_log = zeros(num_samples, 1);
w_log = zeros(num_samples, 1);

%% Input Simulation, Noise Addition, and Data Collection
for i = 1:num_samples
    current_time = (i-1)*dt;
    time_log(i) = current_time;

    % Initialize Control Input Array
    switch selected_input
        case 'aileron'
            ctrl = [-998, 0, -998, -998, -998, -998];  % Only aileron
        case 'rudder'
            ctrl = [-998, -998, 0, -998, -998, -998];  % Only rudder
        case 'both'
            ctrl = [-998, 0, 0, -998, -998, -998];     % Both aileron and rudder
        otherwise
            error('Invalid input selection: choose ''rudder'', ''aileron'', or ''both''.');
    end

    % Apply Aileron Doublet (only if selected)
    if any(strcmp(selected_input, {'aileron', 'both'}))
        if current_time >= aileron_delay && current_time < aileron_delay + 2
            ctrl(2) =  aileron_input_norm;
        elseif current_time >= aileron_delay + 1 && current_time < aileron_delay + 4
            ctrl(2) = -aileron_input_norm;
        else
            ctrl(2) = 0;
        end
    end

    % Apply Rudder Doublet (only if selected)
    if any(strcmp(selected_input, {'rudder', 'both'}))
        if current_time >= rudder_delay && current_time < rudder_delay + 2
            ctrl(3) =  rudder_input_norm;
        elseif current_time >= rudder_delay + 1 && current_time < rudder_delay + 4
            ctrl(3) = -rudder_input_norm;
        else
            ctrl(3) = 0;
        end
    end

    sendCTRL(ctrl, 0, Socket);

    % Define Gaussian Noise Standard Deviations
    noise_std = struct( ...
        'beta', 0.01, ...
        'p', 0.1, ...
        'r', 0.1, ...
        'CAS', 1, ...
        'Pstatic', 1, ...
        'Tstatic', 1, ...
        'aill', 0.1, ...
        'ailr', 0.1, ...
        'rud', 0.1, ...
        'q', 0.001, ...
        'pdot', 0.01, ...
        'rdot', 0.01, ...
        'u', 0.01, ...
        'v', 0.001, ...
        'w', 0.001 ...
    );
    
    result = getDREFs({dref_beta, dref_p, dref_r, dref_CAS, ...
        dref_Pstatic, dref_Tstatic, dref_aill, dref_ailr, dref_rudl, ...
        dref_q, dref_pdot, dref_rdot, dref_u, dref_v, dref_w}, Socket);
    
    % Add Optional Gaussian Noise
    if add_noise
        beta_val    = result(1) + noise_std.beta * randn;
        p_val       = result(2) + noise_std.p * randn;
        r_val       = result(3) + noise_std.r * randn;
        CAS_val     = result(4) + noise_std.CAS * randn;
        Pstatic_val = result(5) + noise_std.Pstatic * randn;
        Tstatic_val = result(6) + noise_std.Tstatic * randn;
        aill_val    = result(7) + noise_std.aill * randn;
        ailr_val    = result(8) + noise_std.ailr * randn;
        rud_val     = result(9) + noise_std.rud * randn;
        q_val       = result(10) + noise_std.q * randn;
        pdot_val    = result(11) + noise_std.pdot * randn;
        rdot_val    = result(12) + noise_std.rdot * randn;
        u_val       = result(13) + noise_std.u * randn;
        v_val       = result(14) + noise_std.v * randn;
        w_val       = result(15) + noise_std.w * randn;
    else
        beta_val    = result(1);
        p_val       = result(2);
        r_val       = result(3);
        CAS_val     = result(4);
        Pstatic_val = result(5);
        Tstatic_val = result(6);
        aill_val    = result(7);
        ailr_val    = result(8);
        rud_val     = result(9);
        q_val       = result(10);
        pdot_val    = result(11);
        rdot_val    = result(12);
        u_val       = result(13);
        v_val       = result(14);
        w_val       = result(15);
    end

    % Unit Conversion
    CAS_m_s = mean(CAS_val) * 0.514444;
    P_static = mean(Pstatic_val) * 3386.39;
    T_static = mean(Tstatic_val) + 273.15;
    rho = P_static / (R_air * T_static);
    V = CAS_m_s * sqrt(rho0 / rho);
    Qbar = 0.5 * rho * V^2;
    p_val = p_val * (pi/180);
    r_val = r_val * (pi/180);
    pdot_val = pdot_val * (pi/180);
    rdot_val = rdot_val * (pi/180);

    u_log(i) = u_val;
    v_log(i) = v_val;
    w_log(i) = w_val;

    if i == 1
        vdot = 0;
    else
        vdot = (v_log(i) - v_log(i-1)) / dt;
    end
    
    if Qbar > 0
        CY = (m * (vdot + u_val * r_val - w_val * p_val)) / (Qbar * S);
        Cl = (Ix * pdot_val + Ixz * rdot_val + q_val * r_val * (Iz - Iy) + Ixz * q_val * p_val) / (Qbar * S * C);
        Cn = (Iz * rdot_val + Ixz * pdot_val + p_val * q_val * (Iy - Ix) - Ixz * q_val * r_val) / (Qbar * S * C);
    else
        CY = 0; Cl = 0; Cn = 0;
    end
    
    % Data Logging
    P_log(i) = p_val;
    R_log(i) = r_val;
    Q_log(i) = q_val;
    pdot_log(i) = pdot_val;
    rdot_log(i) = rdot_val;
    beta_log(i) = beta_val;
    rud_log(i) = rud_val;
    ail_log(i) = 0.5 * (ailr_val - aill_val);
    aill_log(i) = aill_val;
    ailr_log(i) = ailr_val;

    CY_log(i) = CY;
    Cl_log(i) = Cl;
    Cn_log(i) = Cn;
end

%% Close Connection
closeUDP(Socket);
disp('Closing connection. Data collection completed.');

%% Calculate SNR for Noisy Parameters (if noise was added)
if add_noise
    disp('Calculating SNR for each noisy parameter... (>|10 dB| preferred)');

    snr_values = struct();
    signal_data = struct( ...
        'beta', beta_log, ...
        'p', P_log, ...
        'r', R_log, ...
        'CAS', [], ... 
        'Pstatic', [], ...
        'Tstatic', [], ...
        'aill', aill_log, ...
        'ailr', ailr_log, ...
        'rud', rud_log, ...
        'q', Q_log, ...
        'pdot', pdot_log, ...
        'rdot', rdot_log, ...
        'u', u_log, ...
        'v', v_log, ...
        'w', w_log ...
    );

    param_names = fieldnames(noise_std);
    fprintf('\n%-10s | SNR (dB)\n', 'Parameter');
    fprintf('----------------------\n');
    for i = 1:length(param_names)
        name = param_names{i};
        if isempty(signal_data.(name))
            continue  % Skip parameters not logged
        end
        signal_power = var(signal_data.(name));
        noise_power = noise_std.(name)^2;
        snr_db = 10 * log10(signal_power / noise_power);
        snr_values.(name) = snr_db;
        fprintf('%-10s | %6.2f dB\n', name, snr_db);
    end
end

%% Plot SNR Bar Chart
figure('Name', 'Signal-to-Noise Ratio (SNR)', 'NumberTitle', 'off');

param_list = fieldnames(snr_values);
snr_dB = zeros(length(param_list), 1);

for i = 1:length(param_list)
    snr_dB(i) = abs(snr_values.(param_list{i}));
end

barh(snr_dB, 'FaceColor', [0.2 0.6 0.8]);
hold on;

h = xline(10, 'k--', 'LineWidth', 3);

hold off;
set(gca,'fontsize',14)
yticks(1:length(param_list));
yticklabels(param_list);
xlabel('|SNR| (dB)');
ylabel('Measurements');
legend(h, '10 dB Threshold', 'Location', 'southeast');
grid on;

%% Save to CSV and Excel
T = table(time_log, u_log, v_log, w_log, Q_log, pdot_log, rdot_log, beta_log, ...
    P_log, R_log, rud_log, ail_log, aill_log, ailr_log, CY_log, Cl_log, Cn_log, ...
    'VariableNames', {'Time_s', 'u', 'v', 'w', 'q_rad_s', 'p_dot', 'r_dot', ...
    'beta_rad', 'p_rad_s', 'r_rad_s', 'rud_deg', 'ail_deg', 'aill_deg', ...
    'ailr_deg', 'CY', 'Cl', 'Cn'});

writetable(T, 'BothNew_Noise_4.csv');
writetable(T, 'BothNew_Noise_4.xlsx');

disp('Results saved to CSV and Excel.');

%% Post-Simulation Plot
disp('Initializing plot.');
figure('Name', 'Recorded Flight Data vs Time', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);

% Define Variables to Plot (all saved in CSV/XLSX)
plot_vars = {'u', 'v', 'w', 'q_rad_s', 'p_dot', 'r_dot', 'beta_rad', 'p_rad_s', 'r_rad_s', ...
    'rud_deg', 'ail_deg', 'aill_deg', 'ailr_deg', 'CY', 'Cl', 'Cn'};

num_plots = length(plot_vars);

for k = 1:num_plots
    subplot(4, 4, k);  % 4x4 grid
    plot(T.Time_s, T.(plot_vars{k}), 'b', 'LineWidth', 1.2);
    xlabel('Time (s)');
    ylabel(strrep(plot_vars{k}, '_', '\_'));
    title([strrep(plot_vars{k}, '_', '\_') ' vs Time']);
    grid on;
end

sgtitle('Flight Parameters over Time');
