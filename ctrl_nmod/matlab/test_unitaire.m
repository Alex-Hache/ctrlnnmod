clear all
close all
clc
addpath(genpath('weights'))
strMatFileWeights = 'Res_flnsssm_base_16_1_30_512_0.005_adam_20000epch_4.190e-02.mat';
nu = 1;
nx = 2;
np = 1;
dt = 0.1;
fig_name = 'Test';
assignin('base', 'name_fig1', fullfile(pwd, 'figures', [fig_name, '_1.fig']));
sprintf(" Path to figure 1 : %s", fullfile(pwd, 'figures', [fig_name, '_1.fig']));
assignin('base', 'name_fig2', fullfile(pwd, 'figures', [fig_name, '_2.fig']));
sprintf(" Path to figure 2 : %s", fullfile(pwd, 'figures', [fig_name, '_2.fig']));

extract_model_params_from_python(strMatFileWeights, nu, np);
closedLoopresults_pendulum;