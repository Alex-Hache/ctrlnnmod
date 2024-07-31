function  extract_model_params_from_python(strMatFileWeights, nu, np)
sModelWeights = load(strMatFileWeights);

A = double(sModelWeights.weights{1});
B = double(sModelWeights.weights{2}(:, 1:nu));
G = double(sModelWeights.weights{2}(:, nu+1:nu+np));
C = double(sModelWeights.weights{3});

W_y_in = double(sModelWeights.weights{4});
b_alpha_in = double(sModelWeights.biases{1})';
W_p_in = double(sModelWeights.weights{5});
W_out  = double(sModelWeights.weights{6});
b_alpha_out = double(sModelWeights.biases{2});



sBLA = load('pend_BLA.mat');


A_bla = sBLA.A;
B_bla = sBLA.B(:, 1:nu);
C_bla = sBLA.C;
G_bla = sBLA.B(:,nu+1:nu+np);


%% Populate base workspace for simulation
assignin('base', 'A', A);
assignin('base', 'B', B);
assignin('base', 'C', C);
assignin('base', 'G', G);

assignin('base','W_y_in', W_y_in);
assignin('base','b_in', b_alpha_in);
assignin('base', 'W_p_in', W_p_in);
assignin('base', 'W_out', W_out);
assignin('base', 'b_out', b_alpha_out);


assignin('base', 'A_bla', A_bla);
assignin('base', 'B_bla', B_bla);
assignin('base', 'C_bla', C_bla);
assignin('base', 'G_bla', G_bla);

end
