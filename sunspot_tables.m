% SUNSPOTS_TABLE_MOMENTUM.M


clear; clc; close all;

%% ========================================================================
%  1) LOAD AND NORMALIZE SUNSPOT DATA
%% ========================================================================
load sunspot.dat;  % columns: [year, sunspot_number]
year    = sunspot(:,1);
relNums = sunspot(:,2);

minVal  = min(relNums);
maxVal  = max(relNums);
relNumsNorm = 2*((relNums - minVal)/(maxVal - minVal) - 0.5);

%% ========================================================================
%  2) CREATE 5-LAG INPUTS
%% ========================================================================
inputDim = 5; 
N_total  = length(relNumsNorm);

X_all = zeros(N_total - inputDim, inputDim);
Y_all = zeros(N_total - inputDim, 1);

for i = 1:(N_total - inputDim)
    X_all(i,:) = relNumsNorm(i : i+inputDim-1);
    Y_all(i)   = relNumsNorm(i + inputDim);
end

% 80/20 split
N_data   = size(X_all,1);
Ntrain   = floor(0.8*N_data);
Ntest    = N_data - Ntrain;

Xtrain_all = X_all(1:Ntrain,:);
Ytrain_all = Y_all(1:Ntrain);
Xtest_all  = X_all(Ntrain+1:end,:);
Ytest_all  = Y_all(Ntrain+1:end);

%% ========================================================================
%  3) HYPERPARAMETERS
%% ========================================================================
LEARNING_RATE_CNN = 0.005;  
EPOCHS_CNN        = 100;
TSS_LIMIT         = 0.02;
MOMENTUM          = 0.9;  % For CNN updates

LEARNING_RATE_MLP = 0.001;
EPOCHS_MLP        = 100;
INIT_RANGE        = 0.25;

%% ========================================================================
%  4) CNN WITH DROPOUT + MOMENTUM
%% ========================================================================
disp('=== CNN WITH DROPOUT (5,5,5) + MOMENTUM ===');
dropH1 = 0;
dropH2 = 0.01;
dropH3 = 0;

N_H1 = 5; N_H2 = 5; N_H3 = 5;

weights_cnn_drop = initialize_weights(inputDim, N_H1, N_H2, N_H3, INIT_RANGE);
[MSE_cnn_drop, weights_cnn_drop] = train_network_dropout_momentum( ...
    Xtrain_all, Ytrain_all, weights_cnn_drop, ...
    inputDim, N_H1, N_H2, N_H3, ...
    LEARNING_RATE_CNN, EPOCHS_CNN, TSS_LIMIT, ...
    dropH1, dropH2, dropH3, MOMENTUM);

% Predictions
Ypred_cnn_drop_all = predict_network(X_all, weights_cnn_drop, ...
    inputDim, N_H1, N_H2, N_H3);
Ypred_cnn_drop_tr  = Ypred_cnn_drop_all(1:Ntrain);
Ypred_cnn_drop_te  = Ypred_cnn_drop_all(Ntrain+1:end);

[mseTrain_cnn_drop, raeTrain_cnn_drop] = compute_metrics(Ytrain_all, Ypred_cnn_drop_tr);
[mseTest_cnn_drop,  raeTest_cnn_drop ] = compute_metrics(Ytest_all,  Ypred_cnn_drop_te);

figure('Name','CNN Dropout - Train Fit','NumberTitle','off');
plot_fit_train(year, inputDim, Y_all, Ypred_cnn_drop_all, Ntrain, ...
    'CNN (Dropout+Momentum) - Train Fit (Solid)');

figure('Name','CNN Dropout - Forecast','NumberTitle','off');
plot_one_step_forecast_solid(year, relNumsNorm, Xtest_all, Ytest_all, ...
    weights_cnn_drop, inputDim, N_H1, N_H2, N_H3, ...
    dropH1, dropH2, dropH3, 1, ...
    'CNN (Dropout+Momentum) - One-step Forecast (Solid)');

%% ========================================================================
%  5) CNN WITHOUT DROPOUT + MOMENTUM
%% ========================================================================
disp('=== CNN NO DROPOUT (5,5,5) + MOMENTUM ===');
weights_cnn_nodrop = initialize_weights(inputDim, N_H1, N_H2, N_H3, INIT_RANGE);
[MSE_cnn_nodrop, weights_cnn_nodrop] = train_network_momentum( ...
    Xtrain_all, Ytrain_all, weights_cnn_nodrop, ...
    inputDim, N_H1, N_H2, N_H3, ...
    LEARNING_RATE_CNN, EPOCHS_CNN, TSS_LIMIT, MOMENTUM);

% Predictions
Ypred_cnn_nodrop_all = predict_network(X_all, weights_cnn_nodrop, ...
    inputDim, N_H1, N_H2, N_H3);
Ypred_cnn_nodrop_tr  = Ypred_cnn_nodrop_all(1:Ntrain);
Ypred_cnn_nodrop_te  = Ypred_cnn_nodrop_all(Ntrain+1:end);

[mseTrain_cnn_nodrop, raeTrain_cnn_nodrop] = compute_metrics(Ytrain_all, Ypred_cnn_nodrop_tr);
[mseTest_cnn_nodrop,  raeTest_cnn_nodrop ] = compute_metrics(Ytest_all,  Ypred_cnn_nodrop_te);

figure('Name','CNN No Dropout - Train Fit','NumberTitle','off');
plot_fit_train(year, inputDim, Y_all, Ypred_cnn_nodrop_all, Ntrain, ...
    'CNN (No Dropout+Momentum) - Train Fit (Solid)');

figure('Name','CNN No Dropout - Forecast','NumberTitle','off');
plot_one_step_forecast_solid(year, relNumsNorm, Xtest_all, Ytest_all, ...
    weights_cnn_nodrop, inputDim, N_H1, N_H2, N_H3, ...
    0,0,0,0, ...
    'CNN (No Dropout+Momentum) - One-step Forecast (Solid)');

%% ========================================================================
%  6) MLP SINGLE HIDDEN, 5 NEURONS, NO PRUNING
%% ========================================================================
disp('=== MLP Single Hidden (5 neurons, 5-lag), NO PRUNING ===');

hiddenMLP_noPrune = 5;
weights_mlp_noPrune = init_weights_mlp(inputDim, hiddenMLP_noPrune, 1, INIT_RANGE);

[MSE_mlp_noPrune, weights_mlp_noPrune] = train_singleMLP( ...
    Xtrain_all, Ytrain_all, weights_mlp_noPrune, ...
    inputDim, hiddenMLP_noPrune, 1, ...
    LEARNING_RATE_MLP, EPOCHS_MLP);

% Predictions
Ypred_mlp_noPrune_all= predict_singleMLP(X_all, weights_mlp_noPrune, ...
    inputDim, hiddenMLP_noPrune, 1);
Ypred_mlp_noPrune_tr = Ypred_mlp_noPrune_all(1:Ntrain);
Ypred_mlp_noPrune_te = Ypred_mlp_noPrune_all(Ntrain+1:end);

[mseTrain_mlp_noPrune, raeTrain_mlp_noPrune] = compute_metrics(Ytrain_all, Ypred_mlp_noPrune_tr);
[mseTest_mlp_noPrune,  raeTest_mlp_noPrune ] = compute_metrics(Ytest_all,  Ypred_mlp_noPrune_te);

figure('Name','MLP No Pruning - Train Fit','NumberTitle','off');
plot_fit_train_mlp(year, inputDim, Y_all, Ypred_mlp_noPrune_all, Ntrain, ...
    'MLP Single Hidden (No Prune) - Train Fit');

figure('Name','MLP No Pruning - Forecast','NumberTitle','off');
plot_one_step_forecast_solid_mlp(year, relNumsNorm, Xtest_all, Ytest_all, ...
    weights_mlp_noPrune, inputDim, hiddenMLP_noPrune, 1, ...
    'MLP Single Hidden (No Prune) - Forecast');

%% ========================================================================
%  7) MLP SINGLE HIDDEN, 5 NEURONS, WITH PRUNING
%% ========================================================================
disp('=== MLP Single Hidden (5 neurons, 5-lag), WITH PRUNING ===');

hiddenMLP_prune = 5;
weights_mlp_prune = init_weights_mlp(inputDim, hiddenMLP_prune, 1, INIT_RANGE);

[MSE_mlp_prune, weights_mlp_prune] = train_singleMLP_withPruning( ...
    Xtrain_all, Ytrain_all, weights_mlp_prune, ...
    inputDim, hiddenMLP_prune, 1, ...
    LEARNING_RATE_MLP, EPOCHS_MLP);

% Predictions
Ypred_mlp_prune_all= predict_singleMLP(X_all, weights_mlp_prune, ...
    inputDim, hiddenMLP_prune, 1);
Ypred_mlp_prune_tr = Ypred_mlp_prune_all(1:Ntrain);
Ypred_mlp_prune_te = Ypred_mlp_prune_all(Ntrain+1:end);

[mseTrain_mlp_prune, raeTrain_mlp_prune] = compute_metrics(Ytrain_all, Ypred_mlp_prune_tr);
[mseTest_mlp_prune,  raeTest_mlp_prune ] = compute_metrics(Ytest_all,  Ypred_mlp_prune_te);

figure('Name','MLP Pruning - Train Fit','NumberTitle','off');
plot_fit_train_mlp(year, inputDim, Y_all, Ypred_mlp_prune_all, Ntrain, ...
    'MLP Single Hidden (Prune) - Train Fit');

figure('Name','MLP Pruning - Forecast','NumberTitle','off');
plot_one_step_forecast_solid_mlp(year, relNumsNorm, Xtest_all, Ytest_all, ...
    weights_mlp_prune, inputDim, hiddenMLP_prune, 1, ...
    'MLP Single Hidden (Prune) - Forecast');

%% ========================================================================
%  8) Plot MSE Over Epochs for all 4
%% ========================================================================
maxLen = max([
    length(MSE_cnn_drop), ...
    length(MSE_cnn_nodrop), ...
    length(MSE_mlp_noPrune), ...
    length(MSE_mlp_prune)
    ]);
padNaN = @(v) [v; NaN(maxLen - length(v),1)];

MSE_cnn_drop_pad    = padNaN(MSE_cnn_drop);
MSE_cnn_noDrop_pad  = padNaN(MSE_cnn_nodrop);
MSE_mlp_noPrune_pad = padNaN(MSE_mlp_noPrune);
MSE_mlp_prune_pad   = padNaN(MSE_mlp_prune);

figure('Name','MSE Over Epochs - All Models','NumberTitle','off');
plot(MSE_cnn_drop,      '-','LineWidth',1.5,'Color','b','DisplayName','CNN + Dropout + Momentum'); hold on;
plot(MSE_cnn_nodrop,    '-','LineWidth',1.5,'Color','r','DisplayName','CNN No Dropout + Momentum');
plot(MSE_mlp_noPrune,   '-','LineWidth',1.5,'Color','m','DisplayName','MLP No Prune');
plot(MSE_mlp_prune,     '-','LineWidth',1.5,'Color','g','DisplayName','MLP Prune');
xlabel('Epoch'); ylabel('MSE');
title('MSE Over Epochs (All 4 Networks) - 5-lag','FontSize',12);
legend('show'); grid on;

%% ========================================================================
%  9) Final Results Table
%% ========================================================================
NetworkName = {
    'CNN+Dropout+Momentum';
    'CNN NoDropout+Momentum';
    'MLP NoPrune';
    'MLP Prune'
};

MSE_Train = [
    mseTrain_cnn_drop;
    mseTrain_cnn_nodrop;
    mseTrain_mlp_noPrune;
    mseTrain_mlp_prune
];
RAE_Train = [
    raeTrain_cnn_drop;
    raeTrain_cnn_nodrop;
    raeTrain_mlp_noPrune;
    raeTrain_mlp_prune
];
MSE_Test = [
    mseTest_cnn_drop;
    mseTest_cnn_nodrop;
    mseTest_mlp_noPrune;
    mseTest_mlp_prune
];
RAE_Test = [
    raeTest_cnn_drop;
    raeTest_cnn_nodrop;
    raeTest_mlp_noPrune;
    raeTest_mlp_prune
];

ResultsTable = table(NetworkName, MSE_Train, RAE_Train, MSE_Test, RAE_Test);
disp('==========================================');
disp(' Final Results Table (Train & Test) - 4 Models ');
disp(ResultsTable);
disp('==========================================');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                      HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%
%  INITIALIZE WEIGHTS  %
%%%%%%%%%%%%%%%%%%%%%%%%
function weights = initialize_weights(nIn,nH1,nH2,nH3,initRange)
    weights= struct();
    % w1
    for j=1:nH1
        for i=1:(nIn+1)
            weights.(sprintf('w1_%d_%d', j,i))= (rand()*2-1)* initRange;
        end
    end
    % w2
    for j=1:nH2
        for i=1:(nH1+1)
            weights.(sprintf('w2_%d_%d', j,i))= (rand()*2-1)* initRange;
        end
    end
    % w3
    for j=1:nH3
        for i=1:(nH2+1)
            weights.(sprintf('w3_%d_%d', j,i))= (rand()*2-1)* initRange;
        end
    end
    % w4
    for i=1:(nH3+1)
        weights.(sprintf('w4_%d', i))= (rand()*2-1)* initRange;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   CNN NO DROPOUT + MOMENTUM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [MSE_over_epochs, weights] = train_network_momentum(...
    Xtrain, Ytrain, weights, ...
    nIn,nH1,nH2,nH3, ...
    learnRate, maxEpochs, tssLimit, momentum)

velocity = initialize_velocity(weights);
Ntrain   = size(Xtrain,1);
MSE_over_epochs= zeros(maxEpochs,1);

for ep=1:maxEpochs
    SSE=0;
    for i=1:Ntrain
        xp= [Xtrain(i,:)';1];
        target= Ytrain(i);

        [y_out,h1,h2,h3] = forward_pass(xp, weights, nIn,nH1,nH2,nH3);
        e= target - y_out;
        SSE= SSE+ e^2;

        grads = backprop_deltas_noDrop(xp, target, weights, h1,h2,h3, ...
                                       nIn,nH1,nH2,nH3);
        velocity = update_velocity(velocity, grads, learnRate, momentum);
        weights  = apply_velocity(weights, velocity);
    end

    MSE= SSE/Ntrain;
    MSE_over_epochs(ep)= MSE;
    fprintf('CNN NoDrop + Momentum Ep %d/%d: SSE=%.4f, MSE=%.4f\n',...
        ep,maxEpochs,SSE,MSE);
    if SSE< tssLimit, break; end
end
MSE_over_epochs= MSE_over_epochs(1:ep);
end

function grads = backprop_deltas_noDrop(xp, target, weights, h1,h2,h3, ...
    nIn,nH1,nH2,nH3)
grads= struct();

% compute final y_out
y_out= 0;
for i=1:(nH3+1)
    w4Name= sprintf('w4_%d', i);
    val_h3= 0;
    if i<=nH3
        val_h3= h3(i);
    else
        val_h3= 1;
    end
    y_out= y_out + weights.(w4Name)* val_h3;
end
Error= target - y_out;

% partial w.r.t w4
for i=1:(nH3+1)
    w4Name= sprintf('w4_%d', i);
    val_h3= 0;
    if i<=nH3
        val_h3= h3(i);
    else
        val_h3= 1;
    end
    grads.(w4Name)= Error * val_h3;
end

% delta h3
delta_h3= zeros(nH3,1);
for j=1:nH3
    w4Name= sprintf('w4_%d', j);
    delta_h3(j)= (1 - h3(j)^2)* Error * weights.(w4Name);
end

% partial w.r.t w3
for jh3=1:nH3
    for ih2=1:(nH2+1)
        w3Name= sprintf('w3_%d_%d', jh3, ih2);
        val_h2= 0;
        if ih2<=nH2
            val_h2= h2(ih2);
        else
            val_h2= 1;
        end
        grads.(w3Name)= delta_h3(jh3)* val_h2;
    end
end

% delta h2
delta_h2= zeros(nH2,1);
for j=1:nH2
    bperr=0;
    for k=1:nH3
        w3Name= sprintf('w3_%d_%d', k,j);
        bperr= bperr+ delta_h3(k)* weights.(w3Name);
    end
    delta_h2(j)= (1- h2(j)^2)* bperr;
end

% partial w.r.t w2
for jh2=1:nH2
    for ih1=1:(nH1+1)
        w2Name= sprintf('w2_%d_%d', jh2, ih1);
        val_h1= 0;
        if ih1<=nH1
            val_h1= h1(ih1);
        else
            val_h1= 1;
        end
        grads.(w2Name)= delta_h2(jh2)* val_h1;
    end
end

% delta h1
delta_h1= zeros(nH1,1);
for j=1:nH1
    bperr=0;
    for k=1:nH2
        w2Name= sprintf('w2_%d_%d', k,j);
        bperr= bperr+ delta_h2(k)* weights.(w2Name);
    end
    delta_h1(j)= (1- h1(j)^2)* bperr;
end

% partial w.r.t w1
for jh1=1:nH1
    for i=1:(nIn+1)
        w1Name= sprintf('w1_%d_%d', jh1, i);
        val_in= 0;
        if i<=nIn
            val_in= xp(i);
        else
            val_in= 1;
        end
        grads.(w1Name)= delta_h1(jh1)* val_in;
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CNN WITH DROPOUT + MOMENTUM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [MSE_over_epochs, weights] = train_network_dropout_momentum( ...
    Xtrain, Ytrain, weights, ...
    nIn,nH1,nH2,nH3, ...
    learnRate, maxEpochs, tssLimit, ...
    dropH1, dropH2, dropH3, momentum)

velocity = initialize_velocity(weights);
MSE_over_epochs= zeros(maxEpochs,1);
Ntrain= size(Xtrain,1);

for ep=1:maxEpochs
    SSE=0;
    for i=1:Ntrain
        xp= [Xtrain(i,:)';1];
        target= Ytrain(i);

        [y_out,h1,h2,h3,mask1,mask2,mask3] = forward_pass_dropout_forMomentum(...
            xp, weights, nIn,nH1,nH2,nH3, dropH1,dropH2,dropH3);

        e= target - y_out;
        SSE= SSE+ e^2;

        grads = backprop_deltas_dropout(xp, target, weights, ...
            h1,h2,h3, mask1,mask2,mask3, nIn,nH1,nH2,nH3);

        velocity = update_velocity(velocity, grads, learnRate, momentum);
        weights  = apply_velocity(weights, velocity);
    end

    MSE= SSE/Ntrain;
    MSE_over_epochs(ep)= MSE;
    fprintf('CNN Dropout + Momentum Ep %d/%d: SSE=%.4f, MSE=%.4f\n', ...
        ep,maxEpochs,SSE,MSE);
    if SSE< tssLimit, break; end
end
MSE_over_epochs= MSE_over_epochs(1:ep);
end

function [y_out,h1,h2,h3,mask1,mask2,mask3] = forward_pass_dropout_forMomentum(...
    xp, weights, nIn,nH1,nH2,nH3, dropH1,dropH2,dropH3)

% hidden 1
net_h1= zeros(nH1,1);
for j=1:nH1
    accum=0;
    for i=1:(nIn+1)
        w1Name= sprintf('w1_%d_%d', j,i);
        accum= accum + weights.(w1Name)* xp(i);
    end
    net_h1(j)= accum;
end
h1_temp= tanh(net_h1);
mask1= (rand(nH1,1)> dropH1);
h1= h1_temp .* mask1;

% hidden 2
h1_bias= [h1; 1];
net_h2= zeros(nH2,1);
for j=1:nH2
    accum=0;
    for i=1:(nH1+1)
        w2Name= sprintf('w2_%d_%d', j,i);
        accum= accum+ weights.(w2Name)* h1_bias(i);
    end
    net_h2(j)= accum;
end
h2_temp= tanh(net_h2);
mask2= (rand(nH2,1)> dropH2);
h2= h2_temp.* mask2;

% hidden 3
h2_bias= [h2; 1];
net_h3= zeros(nH3,1);
for j=1:nH3
    accum=0;
    for i=1:(nH2+1)
        w3Name= sprintf('w3_%d_%d', j,i);
        accum= accum+ weights.(w3Name)* h2_bias(i);
    end
    net_h3(j)= accum;
end
h3_temp= tanh(net_h3);
mask3= (rand(nH3,1)> dropH3);
h3= h3_temp.* mask3;

% output
h3_bias= [h3; 1];
net_o=0;
for i=1:(nH3+1)
    w4Name= sprintf('w4_%d', i);
    net_o= net_o + weights.(w4Name)* h3_bias(i);
end
y_out= net_o;
end

function grads = backprop_deltas_dropout(xp, target, weights, ...
    h1,h2,h3, mask1,mask2,mask3, ...
    nIn,nH1,nH2,nH3)

grads= struct();

% final output
y_out=0;
for i=1:(nH3+1)
    w4Name= sprintf('w4_%d', i);
    val_h3= 0;
    if i<=nH3
        val_h3= h3(i);
    else
        val_h3= 1;
    end
    y_out= y_out+ weights.(w4Name)* val_h3;
end
Error= target - y_out;

% partial w.r.t w4
for i=1:(nH3+1)
    w4Name= sprintf('w4_%d', i);
    val_h3= 0;
    if i<=nH3
        val_h3= h3(i);
    else
        val_h3= 1;
    end
    grads.(w4Name)= Error * val_h3;
end

% delta h3
delta_h3= zeros(nH3,1);
for j=1:nH3
    w4Name= sprintf('w4_%d', j);
    delta_h3(j)= (1- h3(j)^2)* Error* weights.(w4Name);
    if mask3(j)==0
        delta_h3(j)= 0;  
    end
end

% partial w.r.t w3
for jh3=1:nH3
    for ih2=1:(nH2+1)
        w3Name= sprintf('w3_%d_%d', jh3, ih2);
        val_h2= 0;
        if ih2<=nH2
            val_h2= h2(ih2);
        else
            val_h2= 1;
        end
        grads.(w3Name)= delta_h3(jh3)* val_h2;
    end
end

% delta h2
delta_h2= zeros(nH2,1);
for j=1:nH2
    bperr=0;
    for k=1:nH3
        w3Name= sprintf('w3_%d_%d', k,j);
        bperr= bperr+ delta_h3(k)* weights.(w3Name);
    end
    delta_h2(j)= (1- h2(j)^2)* bperr;
    if mask2(j)==0
        delta_h2(j)= 0;
    end
end

% partial w.r.t w2
for jh2=1:nH2
    for ih1=1:(nH1+1)
        w2Name= sprintf('w2_%d_%d', jh2, ih1);
        val_h1= 0;
        if ih1<=nH1
            val_h1= h1(ih1);
        else
            val_h1= 1;
        end
        grads.(w2Name)= delta_h2(jh2)* val_h1;
    end
end

% delta h1
delta_h1= zeros(nH1,1);
for j=1:nH1
    bperr=0;
    for k=1:nH2
        w2Name= sprintf('w2_%d_%d', k,j);
        bperr= bperr+ delta_h2(k)* weights.(w2Name);
    end
    delta_h1(j)= (1- h1(j)^2)* bperr;
    if mask1(j)==0
        delta_h1(j)= 0;
    end
end

% partial w.r.t w1
for jh1=1:nH1
    for i=1:(nIn+1)
        w1Name= sprintf('w1_%d_%d', jh1, i);
        val_in= 0;
        if i<=nIn
            val_in= xp(i);
        else
            val_in= 1;
        end
        grads.(w1Name)= delta_h1(jh1)* val_in;
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   MOMENTUM UTILITIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function velocity = initialize_velocity(weights)
fNames = fieldnames(weights);
velocity= struct();
for f=1:length(fNames)
    velocity.(fNames{f})= 0;
end
end

function velocity = update_velocity(velocity, grads, learnRate, momentum)
gradFields = fieldnames(grads);
for f=1:length(gradFields)
    fn = gradFields{f};
    velocity.(fn) = momentum * velocity.(fn) + learnRate* grads.(fn);
end
end

function weights = apply_velocity(weights, velocity)
vFields= fieldnames(velocity);
for f=1:length(vFields)
    fn= vFields{f};
    weights.(fn)= weights.(fn) + velocity.(fn);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   FORWARD PASS (no dropout)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y_out,h1,h2,h3] = forward_pass(xp, weights, ...
    nIn,nH1,nH2,nH3)
net_h1= zeros(nH1,1);
for j=1:nH1
    accum=0;
    for i=1:(nIn+1)
        w1Name= sprintf('w1_%d_%d', j,i);
        accum= accum + weights.(w1Name)* xp(i);
    end
    net_h1(j)= accum;
end
h1= tanh(net_h1);

h1_bias= [h1;1];
net_h2= zeros(nH2,1);
for j=1:nH2
    accum=0;
    for i=1:(nH1+1)
        w2Name= sprintf('w2_%d_%d', j,i);
        accum= accum+ weights.(w2Name)* h1_bias(i);
    end
    net_h2(j)= accum;
end
h2= tanh(net_h2);

h2_bias= [h2;1];
net_h3= zeros(nH3,1);
for j=1:nH3
    accum=0;
    for i=1:(nH2+1)
        w3Name= sprintf('w3_%d_%d', j,i);
        accum= accum+ weights.(w3Name)* h2_bias(i);
    end
    net_h3(j)= accum;
end
h3= tanh(net_h3);

h3_bias= [h3;1];
net_o=0;
for i=1:(nH3+1)
    w4Name= sprintf('w4_%d', i);
    net_o= net_o+ weights.(w4Name)* h3_bias(i);
end
y_out= net_o;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   PREDICT ENTIRE DATASET (for CNN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prnout= predict_network(Xall, weights, nIn,nH1,nH2,nH3)
N= size(Xall,1);
prnout= zeros(N,1);
for i=1:N
    xp= [Xall(i,:)';1];
    [y_out,~,~,~]= forward_pass(xp, weights, nIn,nH1,nH2,nH3);
    prnout(i)= y_out;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   MLP SINGLE HIDDEN (NO MOMENTUM)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weights_mlp= init_weights_mlp(inDim, hidDim, outDim, initRange)
weights_mlp= struct();
for j=1:hidDim
    for i=1:(inDim+1)
        wName= sprintf('w1_%d_%d', j,i);
        weights_mlp.(wName)= (rand()*2-1)* initRange;
    end
end
for i=1:(hidDim+1)
    wName= sprintf('w2_%d', i);
    weights_mlp.(wName)= (rand()*2-1)* initRange;
end
end

function [MSE_over_epochs, weights_mlp] = train_singleMLP(...
    Xtrain, Ytrain, weights_mlp, inDim, hidDim, outDim, ...
    learnRate, maxEpochs)

Ntrain= size(Xtrain,1);
MSE_over_epochs= zeros(maxEpochs,1);

for ep=1:maxEpochs
    SSE=0;
    for i=1:Ntrain
        xp= [Xtrain(i,:)';1];
        target= Ytrain(i);

        % forward
        net_h1= zeros(hidDim,1);
        for j=1:hidDim
            accum=0;
            for nn=1:(inDim+1)
                wName= sprintf('w1_%d_%d', j,nn);
                accum= accum+ weights_mlp.(wName)* xp(nn);
            end
            net_h1(j)= accum;
        end
        h1= tanh(net_h1);
        h1_bias=[h1;1];

        net_o=0;
        for nn=1:(hidDim+1)
            w2Name= sprintf('w2_%d', nn);
            net_o= net_o+ weights_mlp.(w2Name)* h1_bias(nn);
        end
        y_out= net_o;
        e= target- y_out;
        SSE= SSE+ e^2;

        % backprop
        Beta_out= e;
        delta_h1= zeros(hidDim,1);
        for jh1=1:hidDim
            w2Name= sprintf('w2_%d', jh1);
            delta_h1(jh1)= (1- h1(jh1)^2)* Beta_out* weights_mlp.(w2Name);
        end

        % update w2
        for nn=1:(hidDim+1)
            w2Name= sprintf('w2_%d', nn);
            weights_mlp.(w2Name)= weights_mlp.(w2Name)+ ...
                learnRate* Beta_out* h1_bias(nn);
        end

        % update w1
        for jh1=1:hidDim
            for ii=1:(inDim+1)
                w1Name= sprintf('w1_%d_%d', jh1, ii);
                weights_mlp.(w1Name)= weights_mlp.(w1Name)+ ...
                    learnRate* delta_h1(jh1)* xp(ii);
            end
        end
    end
    MSE_epoch= SSE/Ntrain;
    MSE_over_epochs(ep)= MSE_epoch;
    fprintf('MLP NoPrune Ep %d/%d: SSE=%.4f, MSE=%.4f\n', ep,maxEpochs,SSE,MSE_epoch);
end
end

function [MSE_over_epochs, weights_mlp] = train_singleMLP_withPruning( ...
    Xtrain, Ytrain, weights_mlp, inDim, hidDim, outDim, ...
    learnRate, maxEpochs)

Ntrain= size(Xtrain,1);
MSE_over_epochs= zeros(maxEpochs,1);

for ep=1:maxEpochs
    SSE=0;
    for i=1:Ntrain
        xp= [Xtrain(i,:)';1];
        target= Ytrain(i);

        % forward
        net_h1= zeros(hidDim,1);
        for j=1:hidDim
            accum=0;
            for nn=1:(inDim+1)
                wName= sprintf('w1_%d_%d', j,nn);
                accum= accum+ weights_mlp.(wName)* xp(nn);
            end
            net_h1(j)= accum;
        end
        h1= tanh(net_h1);
        h1_bias=[h1;1];

        net_o=0;
        for nn=1:(hidDim+1)
            w2Name= sprintf('w2_%d', nn);
            net_o= net_o+ weights_mlp.(w2Name)* h1_bias(nn);
        end
        y_out= net_o;
        e= target- y_out;
        SSE= SSE+ e^2;

        % backprop
        Beta_out= e;
        delta_h1= zeros(hidDim,1);
        for jh1=1:hidDim
            w2Name= sprintf('w2_%d', jh1);
            delta_h1(jh1)= (1- h1(jh1)^2)* Beta_out* weights_mlp.(w2Name);
        end

        % update w2
        for nn=1:(hidDim+1)
            w2Name= sprintf('w2_%d', nn);
            weights_mlp.(w2Name)= weights_mlp.(w2Name)+ ...
                learnRate* Beta_out* h1_bias(nn);
        end
        % update w1
        for jh1=1:hidDim
            for ii=1:(inDim+1)
                w1Name= sprintf('w1_%d_%d', jh1, ii);
                weights_mlp.(w1Name)= weights_mlp.(w1Name)+ ...
                    learnRate* delta_h1(jh1)* xp(ii);
            end
        end
    end
    MSE_epoch= SSE/Ntrain;
    MSE_over_epochs(ep)= MSE_epoch;
    fprintf('MLP Prune Ep %d/%d: SSE=%.4f, MSE=%.4f\n', ep,maxEpochs,SSE,MSE_epoch);

    % Pruning step
    fieldNames= fieldnames(weights_mlp);
    for ff=1:length(fieldNames)
        arr= weights_mlp.(fieldNames{ff});
        arr(abs(arr)<0.001)= 0.0;
        weights_mlp.(fieldNames{ff})= arr;
    end
end
end

function yPred= predict_singleMLP(Xall, weights_mlp, inDim, hidDim, outDim)
N= size(Xall,1);
yPred= zeros(N,1);
for i=1:N
    xp= [Xall(i,:)';1];
    net_h1= zeros(hidDim,1);
    for j=1:hidDim
        accum=0;
        for nn=1:(inDim+1)
            wName= sprintf('w1_%d_%d', j,nn);
            accum= accum+ weights_mlp.(wName)* xp(nn);
        end
        net_h1(j)= accum;
    end
    h1= tanh(net_h1);
    h1_bias=[h1;1];

    net_o=0;
    for nn=1:(hidDim+1)
        w2Name= sprintf('w2_%d', nn);
        net_o= net_o+ weights_mlp.(w2Name)* h1_bias(nn);
    end
    yPred(i)= net_o;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   METRICS & PLOT HELPERS (NO DUPLICATE forward_pass_dropout_forMomentum)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [mseVal, raeVal] = compute_metrics(yTrue, yPred)
err = yTrue - yPred;
mseVal = mean(err.^2);
raeVal = sum(abs(err)) / sum(abs(yTrue - mean(yTrue)));
end

function plot_fit_train(yearVec, nLags, Yall, YpredAll, Ntrain, plotTitle)
trainYears  = yearVec(nLags+1 : nLags+Ntrain);
trainActual = Yall(1:Ntrain);
trainPred   = YpredAll(1:Ntrain);

plot(trainYears, trainActual, 'b','LineWidth',1.5); 
hold on;
plot(trainYears, trainPred,   'r','LineWidth',1.5);
xlabel('Year'); 
ylabel('Normalized Sunspot');
title(plotTitle,'FontSize',12);
legend({'Actual','Predicted'}, 'Location','best');
grid on;
end

function plot_fit_train_mlp(yearVec, nLags, Yall, YpredAll, Ntrain, plotTitle)
trainYears  = yearVec(nLags+1 : nLags+Ntrain);
trainActual = Yall(1:Ntrain);
trainPred   = YpredAll(1:Ntrain);

plot(trainYears, trainActual,'b','LineWidth',1.5); 
hold on;
plot(trainYears, trainPred,  'r','LineWidth',1.5);
xlabel('Year'); 
ylabel('Normalized Sunspot');
title(plotTitle,'FontSize',12);
legend({'Actual','Predicted'}, 'Location','best');
grid on;
end

function plot_one_step_forecast_solid(yearVec, dataNorm, ...
    Xtest, Ytest, ...
    weights, nIn, nH1, nH2, nH3, ...
    dropH1, dropH2, dropH3, applyDrop, ...
    plotTitle)

Ntest     = size(Xtest,1);
testYears = yearVec(end-Ntest+1 : end);

hold on;
plot(testYears, Ytest, 'b','LineWidth',1.5, 'DisplayName','Actual Test');

% Predict entire test portion
yPredTest = zeros(Ntest,1);
for i = 1:Ntest
    xp = [Xtest(i,:)'; 1];
    if applyDrop
        % Use the dropout forward pass function
        [pred,~,~,~] = forward_pass_dropout_forMomentum(xp, weights, ...
                            nIn, nH1, nH2, nH3, ...
                            dropH1, dropH2, dropH3);
    else
        % Normal forward pass
        [pred,~,~,~] = forward_pass(xp, weights, ...
                            nIn, nH1, nH2, nH3);
    end
    yPredTest(i) = pred;
end

plot(testYears, yPredTest, 'm','LineWidth',1.5, 'DisplayName','Predicted Test');

lastTestYear   = testYears(end);
lastTestPred   = yPredTest(end);
lastWindow     = Xtest(end,:)';

% SHIFT+APPEND => newWindow = [ lastWindow(2:end); lastTestPred ]
newWindow = [ lastWindow(2:end); lastTestPred ];

if applyDrop
    [forecastVal,~,~,~] = forward_pass_dropout_forMomentum([newWindow; 1], ...
                                weights, nIn, nH1, nH2, nH3, ...
                                dropH1, dropH2, dropH3);
else
    [forecastVal,~,~,~] = forward_pass([newWindow; 1], ...
                                weights, nIn, nH1, nH2, nH3);
end

forecastYear = yearVec(end) + 1;
plot([lastTestYear forecastYear], [lastTestPred forecastVal], ...
    'r','LineWidth',2,'DisplayName','One-step Forecast');

xlabel('Year'); 
ylabel('Normalized Sunspot');
title(plotTitle,'FontSize',12);
legend('show'); 
grid on;
xlim([testYears(end)-5, forecastYear+1]);
end

function plot_one_step_forecast_solid_mlp(yearVec, dataNorm, ...
    Xtest, Ytest, ...
    weights_mlp, inDim, hidDim, outDim, ...
    plotTitle)

Ntest     = size(Xtest,1);
testYears = yearVec(end-Ntest+1 : end);

hold on;
plot(testYears, Ytest,'b','LineWidth',1.5,'DisplayName','Actual Test');

% Predict entire test portion
yPredTest = zeros(Ntest,1);
for i = 1:Ntest
    xp           = [Xtest(i,:)'; 1];
    yPredTest(i) = singleMLP_forward(xp, weights_mlp, inDim, hidDim);
end

plot(testYears, yPredTest, 'm','LineWidth',1.5, 'DisplayName','Predicted Test');

lastTestYear = testYears(end);
lastTestPred = yPredTest(end);
lastWindow   = Xtest(end,:)';

% SHIFT+APPEND => newWindow = [ lastWindow(2:end); lastTestPred ]
newWindow    = [ lastWindow(2:end); lastTestPred ];

forecastVal  = singleMLP_forward([newWindow; 1], weights_mlp, inDim, hidDim);
forecastYear = yearVec(end) + 1;

plot([lastTestYear forecastYear], [lastTestPred forecastVal], ...
    'r','LineWidth',2,'DisplayName','One-step Forecast');

xlabel('Year'); 
ylabel('Normalized Sunspot');
title(plotTitle,'FontSize',12);
legend('show'); 
grid on;
xlim([testYears(end)-5, forecastYear+1]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SINGLE MLP FORWARD HELPER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y_out = singleMLP_forward(x_pattern, weights_mlp, inDim, hidDim)
    % One-forward-pass for a single-hidden-layer MLP with 'hidDim' neurons.
    %
    % x_pattern   : column vector [inDim inputs; +1 bias]
    % weights_mlp : struct of MLP weights
    % inDim, hidDim

    % Hidden Layer
    net_h1 = zeros(hidDim,1);
    for j = 1:hidDim
        accum=0;
        for ii = 1:(inDim+1)
            wName = sprintf('w1_%d_%d', j, ii);
            accum= accum + weights_mlp.(wName)* x_pattern(ii);
        end
        net_h1(j)= accum;
    end
    h1= tanh(net_h1);

    % Output
    h1_bias= [h1;1];
    net_o  = 0;
    for ii=1:(hidDim+1)
        w2Name= sprintf('w2_%d', ii);
        net_o= net_o+ weights_mlp.(w2Name)* h1_bias(ii);
    end
    y_out= net_o;
end
