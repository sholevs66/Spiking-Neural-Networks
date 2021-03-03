%{
Surrogate gradient spiking RNN for classification tasks in MATLAB
%}

function [V,U,W,b1,b2,acc_train, acc_test] = train_rnn(Lstate, batch_size, epochs)

global T; global th_1;

%% data & spike conversion
% data loading, fill what is needed. xtrain & xtest in [0,1]
ytest = ytest - min(ytest);
ytrain = ytrain - min(ytrain);
Nlabels = size(unique(ytest),1);
Ntest = size(ytest,1);
Ntrain = size(ytrain,1);
ytrain_1_hot = bsxfun(@eq, 1:Nlabels,ytrain+1);     % 1-hot encoding matrix

%% model params
Lout = Nlabels;   % number of labels
Lin = size(xtrain,3);
T = size(xtrain,2);      % rnn time steps (40)

%% other configurations
th_1 = 0.1;
batch_axis = (1:batch_size:Ntrain).';

%% parameters initialization (and state init)
U = single(normrnd(0,1/sqrt(Lstate),[Lstate,Lin]));                                 % [Lstate,Lin]
W = single(unifrnd(-sqrt(3)/sqrt(Lstate),sqrt(3)/sqrt(Lstate),[Lstate,Lstate]));
b1 = single(normrnd(0,0.05,[Lstate,1]));
V = single(unifrnd(-sqrt(3)/sqrt(Lstate),sqrt(3)/sqrt(Lstate), [Lout,Lstate]));
b2 = single(normrnd(0,0.05,[Lout,1]));
s = single(zeros(Lstate,T));


%% optimizer
lr = 1e-4;
dv_m = zeros(size(V));
db2_m = zeros(size(b2));
du_m = zeros(size(U));
dw_m = zeros(size(W));
db1_m = zeros(size(b1));

dv_v = zeros(size(V));
db2_v = zeros(size(b2));
du_v = zeros(size(U));
dw_v = zeros(size(W));
db1_v = zeros(size(b1));
gd_count = 0;

beta1 = 0.98;
beta2 = 0.999;
eps = 1e-8;

%% eval on test set prior to training
acc_train = [];
acc_test = [];
[y_pred_test] = test_predict(xtest, Lstate,V,U,W,b1,b2);
[~,label_pred]=max(y_pred_test);  label_pred = int64(label_pred.'-1);
acc_test = [acc_test;(Ntest - nnz(label_pred - ytest))/Ntest];
fprintf('Test set Acc:    %f\n',acc_test(end));


for q=1:epochs
    
    for b=1:size(batch_axis,1)-1
        x_b = xtrain(batch_axis(b):batch_axis(b+1),:,:);
        y_b = ytrain(batch_axis(b):batch_axis(b+1));    
        y_b_1h = ytrain_1_hot(batch_axis(b):batch_axis(b+1),:);
        [y_pred, ss, sd] = batch_predict(x_b,Lstate,V,U,W,b1,b2);
        
        
        delta2 = y_pred - y_b_1h.';                 % 3,batch
        delta1 = (V')*delta2.*DF1(sd(:,:,T).');     % 30,batch
        
        help = repmat(delta2.',1,1, Lstate);        % batch,3,30
        help2 = permute(repmat(ss(:,:,T),1,1,3), [1,3,2] );  % batch,3,30
        dv = squeeze(mean(help.*help2,1));                      % 3,30
        db2 = mean(delta2,2);                                   % 3,1
        
        
        help = repmat(delta1.',1,1,Lin);        % batch,30,20 = batch,Lstate,Lin
        help2 = repmat(x_b(:,T,:),1,Lstate,1);  % batch,30,20 = batch,Lstate,Lin
        du = squeeze(mean(help.*help2,1));
        
        help = repmat(delta1.',1,1,Lstate);                   % batch,30,30 = batch,Lstate,Lstate
        help2 = permute(repmat(ss(:,:,T),1,1,Lstate), [1,3,2] );  % batch,30,30 = batch,Lstate,Lstate
        dw = squeeze(mean(help.*help2,1));
        
        db1 = mean(delta1,2);
        
        for bptt=T-1:-1:2
            delta1_t_prev = (W')*delta1.*DF1(sd(:,:,bptt).');
            help = repmat(delta1_t_prev.',1,1,Lin);        % batch,30,20 = batch,Lstate,Lin
            help2 = repmat(x_b(:,bptt,:),1,Lstate,1);  % batch,30,20 = batch,Lstate,Lin
            du = du + squeeze(mean(help.*help2,1));
            
            help = repmat(delta1_t_prev.',1,1,Lstate);                   % batch,30,30 = batch,Lstate,Lstate
            help2 = permute(repmat(ss(:,:,bptt),1,1,Lstate), [1,3,2] );  % batch,30,30 = batch,Lstate,Lstate
            dw = dw + squeeze(mean(help.*help2,1));
            
            db1 = db1 + mean(delta1_t_prev,2);
            
            delta1 = delta1_t_prev;
            
        end
        
          
        %% adam try
        gd_count = gd_count + 1;
        % first moments
        db2_m = beta1*db2_m + (1-beta1).*db2;
        db1_m = beta1*db1_m + (1-beta1).*db1;
        dv_m = beta1*dv_m + (1-beta1).*dv;
        du_m = beta1*du_m + (1-beta1).*du;
        dw_m = beta1*dw_m + (1-beta1).*dw;
        
        db2_m_hat = db2_m/(1-beta1^gd_count);
        db1_m_hat = db1_m/(1-beta1^gd_count);
        dv_m_hat = dv_m/(1-beta1^gd_count);
        du_m_hat = du_m/(1-beta1^gd_count);
        dw_m_hat = dw_m/(1-beta1^gd_count);
        
        
        % second moments
        db2_v = beta2*db2_v + (1-beta2).*db2.^2;
        db1_v = beta2*db1_v + (1-beta2).*db1.^2;
        dv_v = beta2*dv_v + (1-beta2).*dv.^2;
        du_v = beta2*du_v + (1-beta2).*du.^2;
        dw_v = beta2*dw_v + (1-beta2).*dw.^2;
        
        db2_v_hat = db2_v/(1-beta2^gd_count);
        db1_v_hat = db1_v/(1-beta2^gd_count);
        dv_v_hat = dv_v/(1-beta2^gd_count);
        du_v_hat = du_v/(1-beta2^gd_count);
        dw_v_hat = dw_v/(1-beta2^gd_count);
%         
        % G.D
        V = V - lr*dv_m_hat ./ (sqrt(dv_v_hat) + eps);
        b2 = b2 - lr*db2_m_hat ./ (sqrt(db2_v_hat) + eps);
        U = U - lr*du_m_hat ./ (sqrt(du_v_hat) + eps);
        W = W - lr*dw_m_hat ./ (sqrt(dw_v_hat) + eps);
        b1 = b1 - lr*db1_m_hat ./ (sqrt(db1_v_hat) + eps);
        
    end
    
    
    % epoch done, calc loss on test data
    [y_pred_test] = test_predict(xtest, Lstate,V,U,W,b1,b2);
    [~,label_pred]=max(y_pred_test);  label_pred = int64(label_pred.'-1);
    acc_test = [acc_test;(Ntest - nnz(label_pred - ytest))/Ntest];
    fprintf('Epoch %d : Test set Acc:    %f\n',q,acc_test(end));
    
    % epoch done, calc loss on train data
    [y_pred_test] = test_predict(xtrain, Lstate,V,U,W,b1,b2);
    [~,label_pred]=max(y_pred_test);  label_pred = int64(label_pred.'-1);
    acc_train = [acc_train;(Ntrain - nnz(label_pred - ytrain))/Ntrain];
    fprintf('Epoch %d : Train set Acc:    %f\n',q,acc_train(end));
    
    
    
    
    
    
    
    
    
end






end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [out, d] = F1(x)
global th_1;
out = x > th_1;
d = x - th_1;
end

function [out] = DF1(x)
    out = (1./sqrt(0.3*pi)) * exp(-x.^2/0.3);
end

function [out] = softmax(x)
out = exp(x);
out = out./sum(out,1);
end



function [y_pred, ss, sd] = batch_predict(x,state_size,V,U,W,b1,b2)
    global T;
    N = size(x,1);
    x = x > rand(size(x));                  % convert to spikes  [N, RNN time steps, d]
    ss = zeros(N, state_size, T);           % [#samples, state size, RNN time steps]
    sd = zeros(N, state_size, T); 
    
    [ss_curr, sd_curr] = F1(U*squeeze(x(:,1,:)).' + b1);   % W is set at 0 so no need it here
    ss(:,:,1) = ss_curr.';
    sd(:,:,1) = sd_curr.';
    for q=2:T
        [ss_curr, sd_curr] = F1(U*squeeze(x(:,q,:)).' + W*ss(:,:,q-1).' + b1);
        ss(:,:,q) = ss_curr.';
        sd(:,:,q) = sd_curr.';
    end
    y_pred = softmax(V*ss(:,:,end).' + b2);
end

function [y_pred] = test_predict(x,state_size,V,U,W,b1,b2)
    global T;
    N = size(x,1);
    x = x > rand(size(x));                  % convert to spikes  [N, RNN time steps, d]
    ss = zeros(N, state_size, T);           % [#samples, state size, RNN time steps]

    ss(:,:,1) = F1(U*squeeze(x(:,1,:)).' + b1).';   % W is set at 0 so no need it here
    for q=2:T
        ss(:,:,q) = F1(U*squeeze(x(:,q,:)).' + W*ss(:,:,q-1).' + b1).';
    end
    y_pred = softmax(V*ss(:,:,end).' + b2);
end



