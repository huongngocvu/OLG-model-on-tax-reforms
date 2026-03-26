
% Reference: Huggett, Ventura & Yaron (2011) - Sources of Lifetime Inequality
% 1 decision variable:          s (time for human capital production, and so 1-s is time for working)
% 1 endogenous state variable:  k (physical capital/assets)
% 1 experience asset:           h (human capital)
% 1 exogenous state variable:   z (shock to human capital)
% Order for Return fn: s,aprime,a,h

% Continue from optimal_tax_MAIN

%% Compensating Variation
% c=1/(1+tau)*(earnings + (1+r)*a - Tax + pension +LumpSum - aprime*(1+g));
% Need to find LumpSum so that when lambda changes, welfare stays the same
% lambda changes -> income tax change -> new eqm: r, tau, transfer

lambda_vec=[0:0.1:0.9,1.1:0.1:1.5];

% GE values (load results from optimal_tax.m)
load welfare_results_all.mat
r_vec=[welfare_results_all(1:10,2);welfare_results_all(12:end,2)];
w_vec=[welfare_results_all(1:10,1);welfare_results_all(12:end,1)];
tau_vec=[welfare_results_all(1:10,3);welfare_results_all(12:end,3)];
transfer_vec=[welfare_results_all(1:10,13);welfare_results_all(12:end,13)];
LumpSum_guess_vec=[1.0743,1.0208,0.9515,0.8704,0.7502,0.6514,0.5205,0.3945,0.2648,0.1331,-0.1269,-0.1230,-0.1200,-0.1150,-0.1132];

CV=struct(); % to store outputs 
for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i); 
    CV.(fieldName) = []; 
end

Params0=Params;
for i=1:length(lambda_vec)
    Params0.lambda=lambda_vec(i);
    Params0.r=r_vec(i);
    Params0.w=w_vec(i);
    Params0.tau=tau_vec(i);
    Params0.transfer=transfer_vec(i);
    Params0.LumpSum_guess=LumpSum_guess_vec(i);

    fieldName = sprintf('loop%d', i);

    out_temp=compensating_variation_TEST(Params0,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,pi_z,DiscountFactorParamNames,AgeWeightsParamNames,PTypeDistParamNames,jequaloneDist,ReturnFn,heteroagentoptions,vfoptions,simoptions);
    CV.(fieldName)=out_temp;
end

%% Collect outputs

CV_results=zeros(length(lambda_vec),14); 

for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i);
    data_here=CV.(fieldName);

    CV_results(i,1)=data_here.Params.LumpSum;         % LumpSum
    CV_results(i,2)=data_here.LumpSum_agg;         % LumpSum
    CV_results(i,3)=data_here.LumpSumRatio;    % LumpSum-to-output
    CV_results(i,4)=data_here.Udist_avg;     % Welfare
    CV_results(i,5)=data_here.w;
    CV_results(i,6)=data_here.r;
    CV_results(i,7)=data_here.tau;
    CV_results(i,8)=data_here.L;
    CV_results(i,9)=data_here.H;
    CV_results(i,10)=data_here.C-AggVars.consumptionTax.Mean;
    CV_results(i,11)=data_here.K;
    CV_results(i,12)=data_here.Y;
    CV_results(i,13)=data_here.Earnings+AggVars.Pensions.Mean-AggVars.incomeTax.Mean;
    CV_results(i,14)=data_here.K/(data_here.Earnings+AggVars.Pensions.Mean-AggVars.incomeTax.Mean);
end

save('CV_results_TEST.mat','CV_results')

% Add benchmark
load welfare_results_all.mat
CV_results_all=[CV_results(1:10,:);zeros(1,14);CV_results(11:end,:)];
CV_results_all(11,4)=welfare_results_all(11,12);     % Welfare
CV_results_all(11,5)=welfare_results_all(11,1);     % w
CV_results_all(11,6)=welfare_results_all(11,2);     % r
CV_results_all(11,7)=welfare_results_all(11,3);     % tau
CV_results_all(11,8)=welfare_results_all(11,4);     % L
CV_results_all(11,9)=welfare_results_all(11,5);     % H
CV_results_all(11,10)=welfare_results_all(11,6);     % C
CV_results_all(11,11)=welfare_results_all(11,7);    % K
CV_results_all(11,12)=welfare_results_all(11,8);    % Y
CV_results_all(11,13)=welfare_results_all(11,9);    % Earnings after tax
CV_results_all(11,14)=welfare_results_all(11,10);   % K/E-after-tax

save('CV_results_TEST_all.mat','CV_results_all')

% Plotting
figure()
plot(CV_results_all(:,[2,3]))
hold on
plot(zeros(1,16))
hold off
legend('LumpSum','LumpSum-to-output')

welfare_gain_fig=figure();
stem(0:0.1:1.5,CV_results_all(:,3),'LineWidth',1.5)
xlabel('\lambda')
ylabel('Compensating wealth (% of output)')

saveas(welfare_gain_fig,'welfare_gain_fig','epsc')