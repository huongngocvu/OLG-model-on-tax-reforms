clear
clc
close all

% Reference: Huggett, Ventura & Yaron (2011) - Sources of Lifetime Inequality
% 1 decision variable:          s (time for human capital production, and so 1-s is time for working)
% 1 endogenous state variable:  k (physical capital/assets)
% 1 experience asset:           h (human capital)
% 1 exogenous state variable:   z (shock to human capital)
% Order for Return fn: s,aprime,a,h

% First, solve for the stationary equilibrium (by renormalizing c,k,w,T etc.)

%% Setup benchmark model vs others

abilityDifferences = 0;     % =1: agents have different abilities; =0: agents have same ability (set at mean ability)
shocks = 1;                 % =1: idiosyncratic shocks; =0: no idiosyncratic shocks
noBorrowing = 1;            % =1: no borrowing allowed; =0: agents are allowed to borrow
useCohortModel = 1;         % =1: use Cohort effect model; =0: use Time effect model

%% Grids

n_d=21;                     % Grid points for decision variable (s: time for producing human capital/learning)
% n_a=[201,501];              % Grid points for assets: physical capital k and human capital h
n_a=[201,101];              % Grid points for assets: physical capital k and human capital h
n_z=0;                      % Grid points for shocks to earnings/wages
n_u=5;                      % Grid points for shocks to human capital accumulation (5 points using Tauchen method from +-2 std dev)

Params.totalhours=1;
s_grid=linspace(0,Params.totalhours,n_d)';          % Grid on human capital investment time 

max_hgrid=200;
h_grid=linspace(0.1,max_hgrid,n_a(2))';    % Grid on human capital (need to be positive as in log)
 
if noBorrowing==1
    borrowingConstraint=0;
else
    borrowingConstraint=200000;   
end

k_grid=20*max_hgrid*linspace(0,1,n_a(1))'.^3 - borrowingConstraint;   

% Rename in toolkit notation
d_grid=s_grid;              % Grid on decision variable s
a_grid=[k_grid; h_grid];    % Grid on assets
z_grid=[];                  % Grid on shocks to earnings/wages
pi_z=[];                    % Transition matrix of shocks to earnings/wages

%% Parameters

% Demographic

N_j=53;                     % total number of periods (23-75)
Params.agejshifter=22;      % starting age 23
Params.J=N_j;               % the model terminal age (real age: 75)
Params.agej=1:1:Params.J;   % the model period
Params.Jr=43;               % the model retirement age (real age: 65)

% Preferences
% Params.beta=0.981;          % PLACEHOLDER
% Params.beta=Params.beta*(1+Params.g)^(1-Params.sigma);          % RENORMALIZATION
Params.beta=0.9817;         % calibrated beta to make sure eqm interest rate = 0.0581
% Note: before renormalization beta=0.9915 (reported in paper)

Params.sigma=2;             % CES utility parameter 
Params.eta=0.9;             % Weight on consumption in utility fn

Params.n=0.0124;            % population growth rate

% Technology: F(K,LA) = K^alpha*(LA)^(1-alpha)
Params.alpha=0.43;
Params.delta_k=0.067;       % capital depreciation rate (PLACEHOLDER)
Params.g=0.01;            % A_{t+1} = A_t*(1+g)

% Shock in human capital production
Params.mean_u=-0.016;
% Params.mean_u=-0.029;
if shocks==1
    Params.stddev_u=0.160;
else
    Params.stddev_u=0;
end

% Equilibrium values (from GE_1.m)
Params.w=1.4459;
Params.r=0.0582;             
Params.transfer=18.7683; 

% Tax system
Params.lambda=1;            % A parameter to adjust income tax (benchmark)
Params.constant=1;          % constant=1: it has no effect on return fn
Params.tau=0.15;            % consumption tax

% Age distribution (equal proportion for each age)
Params.mewj=ones(1,Params.J);               % Marginal distribution of households over age
for jj=2:length(Params.mewj)                % Population growth
    Params.mewj(jj)=Params.mewj(jj-1)/(1+Params.n);
end
Params.mewj=Params.mewj./sum(Params.mewj);  % Normalize to one
AgeWeightsParamNames={'mewj'};              % So VFI Toolkit knows which parameter is the mass of agents of each age

figure_c=0;     % For numbering figures

%% Permanent types: Ability 
N_i=8;                         % number of different ability levels 

load calib_13.mat
Params.gamma=CalibParams1.gamma;    
Params.mean_logability=CalibParams1.mean_logability;
Params.stddev_logability=CalibParams1.stddev_logability;
Params.mean_logh1=CalibParams1.mean_logh1;
Params.stddev_logh1=CalibParams1.stddev_logh1;
Params.FTcorr_logh1logability=CalibParams1.FTcorr_logh1logability;
Params.delta_h= CalibParams1.delta_h;

if abilityDifferences==1
    [ability_grid,pi_ability]=discretizeAR1_FarmerToda(Params.mean_logability,0,Params.stddev_logability,N_i); %rho = 0 as iid
    ability_grid=exp(ability_grid);
    pi_ability=pi_ability(1,:)';    % iid

    Params.ability=ability_grid;    
    PTypeDistParamNames={'abilitydist'};
    Params.abilitydist=pi_ability;
else
    Params.ability=Params.mean_logability;    
end

%% Distribution of initial conditions (h1,ability) 

% Given the grids on h and ability, compute the probabilities of a bivariate log-normal distribution over the existing grids
if abilityDifferences==1
    Params.mean_logh1logability=[Params.mean_logh1; Params.mean_logability];
    [Params.CorrMatrix_logh1logability, ~ ] = GFT_inverse_mapping(Params.FTcorr_logh1logability, 10^(-9));
    Params.CoVarMatrix_logh1logability = corr2cov([Params.stddev_logh1,Params.stddev_logability],Params.CorrMatrix_logh1logability);

    tic;
    P=gpuArray(MVNormal_ProbabilitiesOnGrid(gather([log(h_grid); log(ability_grid)]),gather(Params.mean_logh1logability), gather(Params.CoVarMatrix_logh1logability), [n_a(2),N_i]));
    mvntime=toc     % P is defined on (h1,ability)
    sum(P,1)        % Make sure all the agent ptypes have positive (non-zero) probabilities [as otherwise this would cause problems]. With N_i=15 the highest and lowest are roughly 10 to minus six

    % The distribution of agents at age j=1 is (k1,h1,ability). Agents are born with zero asset: k1=0
    jequaloneDist=zeros([n_a,N_i],'gpuArray');  % First, put no households anywhere on grid
    jequaloneDist(1,:,:)=P;                     % Second, joint log-normal distribution onto our existing grids, zero assets
else
    P=MVNormal_ProbabilitiesOnGrid(log(h_grid),Params.mean_logh1, Params.stddev_logh1, n_a(2)); % n_a(2) is for human capital
    jequaloneDist=zeros([n_a,1],'gpuArray'); % Put no households anywhere on grid
    jequaloneDist(1,:)=P; % joint log-normal distribution onto our existing grids, zero assets
end

%% Experience asset
if shocks==1
    % Set up experience asset with shock
    vfoptions.experienceassetu=1;
    simoptions.experienceassetu=1;

    % Experience asset h' = exp(u)*H(h,s,ability)
    % aprimeFn: hprime(s,h,u,parameters) gives value of aprime given d2 and a2 (d2 is the decision variable relevant to experience asset, a2 is the experience asset)
    vfoptions.aprimeFn=@(s,h,u,ability,gamma,delta_h) u*(h*(1-delta_h) + ability*(h*s)^gamma);
    simoptions.aprimeFn=vfoptions.aprimeFn;
    simoptions.a_grid=a_grid;
    simoptions.d_grid=d_grid;

    % Shocks in human capital production
    [u_grid,pi_u]=discretizeAR1_Tauchen(Params.mean_u,0,Params.stddev_u,n_u,2); % rho = 0 as shock is iid
    pi_u=pi_u(1,:)'; % iid
    u_grid=exp(u_grid); % switch to exp(u), but normalize grid so it is mean 1 exactly

    vfoptions.n_u=n_u;
    vfoptions.u_grid=u_grid;
    vfoptions.pi_u=pi_u;
    simoptions.n_u=vfoptions.n_u;
    simoptions.u_grid=vfoptions.u_grid;
    simoptions.pi_u=vfoptions.pi_u;

    % Following shows how much h can increase before maxing out. Looking at this there is no max h (there is, but it is large),
    % but because we have finite periods there is going to be a maximum that really just comes from the N_j
    % if abilityDifferences==1
    %     figure_c=figure_c+1;
    %     figure(figure_c)
    %     subplot(3,1,1); plot(h_grid,u_grid(1)*(Params.ability(1)*(h_grid.*1').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     subplot(3,1,2); plot(h_grid,u_grid(ceil(n_u/2))*(Params.ability(ceil(n_u/2))*(h_grid.*1').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     subplot(3,1,3); plot(h_grid,u_grid(end)*(Params.ability(end)*(h_grid.*1').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     legend('human capital prodn','45 degree')
    %     % Same, but use midpoint of s_grid
    %     figure_c=figure_c+1;
    %     figure(figure_c)
    %     subplot(3,1,1); plot(h_grid,u_grid(1)*(Params.ability(1)*(h_grid.*s_grid(ceil(n_d/2))').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     subplot(3,1,2); plot(h_grid,u_grid(ceil(n_u/2))*(Params.ability(ceil(n_u/2))*(h_grid.*s_grid(ceil(n_d/2))').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     subplot(3,1,3); plot(h_grid,u_grid(end)*(Params.ability(end)*(h_grid.*s_grid(ceil(n_d/2))').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     legend('human capital prodn','45 degree')
    % else
    %     figure_c=figure_c+1;
    %     figure(figure_c)
    %     subplot(2,1,1); plot(h_grid,u_grid(1)*(Params.ability*(h_grid.*1').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     subplot(2,1,2); plot(h_grid,u_grid(end)*(Params.ability*(h_grid.*1').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     legend('human capital prodn','45 degree')
    %     % Same, but use midpoint of s_grid
    %     figure_c=figure_c+1;
    %     figure(figure_c)
    %     subplot(2,1,1); plot(h_grid,u_grid(1)*(Params.ability*(h_grid.*s_grid(ceil(n_d/2))').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     subplot(2,1,2); plot(h_grid,u_grid(end)*(Params.ability*(h_grid.*s_grid(ceil(n_d/2))').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    %     legend('human capital prodn','45 degree')
    % end
else
    vfoptions.experienceasset=1;
    simoptions.experienceasset=1;

    % Experience asset h' = exp(mean_u)*H(h,s,ability)
    % aprimeFn: hprime(s,h,u,parameters) gives value of aprime given d2 and a2 (d2 is the decision variable relevant to experience asset, a2 is the experience asset)
    vfoptions.aprimeFn=@(s,h,ability,gamma,delta_h) exp(-0.016)*(h*(1-delta_h) + ability*(h*s)^gamma);
    simoptions.aprimeFn=vfoptions.aprimeFn;
    simoptions.a_grid=a_grid;
    simoptions.d_grid=d_grid;
end

%% Value function

DiscountFactorParamNames={'beta'};

ReturnFn=@(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant)...
    final_model_ReturnFn(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant);

tic;
vfoptions.verbose=1;

if abilityDifferences==1
    [V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
else
    [V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
end

vftime=toc

%% Stationary distribution of households

tic;
simoptions.verbose=1;

if abilityDifferences==1
    StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,[],Params,simoptions);
else
    StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
end

statdisttime=toc

%% Calculate the life-cycle profiles

% @(d,a1prime,a1,a2,...)
FnsToEvaluate.human_capital     =@(s,aprime,a,h,agej,Jr) h*(agej<Jr);
FnsToEvaluate.time_studying     =@(s,aprime,a,h) s;
FnsToEvaluate.time_working      =@(s,aprime,a,h,totalhours) totalhours-s;
FnsToEvaluate.labor_supply      =@(s,aprime,a,h,agej,Jr) h*(agej<Jr)*(1-s);   % Aggregate labor supply
FnsToEvaluate.assets            =@(s,aprime,a,h) a;
FnsToEvaluate.earnings          =@(s,aprime,a,h,w,agej,Jr,totalhours,constant) constant*w*h*(agej<Jr)*(totalhours-s);
FnsToEvaluate.Pensions          =@(s,aprime,a,h,transfer,agej,Jr) (agej>=Jr)*transfer;
FnsToEvaluate.consumption       =@(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant)  final_model_ConsFn(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant);
FnsToEvaluate.incomeTax         =@(s,aprime,a,h,w,agej,Jr,totalhours,transfer,constant) statutory_tax_fn_2010(constant*w*h*(agej<Jr)*(totalhours-s)+(agej>=Jr)*transfer);
FnsToEvaluate.consumptionTax    =@(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant) tau*final_model_ConsFn(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant);
FnsToEvaluate.utility           =@(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant) final_model_ReturnFn(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant); 

tic;

if abilityDifferences==1
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,[],simoptions);
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,[],simoptions);
    AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,[],simoptions);
else
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,[],simoptions);
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,[],simoptions);
    AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,[],simoptions);
end

lcptime=toc;

if abilityDifferences==1 && shocks==1
    save('BM','AgeConditionalStats')
    save('BM_AllStats','AllStats')
end

if abilityDifferences==1 && shocks==0
    save('no_shock','AgeConditionalStats')
    save('no_shock_AllStats','AllStats')
end

if abilityDifferences==0 && shocks==1
    save('no_type','AgeConditionalStats')
    save('no_type_AllStats','AllStats')
end


if abilityDifferences==1 && shocks==1
    % Lifetime utility (value function)
    Udist_ptype=zeros(1,N_i);
    for jj=1:N_i
        typejj=['ptype00',num2str(jj)];
        % Utility
        Udist_jj=V.(typejj).*StationaryDist.(typejj);
        Udist_ptype(jj)=sum(sum(sum(Udist_jj)));
    end
    Udist_avg=Udist_ptype*StationaryDist.ptweights;
end

if abilityDifferences==0 && shocks==1
    Udist=V.*StationaryDist;
    Udist_avg=sum(sum(sum(Udist)));
end


%% Plotting

fig1=figure(1);
subplot(5,2,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.human_capital.Mean)
title('Human capital')
xlim([23,64])
% ylim([0,4000])
subplot(5,2,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.labor_supply.Mean)
title('Labor supply')
xlim([23,64])
subplot(5,2,3); plot(Params.agejshifter+Params.agej, AgeConditionalStats.time_studying.Mean)
title('Time studying')
xlim([23,64])
% ylim([0,0.4])
subplot(5,2,4); plot(Params.agejshifter+Params.agej, AgeConditionalStats.time_working.Mean)
title('Time working')
xlim([23,64])
% ylim([0.6,1.2])
subplot(5,2,5); plot(Params.agejshifter+Params.agej, AgeConditionalStats.assets.Mean)
title('Assets')
xlim([23,75])
subplot(5,2,6); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean+AgeConditionalStats.Pensions.Mean)
title('Total earnings')
xlim([23,75])
subplot(5,2,7); plot(Params.agejshifter+Params.agej, AgeConditionalStats.consumption.Mean)
title('Consumption')
xlim([23,75])
% ylim([0,50000])
subplot(5,2,8); plot(Params.agejshifter+Params.agej, AgeConditionalStats.incomeTax.Mean)
title('Income tax')
xlim([23,75])
subplot(5,2,9); plot(Params.agejshifter+Params.agej, AgeConditionalStats.consumptionTax.Mean)
title('Consumption tax')
xlim([23,75])

if abilityDifferences==1
    if shocks==1
        saveas(fig1,'BM_profiles','epsc')
    else
        saveas(fig1,'no_shock_profiles','epsc')
    end
else
    saveas(fig1,'no_type_profiles','epsc')
end

if abilityDifferences==1
    fig2=figure(2);
    subplot(3,2,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.human_capital.ptype001.Mean,Params.agejshifter+Params.agej, AgeConditionalStats.human_capital.ptype004.Mean,Params.agejshifter+Params.agej, AgeConditionalStats.human_capital.ptype008.Mean)
    title('Human capital')
    xlim([23,75])
    subplot(3,2,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.assets.ptype001.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.assets.ptype004.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.assets.ptype008.Mean)
    title('Assets')
    legend('min ability','mid ability','max ability','Location','best')
    xlim([23,75])
    subplot(3,2,3); plot(Params.agejshifter+Params.agej, AgeConditionalStats.time_studying.ptype001.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.time_studying.ptype004.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.time_studying.ptype008.Mean)
    title('Time studying')
    xlim([23,64])
    ylim([0,0.25])
    subplot(3,2,4); plot(Params.agejshifter+Params.agej, AgeConditionalStats.time_working.ptype001.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.time_working.ptype004.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.time_working.ptype008.Mean)
    title('Time working')
    xlim([23,64])
    ylim([0.75,1.0])
    subplot(3,2,5); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.ptype001.Mean+AgeConditionalStats.Pensions.ptype001.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.earnings.ptype004.Mean+AgeConditionalStats.Pensions.ptype004.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.earnings.ptype008.Mean+AgeConditionalStats.Pensions.ptype008.Mean)
    title('Total earnings')
    xlim([23,75])
    subplot(3,2,6); plot(Params.agejshifter+Params.agej, AgeConditionalStats.consumption.ptype001.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.consumption.ptype004.Mean, Params.agejshifter+Params.agej, AgeConditionalStats.consumption.ptype008.Mean)
    title('Consumption')
    xlim([23,75])
end

if abilityDifferences==1
    if shocks==1
        saveas(fig2,'BM_3types','epsc')
    else
        saveas(fig2,'no_shock_3types','epsc')
    end
end

% Load age profiles data
mean_data = readtable('data_mean.xlsx');
% var_data = readtable('data_var.xlsx');
skew_data = readtable('data_skew.xlsx');
gini_data = readtable('data_gini.xlsx');

if useCohortModel==1
    % mean_profile = 1000*mean_data.Cohort_effects;
    mean_profile = mean_data.Cohort_effects;
    % var_profile = var_data.Cohort_effects;
    skew_profile = skew_data.Cohort_effects;
    gini_profile = gini_data.Cohort_effects;
else
    mean_profile = mean_data.Time_effects;
    % var_profile = var_data.Time_effects;
    skew_profile = skew_data.Time_effects;
    gini_profile = gini_data.Time_effects;
end

% For paper
clear AgeConditionalStats
mean_md=zeros(Params.Jr-5,3);
gini_md=zeros(Params.Jr-5,3);
skew_md=zeros(Params.Jr-5,3);

load BM.mat
mean_md(:,1)=AgeConditionalStats.earnings.Mean(1:Params.Jr-5);
gini_md(:,1)=AgeConditionalStats.earnings.Gini(1:Params.Jr-5);
skew_md(:,1)=AgeConditionalStats.earnings.RatioMeanToMedian(1:Params.Jr-5);
clear AgeConditionalStats

load no_shock.mat
mean_md(:,2)=AgeConditionalStats.earnings.Mean(1:Params.Jr-5);
gini_md(:,2)=AgeConditionalStats.earnings.Gini(1:Params.Jr-5);
skew_md(:,2)=AgeConditionalStats.earnings.RatioMeanToMedian(1:Params.Jr-5);
clear AgeConditionalStats

load no_type.mat
mean_md(:,3)=AgeConditionalStats.earnings.Mean(1:Params.Jr-5);
gini_md(:,3)=AgeConditionalStats.earnings.Gini(1:Params.Jr-5);
skew_md(:,3)=AgeConditionalStats.earnings.RatioMeanToMedian(1:Params.Jr-5);
clear AgeConditionalStats

fig4 = figure(4);
subplot(2,2,1); 
plot(Params.agejshifter+(1:1:(Params.Jr-5)), mean_md(:,2),'--o','Color',[0.4660 0.6740 0.1880],'LineWidth',0.5,'MarkerSize',3)
hold on
plot(Params.agejshifter+(1:1:(Params.Jr-5)), mean_md(:,3),'--+','Color',[0.4660 0.6740 0.1880],'LineWidth',0.5,'MarkerSize',3)
plot(Params.agejshifter+(1:1:(Params.Jr-5)), mean_md(:,1),'-.','Color',[0 0.4470 0.7410],'LineWidth',1.5)
plot(Params.agejshifter+(1:1:(Params.Jr-5)),mean_profile,'r','LineWidth',1.5)
title('Mean earnings')
ax=gca;
ax.FontSize=8;
ylim([0,80])
hold off

subplot(2,2,2); 
plot(Params.agejshifter+(1:1:(Params.Jr-5)), gini_md(:,2),'--o','Color',[0.4660 0.6740 0.1880],'LineWidth',0.5,'MarkerSize',3)
hold on
plot(Params.agejshifter+(1:1:(Params.Jr-5)), gini_md(:,3),'--+','Color',[0.4660 0.6740 0.1880],'LineWidth',0.5,'MarkerSize',3)
plot(Params.agejshifter+(1:1:(Params.Jr-5)), gini_md(:,1),'-.','Color',[0 0.4470 0.7410],'LineWidth',1.5)
plot(Params.agejshifter+(1:1:(Params.Jr-5)),gini_profile,'r','LineWidth',1.5)
title('Earnings Gini')
ax=gca;
ax.FontSize=8;
ylim([0,0.5])
legend('No shock economy','No type economy','Benchmark economy','Data','Location','best')
hold off

subplot(2,2,3); 
plot(Params.agejshifter+(1:1:(Params.Jr-5)), skew_md(:,2),'--o','Color',[0.4660 0.6740 0.1880],'LineWidth',0.5,'MarkerSize',3)
hold on
plot(Params.agejshifter+(1:1:(Params.Jr-5)), skew_md(:,3),'--+','Color',[0.4660 0.6740 0.1880],'LineWidth',0.5,'MarkerSize',3)
plot(Params.agejshifter+(1:1:(Params.Jr-5)), skew_md(:,1),'-.','Color',[0 0.4470 0.7410],'LineWidth',1.5)
plot(Params.agejshifter+(1:1:(Params.Jr-5)),skew_profile,'r','LineWidth',1.5)
title('Earnings skewness')
ax=gca;
ax.FontSize=8;
ylim([0.7,1.6])
hold off

saveas(fig4,'md_vs_data','epsc')

%%
if abilityDifferences==1
    figure_c=figure_c+1;
    figure(figure_c)
    subplot(2,1,1);
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype001.Mean)
    hold on
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype002.Mean)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype003.Mean)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype004.Mean)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype005.Mean)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype006.Mean)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype007.Mean)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype008.Mean)
    hold off
    title('Mean earnings conditional on ptype')
    % Plot median earnings conditional on ptype
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype001.Median)
    hold on
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype002.Median)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype003.Median)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype004.Median)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype005.Median)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype006.Median)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype007.Median)
    plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.earnings.ptype008.Median)
    hold off
    title('Median earnings conditional on ptype')

    figure_c=figure_c+1;
    figure(figure_c)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype001.Mean)
    hold on
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype002.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype003.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype004.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype005.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype006.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype007.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype008.Mean)
    hold off
    title('Mean Income tax conditional on ptype')
    % Plot median humancapital conditional on ptype
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype001.Median)
    hold on
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype002.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype003.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype004.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype005.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype006.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype007.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.incomeTax.ptype008.Median)
    hold off
    title('Median Income tax conditional on ptype')


    figure_c=figure_c+1;
    figure(figure_c)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype001.Mean)
    hold on
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype002.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype003.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype004.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype005.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype006.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype007.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype008.Mean)
    hold off
    title('Mean consumption conditional on ptype')
    % Plot median humancapital conditional on ptype
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype001.Median)
    hold on
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype002.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype003.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype004.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype005.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype006.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype007.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumption.ptype008.Median)
    hold off
    title('Median consumption conditional on ptype')

    figure_c=figure_c+1;
    figure(figure_c)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype001.Mean)
    hold on
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype002.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype003.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype004.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype005.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype006.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype007.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype008.Mean)
    hold off
    title('Mean Consumption tax conditional on ptype')
    % Plot median humancapital conditional on ptype
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype001.Median)
    hold on
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype002.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype003.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype004.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype005.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype006.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype007.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.consumptionTax.ptype008.Median)
    hold off
    title('Median Consumption tax conditional on ptype')


    % Plot mean timel conditional on ptype
    figure_c=figure_c+1;
    figure(figure_c)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype001.Mean)
    hold on
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype002.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype003.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype004.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype005.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype006.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype007.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype008.Mean)
    hold off
    title('Mean time studying (time in human capital creation) conditional on ptype')
    % Plot median timel conditional on ptype
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype001.Median)
    hold on
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype002.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype003.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype004.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype005.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype006.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype007.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.time_studying.ptype008.Median)
    hold off
    title('Median time studying (time in human capital creation) conditional on ptype')

    figure_c=figure_c+1;
    figure(figure_c)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype001.Mean)
    hold on
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype002.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype003.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype004.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype005.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype006.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype007.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype008.Mean)
    hold off
    title('Mean human capital conditional on ptype')
    % Plot median humancapital conditional on ptype
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype001.Median)
    hold on
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype002.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype003.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype004.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype005.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype006.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype007.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.human_capital.ptype008.Median)
    hold off
    title('Median human capital conditional on ptype')

    figure_c=figure_c+1;
    figure(figure_c)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype001.Mean)
    hold on
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype002.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype003.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype004.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype005.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype006.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype007.Mean)
    subplot(2,1,1); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype008.Mean)
    hold off
    title('Mean assets conditional on ptype')
    % Plot median humancapital conditional on ptype
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype001.Median)
    hold on
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype002.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype003.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype004.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype005.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype006.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype007.Median)
    subplot(2,1,2); plot(Params.agejshifter+(1:1:N_j),AgeConditionalStats.assets.ptype008.Median)
    hold off
    title('Median assets conditional on ptype')
end






