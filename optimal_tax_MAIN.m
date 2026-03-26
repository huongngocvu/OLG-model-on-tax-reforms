clear
clc
close all

% Reference: Huggett, Ventura & Yaron (2011) - Sources of Lifetime Inequality
% 1 decision variable:          s (time for human capital production, and so 1-s is time for working)
% 1 endogenous state variable:  k (physical capital/assets)
% 1 experience asset:           h (human capital)
% 1 exogenous state variable:   z (shock to human capital)
% Order for Return fn: s,aprime,a,h

% Code for the Optimal tax section (GitHub version)
% Added a parameter, lambda, to scale income tax rates

%% Setup benchmark model vs others

noBorrowing = 1;            % =1: no borrowing allowed; =0: agents are allowed to borrow
forServer = 1;              % =1: run on server; =0: run on laptop
useGridInterp = 1;          % =1 if using grid interpolation

if forServer == 1
    addpath(genpath('./VFIToolkit-matlab-master/'))
    if useGridInterp == 1
        vfoptions.gridinterplayer=1;
        vfoptions.ngridinterp=20;
        simoptions.gridinterplayer=vfoptions.gridinterplayer;
        simoptions.ngridinterp=vfoptions.ngridinterp;
    end
else
    if useGridInterp == 1
        vfoptions.gridinterplayer=1;
        vfoptions.ngridinterp=10;
        simoptions.gridinterplayer=vfoptions.gridinterplayer;
        simoptions.ngridinterp=vfoptions.ngridinterp;
    end
end

vfoptions.divideandconquer=1;

%% Grids

n_d=21;                     % Grid points for decision variable (s: time for producing human capital/learning)
n_a=[201,101];              % Grid points for assets: physical capital k and human capital h
% n_a=[501,101];              % Grid points for assets: physical capital k and human capital h
n_z=0;                      % Grid points for shocks to earnings/wages
n_u=5;                      % Grid points for shocks to human capital accumulation (5 points using Tauchen method from +-2 std dev)

s_grid=linspace(0,1,n_d)';          % Grid on human capital investment time 
h_grid=linspace(0.1,200,n_a(2))';    % Grid on human capital (need to be positive as in log)
maxasset=max(h_grid)*20;

if noBorrowing==1
    borrowingConstraint=0;
else
    borrowingConstraint=200;   
end
k_grid=maxasset*linspace(0,1,n_a(1))'.^3 - borrowingConstraint;   

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
% Params.beta=Params.beta*(1+Params.g)^(1-Params.sigma);          % RENORMALIZATION
Params.beta=0.9817;         % Note: before renormalization beta=0.9915 (reported in paper)
Params.sigma=2;             % CES utility parameter 
Params.n=0.0124;            % population growth rate

% Technology: F(K,LA) = K^alpha*(LA)^(1-alpha)
Params.alpha=0.43;
Params.delta_k=0.067;       % capital depreciation rate (PLACEHOLDER)
Params.g=0.01;            % A_{t+1} = A_t*(1+g)

% Shock in human capital production
Params.mean_u=-0.016;
Params.stddev_u=0.160;

% Tax system
Params.tau=0.15;            % consumption tax
Params.pension_ratio=0.2;   % fraction of gov revenues to fund pensions

% Age distribution (equal proportion for each age)
Params.mewj=ones(1,Params.J);               % Marginal distribution of households over age
for jj=2:length(Params.mewj)                % Population growth
    Params.mewj(jj)=Params.mewj(jj-1)/(1+Params.n);
end
Params.mewj=Params.mewj./sum(Params.mewj);  % Normalize to one
AgeWeightsParamNames={'mewj'};              % So VFI Toolkit knows which parameter is the mass of agents of each age

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

[ability_grid,pi_ability]=discretizeAR1_FarmerToda(Params.mean_logability,0,Params.stddev_logability,N_i); %rho = 0 as iid
ability_grid=exp(ability_grid);
pi_ability=pi_ability(1,:)';    % iid

Params.ability=ability_grid;
PTypeDistParamNames={'abilitydist'};
Params.abilitydist=pi_ability;

%% Distribution of initial conditions (h1,ability) 

Params.mean_logh1logability=[Params.mean_logh1; Params.mean_logability];
[Params.CorrMatrix_logh1logability, ~ ] = GFT_inverse_mapping(Params.FTcorr_logh1logability, 10^(-9));
Params.CoVarMatrix_logh1logability = corr2cov([Params.stddev_logh1,Params.stddev_logability],Params.CorrMatrix_logh1logability);

% Given the grids on h and ability, compute the probabilities of a bivariate log-normal distribution over the existing grids
P=gpuArray(MVNormal_ProbabilitiesOnGrid(gather([log(h_grid); log(ability_grid)]),gather(Params.mean_logh1logability), gather(Params.CoVarMatrix_logh1logability), [n_a(2),N_i]));
sum(P,1);        % Make sure all the agent ptypes have positive (non-zero) probabilities [as otherwise this would cause problems]. With N_i=15 the highest and lowest are roughly 10 to minus six

% The distribution of agents at age j=1 is (k1,h1,ability). Agents are born with zero asset: k1=0
jequaloneDist=zeros([n_a,N_i],'gpuArray');  % First, put no households anywhere on grid
jequaloneDist(1,:,:)=P;                     % Second, joint log-normal distribution onto our existing grids, zero assets

%% Experience asset
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
if forServer==0
    figure_c=0;     % For numbering figures
    figure_c=figure_c+1;
    figure(figure_c)
    subplot(3,1,1); plot(h_grid,u_grid(1)*(Params.ability(1)*(h_grid.*1').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    subplot(3,1,2); plot(h_grid,u_grid(ceil(n_u/2))*(Params.ability(ceil(n_u/2))*(h_grid.*1').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    subplot(3,1,3); plot(h_grid,u_grid(end)*(Params.ability(end)*(h_grid.*1').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    legend('human capital prodn','45 degree')
    % Same, but use midpoint of s_grid
    figure_c=figure_c+1;
    figure(figure_c)
    subplot(3,1,1); plot(h_grid,u_grid(1)*(Params.ability(1)*(h_grid.*s_grid(ceil(n_d/2))').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    subplot(3,1,2); plot(h_grid,u_grid(ceil(n_u/2))*(Params.ability(ceil(n_u/2))*(h_grid.*s_grid(ceil(n_d/2))').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    subplot(3,1,3); plot(h_grid,u_grid(end)*(Params.ability(end)*(h_grid.*s_grid(ceil(n_d/2))').^Params.gamma+h_grid.*(1-Params.delta_h)), h_grid, h_grid)
    legend('human capital prodn','45 degree')
end

%% Return fn

DiscountFactorParamNames={'beta'};

ReturnFn=@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum)...
    optimal_tax_ReturnFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);

%% Initial guess
Params.lambda=1;            % A parameter to adjust income tax
Params.LumpSum=0;           % In baseline LumpSum=0

Params.r=0.0581;  
Params.w=(1-Params.alpha)*((Params.r+Params.delta_k)/Params.alpha)^(Params.alpha/(Params.alpha-1));
Params.transfer=18.8094; 

%% Solve benchmark model
% vfoptions.verbose=1;
% simoptions.verbose=1;
heteroagentoptions.toleranceGEcondns=10^(-6);
heteroagentoptions.toleranceGEprices=10^(-6);

[~, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);

% Convert Policy
if useGridInterp == 1
    PolicyVals=PolicyInd2Val_Case1_FHorz_PType(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions);
end

StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,[],Params,simoptions);

GEPriceParamNames={'r','transfer'};

% Aggregates: @(d,a1prime,a1,a2,...)
FnsToEvaluate.L     =@(s,aprime,a,h) h*(1-s);   % Aggregate labor supply
FnsToEvaluate.K     =@(s,aprime,a,h) a;         % Aggregate physical capital
FnsToEvaluate.Pensions          =@(s,aprime,a,h,transfer,agej,Jr) (agej>=Jr)*transfer;
FnsToEvaluate.incomeTax         =@(s,aprime,a,h,w,agej,Jr,transfer,lambda) lambda*statutory_tax_fn_2010(w*h*(agej<Jr)*(1-s)+(agej>=Jr)*transfer);
FnsToEvaluate.consumptionTax    =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) tau*optimal_tax_ConsFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);

% GE conditions
GeneralEqmEqns.capitalmarket=@(r,K,L,alpha,delta_k) r-alpha*(K/L)^(alpha-1)+delta_k;
GeneralEqmEqns.govbudget=@(Pensions,incomeTax,consumptionTax,pension_ratio) Pensions-pension_ratio*(incomeTax+consumptionTax);

% Solve for the General Equilibrium
heteroagentoptions.verbose=1;
[p_eqm,~,GeneralEqmEqnsValues]=HeteroAgentStationaryEqm_Case1_FHorz_PType(n_d, n_a, n_z, N_j, N_i, [], pi_z, d_grid, a_grid, z_grid,jequaloneDist, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);

% Put this into Params so we can calculate things about the initial equilibrium
Params.r=p_eqm.r;
Params.transfer=p_eqm.transfer;
Params.w=(1-Params.alpha)*((Params.r+Params.delta_k)/Params.alpha)^(Params.alpha/(Params.alpha-1));

%% Preparing for optimal tax experiments

% Save the benchmark GE prices
Params.benchmarkR=Params.r;
Params.benchmarkTransfer=Params.transfer;
Params.benchmarkW=Params.w;

% Calculate a few things related to the general equilibrium

[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,[],Params,simoptions);

% Some variables to evaluate
% @(d,a1prime,a1,a2,...)
clear FnsToEvaluate
FnsToEvaluate.human_capital     =@(s,aprime,a,h,agej,Jr) h*(agej<Jr);
FnsToEvaluate.time_studying     =@(s,aprime,a,h) s;
FnsToEvaluate.time_working      =@(s,aprime,a,h) 1-s;
FnsToEvaluate.labor_supply      =@(s,aprime,a,h,agej,Jr) h*(agej<Jr)*(1-s);   % Aggregate labor supply
FnsToEvaluate.assets            =@(s,aprime,a,h) a;
FnsToEvaluate.earnings          =@(s,aprime,a,h,w,agej,Jr) w*h*(agej<Jr)*(1-s);
FnsToEvaluate.Pensions          =@(s,aprime,a,h,transfer,agej,Jr) (agej>=Jr)*transfer;
FnsToEvaluate.consumption       =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) optimal_tax_ConsFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);
FnsToEvaluate.incomeTax         =@(s,aprime,a,h,w,agej,Jr,transfer,lambda) lambda*statutory_tax_fn_2010(w*h*(agej<Jr)*(1-s)+(agej>=Jr)*transfer);
FnsToEvaluate.consumptionTax    =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) tau*optimal_tax_ConsFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);
FnsToEvaluate.Utility           =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) optimal_tax_ReturnFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);

AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,[],simoptions);

% Calculate aggregates
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Params, n_d, n_a, n_z,N_j,N_i, d_grid, a_grid, z_grid,simoptions);

H=AggVars.human_capital.Mean;
S=AggVars.time_studying.Mean;
L=AggVars.labor_supply.Mean;
K=AggVars.assets.Mean;
Earnings=AggVars.earnings.Mean;
C=AggVars.consumption.Mean;
Y=(K^Params.alpha)*(L^(1-Params.alpha));  % GDP
TaxtoGDP=(AggVars.incomeTax.Mean+AggVars.consumptionTax.Mean)/Y; 
Params.benchmarkTaxratio=TaxtoGDP;   % save as a param so that can be used later

Udist_ptype=zeros(1,N_i);
for jj=1:N_i
    if jj<10
        typejj=['ptype00',num2str(jj)];
    elseif jj<20
        typejj=['ptype0',num2str(jj)];
    end
    % Utility
    Udist_jj=V.(typejj).*StationaryDist.(typejj);
    Udist_ptype(jj)=sum(sum(sum(Udist_jj)));

end

Udist_avg=Udist_ptype*StationaryDist.ptweights; 

%% Compute new GE given new value of lambda

lambda_vec=[0:0.1:0.9,1.1:0.1:1.5];

% Use better initial guess (got these by solving eqm several times)
r_vec=[0.0456,0.0468,0.0480,0.0492,0.0503,0.0516,0.0529,0.0542,0.0555,0.0569,0.0597,0.0611,0.0626,0.0641,0.0656];
tau_vec=[0.3999,0.3758,0.3514,0.3267,0.3019,0.2769,0.2517,0.2264,0.2010,0.1755,0.1246,0.0991,0.0738,0.0485,0.0233];
transfer_vec=[20.1156,19.9680,19.8162,19.6672,19.5366,19.3868,19.2373,19.0817,18.9238,18.7622,18.4426,18.2828,18.1304,17.9759,17.8174];

% Note: running time is long, so may need to break the computation of Output into 2 parts
Output=struct(); % to store outputs 
for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i); 
    Output.(fieldName) = []; 
end

Params0=Params;
for i=1:length(lambda_vec)
    Params0.lambda=lambda_vec(i);
    Params0.tau=tau_vec(i);
    Params0.r=r_vec(i);
    Params0.transfer=transfer_vec(i);

    fieldName = sprintf('loop%d', i);

    out_temp=compute_new_welfare(Params0,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,pi_z,DiscountFactorParamNames,AgeWeightsParamNames,PTypeDistParamNames,jequaloneDist,ReturnFn,heteroagentoptions,vfoptions,simoptions);
    Output.(fieldName)=out_temp;
end


%% Plotting prices and aggregates
% This section plots prices and aggregates across lambda values
% x-axis: lambda from 0-1.5
% y-axis: all variables in variableNames

red = [0.8500, 0.3250, 0.0980];
blue = [0.0000, 0.4470, 0.7410];
yellow = [0.9290, 0.6940, 0.1250];
purple = [0.4940, 0.1840, 0.5560];
green = [0.4660 0.6740 0.1880];

variableNames={'Wage','Interest rate','Consumption tax rate',...
    'Labor supply','Human capital','Consumption','Assets','Output','After-tax-earnings','K/E ratio',...
    'Pension','Welfare','transfer','Time working','Time studying',...
    'Earnings','Income tax'};
n_variables=length(variableNames);
n_experiments=length(lambda_vec);

welfare_results=zeros(n_experiments,n_variables);

for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i);
    data_here=Output.(fieldName);

    welfare_results(i,1)=data_here.w;
    welfare_results(i,2)=data_here.r;
    welfare_results(i,3)=data_here.tau;
    welfare_results(i,4)=data_here.L;
    welfare_results(i,5)=data_here.H;
    welfare_results(i,6)=data_here.C;
    welfare_results(i,7)=data_here.K;
    welfare_results(i,8)=data_here.Y;
    welfare_results(i,9)=data_here.Earnings+data_here.Pensions-data_here.incomeTax;
    welfare_results(i,10)=data_here.K/(data_here.Earnings+data_here.Pensions-data_here.incomeTax);
    welfare_results(i,11)=data_here.Pensions;
    welfare_results(i,12)=data_here.Udist_avg;
    welfare_results(i,13)=data_here.transfer;
    welfare_results(i,14)=data_here.AggVars.timeWorking.Mean;
    welfare_results(i,15)=data_here.AggVars.timeStudying.Mean;
    welfare_results(i,16)=data_here.Earnings+data_here.Pensions;
    welfare_results(i,17)=data_here.incomeTax;
end

% Add benchmark lambda=1
welfare_results_all=zeros(n_experiments+1,n_variables);

welfare_results_all(1:10,:)=welfare_results(1:10,:);
welfare_results_all(12:end,:)=welfare_results(11:end,:);

welfare_results_all(11,1)=Params.w;
welfare_results_all(11,2)=Params.r;
welfare_results_all(11,3)=Params.tau;
welfare_results_all(11,4)=L;
welfare_results_all(11,5)=H;
welfare_results_all(11,6)=C;
welfare_results_all(11,7)=K;
welfare_results_all(11,8)=Y;
welfare_results_all(11,9)=Earnings+AggVars.Pensions.Mean-AggVars.incomeTax.Mean;
welfare_results_all(11,10)=K/(Earnings+AggVars.Pensions.Mean-AggVars.incomeTax.Mean);
welfare_results_all(11,11)=AggVars.Pensions.Mean;
welfare_results_all(11,12)=Udist_avg;
welfare_results_all(11,13)=Params.transfer;
welfare_results_all(11,14)=AggVars.time_working.Mean;
welfare_results_all(11,15)=AggVars.time_studying.Mean;
welfare_results_all(11,16)=Earnings+AggVars.Pensions.Mean;
welfare_results_all(11,17)=AggVars.incomeTax.Mean;

save('welfare_results_all.mat','welfare_results_all')

% Figure: all prices and aggregates
welfare_fig=figure();
for i=[1,2,3,4,5,6,7,8,9,10,11,12,14,15]
    if i==14 || i==15
        subplot(ceil((n_variables-1)/4),4,i-10)
    elseif i>=4 && i<=12
        subplot(ceil((n_variables-1)/4),4,i+2)
    else
        subplot(ceil((n_variables-1)/4),4,i)
    end

    plot(0:0.1:1.5,welfare_results_all(:,i))
    if i==5
        ylim([32.1,32.2])
    end

    title(variableNames{i})
    xlabel('\lambda')
end

% saveas(welfare_fig,'welfare_fig','epsc')

%% Welfare by types
% This section plots welfare (value fn) of 8 types across lambda values
% x-axis: lambda from 0-1.5
% y-axis: type i welfare, average welfare

welfare_types=zeros(n_experiments,N_i);

for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i);
    data_here=Output.(fieldName);

    welfare_types(i,:)=data_here.Udist_ptype;
end

% Add benchmark lambda=1
welfare_types_all=zeros(n_experiments+1,N_i);

welfare_types_all(1:10,:)=welfare_types(1:10,:);
welfare_types_all(12:end,:)=welfare_types(11:end,:);

welfare_types_all(11,:)=Udist_ptype;

% Add average welfare (i.e. average utility or value fn)
welfare_types_all=[welfare_types_all welfare_results_all(:,12)];

% Figure: welfare by types
welfare_type_fig1=figure();
plot(0:0.1:1.5,welfare_types_all(:,1),'Marker','square','MarkerSize',5)
hold on
plot(0:0.1:1.5,welfare_types_all(:,[2,3]))
plot(0:0.1:1.5,welfare_types_all(:,4),'Marker','o','MarkerSize',5)
plot(0:0.1:1.5,welfare_types_all(:,[5,6,7]))
plot(0:0.1:1.5,welfare_types_all(:,8),'Marker','*','MarkerSize',5)
plot(0:0.1:1.5,welfare_types_all(:,end),'LineWidth',2)
hold off
legend('Lowest ability','','','Median ability','','','','Highest ability','Average welfare','Location','best')
xlabel('\lambda')
ylabel('Welfare (Average utility)')
% saveas(welfare_type_fig1,'welfare_type_fig','epsc')

% % Figure: lowest, median, highest ability
% welfare_type_fig2=figure();
% subplot(2,2,1)
% plot(0:0.1:1.5,welfare_types_all(:,1))
% title('Lowest ability')
% ylim([-3.4,-2.4])
% 
% subplot(2,2,2)
% plot(0:0.1:1.5,welfare_types_all(:,4))
% title('Median ability')
% ylim([-0.94,-0.84])
% 
% subplot(2,2,3)
% plot(0:0.1:1.5,welfare_types_all(:,8))
% title('Highest ability')
% ylim([-0.25,-0.15])
% 
% subplot(2,2,4)
% plot(0:0.1:1.5,welfare_types_all(:,9))
% title('Average welfare')
% ylim([-0.83,-0.73])

%% Welfare as function of age
% This section plots welfare profiles
% x-axis: age
% y-axis: (average) welfare conditional on age
% Welfare (i.e. average utility) is calculated from value fn and stationary
% distribution

welfare_profiles=zeros(n_experiments,N_j);

for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i);
    data_here=Output.(fieldName);

    V_here=data_here.V;
    StationaryDist_here=data_here.StationaryDist;
    weights_here=StationaryDist_here.ptweights;

    Udist_by_type=zeros(N_i,N_j);
    for jj=1:N_i
        if jj<10
            typejj=['ptype00',num2str(jj)];
        elseif jj<20
            typejj=['ptype0',num2str(jj)];
        end
        % Utility
        Udist_jj=V_here.(typejj).*StationaryDist_here.(typejj);
        % Udist_jj=Udist_jj(~isnan(Udist_jj)); % This will change the shape of Udist_jj (it works fine with 2D matrix, but Udist_jj is 3D)
        Udist_by_type(jj,:)=sum(sum(Udist_jj));
    end

    welfare_profiles(i,:)=weights_here'*Udist_by_type;
end

% Add benchmark
welfare_profiles_all=zeros(n_experiments+1,N_j);

welfare_profiles_all(1:10,:)=welfare_profiles(1:10,:);
welfare_profiles_all(12:end,:)=welfare_profiles(11:end,:);

Udist_type_BM=zeros(N_i,N_j);
for jj=1:N_i
    if jj<10
        typejj=['ptype00',num2str(jj)];
    elseif jj<20
        typejj=['ptype0',num2str(jj)];
    end
    % Utility
    Udist_jj_BM=V.(typejj).*StationaryDist.(typejj);
    % Udist_jj=Udist_jj(~isnan(Udist_jj)); % This will change the shape of Udist_jj (it works fine with 2D matrix, but Udist_jj is 3D)
    Udist_type_BM(jj,:)=sum(sum(Udist_jj_BM));
end

welfare_profiles_all(11,:)=weights_here'*Udist_type_BM;

welfare_profile_fig=figure();
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
legend('\lambda=0.5','\lambda=1','\lambda=1.5','Location','best')
xlabel('Age')
ylabel('Average utility (Value function)')
xlim([30,45])

% saveas(welfare_profile_fig,'welfare_profile_fig','epsc')

%% Age profiles of consumption, assets, after-tax-earnings, human capital, labor supply
% This section plots age profiles of various variables
% x-axis: age
% y-axis: aggregate value conditional on age

consumption_profiles=zeros(n_experiments,N_j);
consTax_profiles=zeros(n_experiments,N_j);
incTax_profiles=zeros(n_experiments,N_j);
asset_profiles=zeros(n_experiments,N_j);
income_profiles=zeros(n_experiments,N_j);
h_profiles=zeros(n_experiments,N_j);
l_profiles=zeros(n_experiments,N_j);
timeWorking_profiles=zeros(n_experiments,N_j);
timeStudying_profiles=zeros(n_experiments,N_j);

for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i);
    data_here=Output.(fieldName);

    consumption_profiles(i,:)=data_here.AgeConditionalStats.Consumption.Mean;
    consTax_profiles(i,:)=data_here.AgeConditionalStats.consumptionTax.Mean;
    incTax_profiles(i,:)=data_here.AgeConditionalStats.incomeTax.Mean;
    asset_profiles(i,:)=data_here.AgeConditionalStats.K.Mean;
    income_profiles(i,:)=data_here.AgeConditionalStats.Earnings.Mean+data_here.AgeConditionalStats.Pensions.Mean;
    h_profiles(i,:)=data_here.AgeConditionalStats.H.Mean;
    l_profiles(i,:)=data_here.AgeConditionalStats.L.Mean;
    timeWorking_profiles(i,:)=data_here.AgeConditionalStats.timeWorking.Mean;
    timeStudying_profiles(i,:)=data_here.AgeConditionalStats.timeStudying.Mean;
end

% Add benchmark
consumption_profiles_all=zeros(n_experiments+1,N_j);
consTax_profiles_all=zeros(n_experiments+1,N_j);
incTax_profiles_all=zeros(n_experiments+1,N_j);
asset_profiles_all=zeros(n_experiments+1,N_j);
income_profiles_all=zeros(n_experiments+1,N_j);
h_profiles_all=zeros(n_experiments+1,N_j);
l_profiles_all=zeros(n_experiments+1,N_j);
timeWorking_profiles_all=zeros(n_experiments+1,N_j);
timeStudying_profiles_all=zeros(n_experiments+1,N_j);

consumption_profiles_all(1:10,:)=consumption_profiles(1:10,:);
consumption_profiles_all(11,:)=AgeConditionalStats.consumption.Mean;
consumption_profiles_all(12:end,:)=consumption_profiles(11:end,:);

consTax_profiles_all(1:10,:)=consTax_profiles(1:10,:);
consTax_profiles_all(11,:)=AgeConditionalStats.consumptionTax.Mean;
consTax_profiles_all(12:end,:)=consTax_profiles(11:end,:);

incTax_profiles_all(1:10,:)=incTax_profiles(1:10,:);
incTax_profiles_all(11,:)=AgeConditionalStats.incomeTax.Mean;
incTax_profiles_all(12:end,:)=incTax_profiles(11:end,:);

asset_profiles_all(1:10,:)=asset_profiles(1:10,:);
asset_profiles_all(11,:)=AgeConditionalStats.assets.Mean;
asset_profiles_all(12:end,:)=asset_profiles(11:end,:);

income_profiles_all(1:10,:)=income_profiles(1:10,:);
income_profiles_all(11,:)=AgeConditionalStats.earnings.Mean+AgeConditionalStats.Pensions.Mean;
income_profiles_all(12:end,:)=income_profiles(11:end,:);

h_profiles_all(1:10,:)=h_profiles(1:10,:);
h_profiles_all(11,:)=AgeConditionalStats.human_capital.Mean;
h_profiles_all(12:end,:)=h_profiles(11:end,:);

l_profiles_all(1:10,:)=l_profiles(1:10,:);
l_profiles_all(11,:)=AgeConditionalStats.labor_supply.Mean;
l_profiles_all(12:end,:)=l_profiles(11:end,:);

timeWorking_profiles_all(1:10,:)=timeWorking_profiles(1:10,:);
timeWorking_profiles_all(11,:)=AgeConditionalStats.time_working.Mean;
timeWorking_profiles_all(12:end,:)=timeWorking_profiles(11:end,:);

timeStudying_profiles_all(1:10,:)=timeStudying_profiles(1:10,:);
timeStudying_profiles_all(11,:)=AgeConditionalStats.time_studying.Mean;
timeStudying_profiles_all(12:end,:)=timeStudying_profiles(11:end,:);

profiles_fig=figure();
subplot(3,3,1)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
legend('\lambda=0.5','\lambda=1','\lambda=1.5','Location','best')
xlabel('Age')
title('Consumption')

subplot(3,3,2)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
xlabel('Age')
title('Consumption tax')

subplot(3,3,3)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_all(6,:)-consTax_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_all(11,:)-consTax_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_all(16,:)-consTax_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
xlabel('Age')
title('Net consumption')

subplot(3,3,4)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),asset_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),asset_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),asset_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
xlabel('Age')
title('Assets')

subplot(3,3,5)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),income_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),income_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),income_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
xlabel('Age')
title('Earnings')

subplot(3,3,6)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),incTax_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),incTax_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),incTax_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
xlabel('Age')
title('Income tax')

subplot(3,3,7)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),income_profiles_all(6,:)-incTax_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),income_profiles_all(11,:)-incTax_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),income_profiles_all(16,:)-incTax_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
xlabel('Age')
title('After-tax-earnings')

subplot(3,3,8)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),asset_profiles_all(6,:)./(income_profiles_all(6,:)-incTax_profiles_all(6,:)),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),asset_profiles_all(11,:)./(income_profiles_all(11,:)-incTax_profiles_all(11,:)),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),asset_profiles_all(16,:)./(income_profiles_all(16,:)-incTax_profiles_all(16,:)),'LineWidth',1,'Color',yellow)
hold off
xlabel('Age')
xlim([23,64])
title('K/E ratio')

% subplot(3,4,9)
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),h_profiles_all(6,:),'LineWidth',1,'Color',blue)
% hold on
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),h_profiles_all(11,:),'LineWidth',1,'Color',red)
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),h_profiles_all(16,:),'LineWidth',1,'Color',yellow)
% hold off
% xlabel('Age')
% xlim([23,64])
% title('Human capital')
% 
% subplot(3,4,10)
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),l_profiles_all(6,:),'LineWidth',1,'Color',blue)
% hold on
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),l_profiles_all(11,:),'LineWidth',1,'Color',red)
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),l_profiles_all(16,:),'LineWidth',1,'Color',yellow)
% hold off
% xlabel('Age')
% xlim([23,64])
% title('Labor supply')

subplot(3,3,9)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_all(6,:),'LineWidth',1,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_all(11,:),'LineWidth',1,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_all(16,:),'LineWidth',1,'Color',yellow)
hold off
xlabel('Age')
xlim([23,64])
title('Time studying')

% figure()
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeWorking_profiles_all(6,:),'LineWidth',1,'Color',blue)
% hold on
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeWorking_profiles_all(11,:),'LineWidth',1,'Color',red)
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeWorking_profiles_all(16,:),'LineWidth',1,'Color',yellow)
% hold off
% xlabel('Age')
% xlim([23,64])
% title('Time working')

% subplot(3,4,12)
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_all(6,:),'LineWidth',1,'Color',blue)
% hold on
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_all(11,:),'LineWidth',1,'Color',red)
% plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_all(16,:),'LineWidth',1,'Color',yellow)
% hold off
% xlabel('Age')
% ylim([-0.05,0])
% title('Average utility')

% saveas(profiles_fig,'profiles_fig','epsc')


%% Age profiles of consumption, assets, after-tax-earnings, human capital, labor supply by types
% This section plots age profiles for every permanent type
% x-axis: age
% y-axis: aggregate value conditional on age and type

consumption_profiles_types=zeros(n_experiments,N_j,N_i);
consTax_profiles_types=zeros(n_experiments,N_j,N_i);
incTax_profiles_types=zeros(n_experiments,N_j,N_i);
asset_profiles_types=zeros(n_experiments,N_j,N_i);
income_profiles_types=zeros(n_experiments,N_j,N_i);
h_profiles_types=zeros(n_experiments,N_j,N_i);
l_profiles_types=zeros(n_experiments,N_j,N_i);
timeWorking_profiles_types=zeros(n_experiments,N_j,N_i);
timeStudying_profiles_types=zeros(n_experiments,N_j,N_i);

for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i);
    data_here=Output.(fieldName);

    for j = 1: N_i
        typej=['ptype00',num2str(j)];
        consumption_profiles_types(i,:,j)=data_here.AgeConditionalStats.Consumption.(typej).Mean;
        consTax_profiles_types(i,:,j)=data_here.AgeConditionalStats.consumptionTax.(typej).Mean;
        incTax_profiles_types(i,:,j)=data_here.AgeConditionalStats.incomeTax.(typej).Mean;
        asset_profiles_types(i,:,j)=data_here.AgeConditionalStats.K.(typej).Mean;
        income_profiles_types(i,:,j)=data_here.AgeConditionalStats.Earnings.(typej).Mean+data_here.AgeConditionalStats.Pensions.(typej).Mean;
        h_profiles_types(i,:,j)=data_here.AgeConditionalStats.H.(typej).Mean;
        l_profiles_types(i,:,j)=data_here.AgeConditionalStats.L.(typej).Mean;
        timeWorking_profiles_types(i,:,j)=data_here.AgeConditionalStats.timeWorking.(typej).Mean;
        timeStudying_profiles_types(i,:,j)=data_here.AgeConditionalStats.timeStudying.(typej).Mean;

    end
end

% Add benchmark
consumption_profiles_types_all=zeros(n_experiments+1,N_j,N_i);
consTax_profiles_types_all=zeros(n_experiments+1,N_j,N_i);
incTax_profiles_types_all=zeros(n_experiments+1,N_j,N_i);
asset_profiles_types_all=zeros(n_experiments+1,N_j,N_i);
income_profiles_types_all=zeros(n_experiments+1,N_j,N_i);
h_profiles_types_all=zeros(n_experiments+1,N_j,N_i);
l_profiles_types_all=zeros(n_experiments+1,N_j,N_i);
timeWorking_profiles_types_all=zeros(n_experiments+1,N_j,N_i);
timeStudying_profiles_types_all=zeros(n_experiments+1,N_j,N_i);

consumption_profiles_types_all(1:10,:,:)=consumption_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    consumption_profiles_types_all(11,:,j)=AgeConditionalStats.consumption.(typej).Mean;
end
consumption_profiles_types_all(12:end,:,:)=consumption_profiles_types(11:end,:,:);

consTax_profiles_types_all(1:10,:,:)=consTax_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    consTax_profiles_types_all(11,:,j)=AgeConditionalStats.consumptionTax.(typej).Mean;
end
consTax_profiles_types_all(12:end,:,:)=consTax_profiles_types(11:end,:,:);

incTax_profiles_types_all(1:10,:,:)=incTax_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    incTax_profiles_types_all(11,:,j)=AgeConditionalStats.incomeTax.(typej).Mean;
end
incTax_profiles_types_all(12:end,:,:)=incTax_profiles_types(11:end,:,:);

asset_profiles_types_all(1:10,:,:)=asset_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    asset_profiles_types_all(11,:,j)=AgeConditionalStats.assets.(typej).Mean;
end
asset_profiles_types_all(12:end,:,:)=asset_profiles_types(11:end,:,:);

income_profiles_types_all(1:10,:,:)=income_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    income_profiles_types_all(11,:,j)=AgeConditionalStats.earnings.(typej).Mean+AgeConditionalStats.Pensions.(typej).Mean;
end
income_profiles_types_all(12:end,:,:)=income_profiles_types(11:end,:,:);

h_profiles_types_all(1:10,:,:)=h_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    h_profiles_types_all(11,:,j)=AgeConditionalStats.human_capital.(typej).Mean;
end
h_profiles_types_all(12:end,:,:)=h_profiles_types(11:end,:,:);

l_profiles_types_all(1:10,:,:)=l_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    l_profiles_types_all(11,:,j)=AgeConditionalStats.labor_supply.(typej).Mean;
end
l_profiles_types_all(12:end,:,:)=l_profiles_types(11:end,:,:);

timeWorking_profiles_types_all(1:10,:,:)=timeWorking_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    timeWorking_profiles_types_all(11,:,j)=AgeConditionalStats.time_working.(typej).Mean;
end
timeWorking_profiles_types_all(12:end,:,:)=timeWorking_profiles_types(11:end,:,:);

timeStudying_profiles_types_all(1:10,:,:)=timeStudying_profiles_types(1:10,:,:);
for j=1:N_i
    typej=['ptype00',num2str(j)];
    timeStudying_profiles_types_all(11,:,j)=AgeConditionalStats.time_studying.(typej).Mean;
end
timeStudying_profiles_types_all(12:end,:,:)=timeStudying_profiles_types(11:end,:,:);


% Welfare conditional on age and type
welfare_profiles_types=zeros(n_experiments,N_j,N_i);

for i = 1:length(lambda_vec)
    fieldName = sprintf('loop%d', i);
    data_here=Output.(fieldName);

    V_here=data_here.V;
    StationaryDist_here=data_here.StationaryDist;
    for jj=1:N_i
        typejj=['ptype00',num2str(jj)];
        Udist_jj=V_here.(typejj).*StationaryDist_here.(typejj);
        % Udist_jj=Udist_jj(~isnan(Udist_jj)); % This will change the shape of Udist_jj (it works fine with 2D matrix, but Udist_jj is 3D)
        welfare_profiles_types(i,:,jj)=sum(sum(Udist_jj));
    end
end

% Add benchmark
welfare_profiles_types_all=zeros(n_experiments+1,N_j,N_i);

welfare_profiles_types_all(1:10,:,:)=welfare_profiles_types(1:10,:,:);
welfare_profiles_types_all(12:end,:,:)=welfare_profiles_types(11:end,:,:);

for jj=1:N_i
    typejj=['ptype00',num2str(jj)];
    Udist_jj_BM=V.(typejj).*StationaryDist.(typejj);
    % Udist_jj=Udist_jj(~isnan(Udist_jj)); % This will change the shape of Udist_jj (it works fine with 2D matrix, but Udist_jj is 3D)
    welfare_profiles_types_all(11,:,jj)=sum(sum(Udist_jj_BM));
end


profiles_types_fig=figure();

subplot(2,2,1)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_types_all(11,:,1),'LineWidth',1.5,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_types_all(11,:,4),'LineWidth',1.5,'Color',yellow)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_types_all(11,:,8),'LineWidth',1.5,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_types_all(16,:,1),'LineWidth',1.5,'Color',blue,'LineStyle','-.')
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_types_all(16,:,4),'LineWidth',1.5,'Color',yellow,'LineStyle','-.')
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),timeStudying_profiles_types_all(16,:,8),'LineWidth',1.5,'Color',red,'LineStyle','-.')
hold off
xlabel('Age')
xlim([23,64])
legend('BM - lowest ability','BM - median ability','BM - highest ability','','','','Location','best','Box','off')
title('Time studying')

subplot(2,2,2)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_types_all(11,:,1)-consTax_profiles_types_all(11,:,1),'LineWidth',1.5,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_types_all(11,:,4)-consTax_profiles_types_all(11,:,4),'LineWidth',1.5,'Color',yellow)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_types_all(11,:,8)-consTax_profiles_types_all(11,:,8),'LineWidth',1.5,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_types_all(16,:,1)-consTax_profiles_types_all(16,:,1),'LineWidth',1.5,'Color',blue,'LineStyle','-.')
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_types_all(16,:,4)-consTax_profiles_types_all(16,:,4),'LineWidth',1.5,'Color',yellow,'LineStyle','-.')
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consumption_profiles_types_all(16,:,8)-consTax_profiles_types_all(16,:,8),'LineWidth',1.5,'Color',red,'LineStyle','-.')
hold off
xlabel('Age')
xlim([23,75])
title('Net consumption')

subplot(2,2,3)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_types_all(11,:,1)+incTax_profiles_types_all(11,:,1),'LineWidth',1.5,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_types_all(11,:,4)+incTax_profiles_types_all(11,:,4),'LineWidth',1.5,'Color',yellow)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_types_all(11,:,8)+incTax_profiles_types_all(11,:,8),'LineWidth',1.5,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_types_all(16,:,1)+incTax_profiles_types_all(16,:,1),'LineWidth',1.5,'Color',blue,'LineStyle','-.')
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_types_all(16,:,4)+incTax_profiles_types_all(16,:,4),'LineWidth',1.5,'Color',yellow,'LineStyle','-.')
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),consTax_profiles_types_all(16,:,8)+incTax_profiles_types_all(16,:,8),'LineWidth',1.5,'Color',red,'LineStyle','-.')
hold off
xlabel('Age')
xlim([23,75])
title('Total taxes')

subplot(2,2,4)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_types_all(11,:,1),'LineWidth',1.5,'Color',blue)
hold on
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_types_all(11,:,4),'LineWidth',1.5,'Color',yellow)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_types_all(11,:,8),'LineWidth',1.5,'Color',red)
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_types_all(16,:,1),'LineWidth',1.5,'Color',blue,'LineStyle','-.')
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_types_all(16,:,4),'LineWidth',1.5,'Color',yellow,'LineStyle','-.')
plot(Params.agejshifter+1:1:(Params.agejshifter+N_j),welfare_profiles_types_all(16,:,8),'LineWidth',1.5,'Color',red,'LineStyle','-.')
hold off
xlabel('Age')
legend('','','','Optimal - lowest ability','Optimal - median ability','Optimal - highest ability','Location','best','Box','off')
xlim([23,75])
title('Welfare')

