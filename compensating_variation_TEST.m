function out=compensating_variation_TEST(Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,pi_z,DiscountFactorParamNames,AgeWeightsParamNames,PTypeDistParamNames,jequaloneDist,ReturnFn,heteroagentoptions,vfoptions,simoptions)

% To compute LumpSum so that when lambda changes (which leads to new eqm),
% welfare remains unchanged

% As income tax is changed through lambda, need to solve for eqm:
% (1) Capital market condition holds -> pin down r
% (2) Tax/Y = benchmarkTaxratio -> pin down tau
% (3) Welfare = benchmarkWelfare -> pin down LumpSum 
% (4) Pensions = Tax revenues -> pin down transfer (i.e. pension)

% c=1/(1+tau)*(earnings + (1+r)*a - Tax + pension +LumpSum - aprime*(1+g));
% Step 1: Solve for (r, tau, transfer) using 1,2, 4
% Step 2: given (r, tau, transfer), solve for LumpSum so that Welfare = benchmarkWelfare

%% Step 1: solve for (r, tau, transfer) given new lambda 
% This step is the same as optimal_tax.m, so just load GE values
GEPriceParamNames={'r','tau','transfer'}; 

% Aggregates: @(d,a1prime,a1,a2,...)
FnsToEvaluate.H     =@(s,aprime,a,h,agej,Jr) h*(agej<Jr);           % Aggregate human capital
FnsToEvaluate.L     =@(s,aprime,a,h,agej,Jr) h*(1-s)*(agej<Jr);     % Aggregate labor supply
FnsToEvaluate.K     =@(s,aprime,a,h) a;                             % Aggregate physical capital
FnsToEvaluate.Earnings          =@(s,aprime,a,h,w,agej,Jr) w*h*(agej<Jr)*(1-s);
FnsToEvaluate.Pensions          =@(s,aprime,a,h,transfer,agej,Jr) (agej>=Jr)*transfer;
FnsToEvaluate.Consumption       =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) optimal_tax_ConsFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);
FnsToEvaluate.incomeTax         =@(s,aprime,a,h,w,agej,Jr,transfer,lambda) lambda*statutory_tax_fn_2010(w*h*(agej<Jr)*(1-s)+(agej>=Jr)*transfer);
FnsToEvaluate.consumptionTax    =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) tau*optimal_tax_ConsFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);
FnsToEvaluate.Utility           =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) optimal_tax_ReturnFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);
FnsToEvaluate.LS     =@(s,aprime,a,h,LumpSum) LumpSum;           % Aggregate LumpSum

% GE conditions
GeneralEqmEqns.capitalmarket=@(r,K,L,alpha,delta_k) r-alpha*(K/L)^(alpha-1)+delta_k;
GeneralEqmEqns.taxratio=@(incomeTax,consumptionTax,K,L,alpha,benchmarkTaxratio) (incomeTax+consumptionTax)/((K^alpha)*(L^(1-alpha)))-benchmarkTaxratio;
GeneralEqmEqns.govbudget=@(Pensions,incomeTax,consumptionTax,pension_ratio) Pensions-pension_ratio*(incomeTax+consumptionTax);

% % Solve for the General Equilibrium 
heteroagentoptions.verbose=1;
% [p_eqm,~,~]=HeteroAgentStationaryEqm_Case1_FHorz_PType(n_d, n_a, n_z, N_j, N_i, [], pi_z, d_grid, a_grid, z_grid,jequaloneDist, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% 
% Params.r=p_eqm.r;
% Params.tau=p_eqm.tau;
% Params.transfer=p_eqm.transfer;
% Params.w=(1-Params.alpha)*((Params.r+Params.delta_k)/Params.alpha)^(Params.alpha/(Params.alpha-1));


%% Step 2: given (r,tau,transfer), solve for LumpSum
absOmega0minusOmega1=@(guess_LumpSum) welfare_objectivefn_TEST(guess_LumpSum,Params,jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,n_d,n_a,n_z,N_j,N_i,pi_z,d_grid,a_grid,z_grid,ReturnFn,FnsToEvaluate,GeneralEqmEqns,GEPriceParamNames,DiscountFactorParamNames,heteroagentoptions,simoptions,vfoptions);

minoptions = optimset('TolX',10^(-16),'TolFun',10^(-16));
[GE,welfarecondn]=fminsearch(absOmega0minusOmega1,Params.LumpSum_guess,minoptions);

Params.LumpSum=GE;

%% Calculate aggregates

[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,[],Params,simoptions);

AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,[],simoptions);

AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Params, n_d, n_a, n_z,N_j,N_i, d_grid, a_grid, z_grid,simoptions);

K=AggVars.K.Mean;
H=AggVars.H.Mean;
L=AggVars.L.Mean;
Earnings=AggVars.Earnings.Mean;
C=AggVars.Consumption.Mean;
Y=(K^Params.alpha)*(L^(1-Params.alpha));  % GDP
Pensions=AggVars.Pensions.Mean;
incomeTax=AggVars.incomeTax.Mean;
consumptionTax=AggVars.consumptionTax.Mean;
Utility=AggVars.Utility.Mean;
LumpSum_agg=AggVars.LS.Mean;
TaxtoGDP=(AggVars.incomeTax.Mean+AggVars.consumptionTax.Mean)/Y; 

Udist_ptype=zeros(1,N_i);
for jj=1:N_i
    if jj<10
        typejj=['ptype00',num2str(jj)];
    elseif jj<20
        typejj=['ptype0',num2str(jj)];
    end
    % Utility
    Udist_jj=V.(typejj).*StationaryDist.(typejj);
    Udist_jj(isnan(Udist_jj))=0; % replace nan by 0, otherwise the next line might result in NAN
    Udist_ptype(jj)=sum(sum(sum(Udist_jj)));
end

Udist_avg=Udist_ptype*StationaryDist.ptweights; 

out.Params=Params;
out.V=gather(V);
out.Policy=gather(Policy);
out.StationaryDist=gather(StationaryDist);
out.welfarecondn=welfarecondn;

out.LumpSum_agg=LumpSum_agg;
out.LumpSumRatio=LumpSum_agg/Y*100;
out.tau=Params.tau;
out.transfer=Params.transfer;
out.r=Params.r;
out.w=Params.w;
out.K=gather(K);
out.H=gather(H);
out.L=gather(L);
out.Earnings=gather(Earnings);
out.C=gather(C);
out.Y=gather(Y);
out.Pensions=gather(Pensions);
out.incomeTax=gather(incomeTax);
out.consumptionTax=gather(consumptionTax);
out.Utility=gather(Utility);
out.TaxtoGDP=gather(TaxtoGDP);
out.Udist_avg=gather(Udist_avg);

out.AgeConditionalStats=AgeConditionalStats;
out.AggVars=AggVars;
out.Udist_ptype=Udist_ptype;

end