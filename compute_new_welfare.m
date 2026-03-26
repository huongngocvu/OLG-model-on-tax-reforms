function out=compute_new_welfare(Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,pi_z,DiscountFactorParamNames,AgeWeightsParamNames,PTypeDistParamNames,jequaloneDist,ReturnFn,heteroagentoptions,vfoptions,simoptions)

% Given a new value of lambda and an initial guess for consumption tax
% rate, solve for new eqm, then compute utility and welfare

%% First, solve for r, tau, transfer given new lambda
GEPriceParamNames={'r','tau','transfer'}; 

% Aggregates: @(d,a1prime,a1,a2,...)
FnsToEvaluate.H     =@(s,aprime,a,h,agej,Jr) h*(agej<Jr);           % Aggregate human capital
FnsToEvaluate.L     =@(s,aprime,a,h,agej,Jr) h*(1-s)*(agej<Jr);     % Aggregate labor supply
FnsToEvaluate.K     =@(s,aprime,a,h) a;                             % Aggregate physical capital
FnsToEvaluate.timeStudying      =@(s,aprime,a,h) s;                            
FnsToEvaluate.timeWorking       =@(s,aprime,a,h) 1-s;                            
FnsToEvaluate.Earnings          =@(s,aprime,a,h,w,agej,Jr) w*h*(agej<Jr)*(1-s);
FnsToEvaluate.Pensions          =@(s,aprime,a,h,transfer,agej,Jr) (agej>=Jr)*transfer;
FnsToEvaluate.Consumption       =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) optimal_tax_ConsFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);
FnsToEvaluate.incomeTax         =@(s,aprime,a,h,w,agej,Jr,transfer,lambda) lambda*statutory_tax_fn_2010(w*h*(agej<Jr)*(1-s)+(agej>=Jr)*transfer);
FnsToEvaluate.consumptionTax    =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) tau*optimal_tax_ConsFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);
FnsToEvaluate.Utility           =@(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum) optimal_tax_ReturnFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum);

% GE conditions
GeneralEqmEqns.capitalmarket=@(r,K,L,alpha,delta_k) r-alpha*(K/L)^(alpha-1)+delta_k;
GeneralEqmEqns.taxratio=@(incomeTax,consumptionTax,K,L,alpha,benchmarkTaxratio) (incomeTax+consumptionTax)/((K^alpha)*(L^(1-alpha)))-benchmarkTaxratio;
GeneralEqmEqns.govbudget=@(Pensions,incomeTax,consumptionTax,pension_ratio) Pensions-pension_ratio*(incomeTax+consumptionTax);

% Solve for the General Equilibrium 
heteroagentoptions.verbose=1;
[p_eqm,~,GEcondvalues]=HeteroAgentStationaryEqm_Case1_FHorz_PType(n_d, n_a, n_z, N_j, N_i, [], pi_z, d_grid, a_grid, z_grid,jequaloneDist, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
Params.r=p_eqm.r;
Params.tau=p_eqm.tau;
Params.transfer=p_eqm.transfer;
Params.w=(1-Params.alpha)*((Params.r+Params.delta_k)/Params.alpha)^(Params.alpha/(Params.alpha-1));

%% Aggregates
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
TaxtoGDP=(AggVars.incomeTax.Mean+AggVars.consumptionTax.Mean)/Y; 

% ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,simoptions);

Udist_ptype=zeros(1,N_i);
% Welfare_ptype=zeros(1,N_i);
for jj=1:N_i
    if jj<10
        typejj=['ptype00',num2str(jj)];
    elseif jj<20
        typejj=['ptype0',num2str(jj)];
    end
    % Utility
    Udist_jj=V.(typejj).*StationaryDist.(typejj);
    % Udist_jj=Udist_jj(~isnan(Udist_jj));
    Udist_ptype(jj)=sum(sum(sum(Udist_jj)));
end

Udist_avg=Udist_ptype*StationaryDist.ptweights; 

out.Params=Params;
out.V=gather(V);
out.Policy=gather(Policy);
out.StationaryDist=gather(StationaryDist);
out.GEcondvalues=GEcondvalues;

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
% out.ValuesOnGrid=ValuesOnGrid;
out.AggVars=AggVars;
out.Udist_ptype=Udist_ptype;

end