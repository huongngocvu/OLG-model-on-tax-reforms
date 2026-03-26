function EqmCondns=welfare_objectivefn(GEandLumpSum,Params,jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,n_d,n_a,n_z,N_j,N_i,pi_z,d_grid,a_grid,z_grid,ReturnFn,FnsToEvaluate,GeneralEqmEqns,GEPriceParamNames,DiscountFactorParamNames,heteroagentoptions,simoptions,vfoptions)
% Welfare with compensation LumpSum in each period of life and new income tax
% GEandLumpSum is a vector of (r,tau,transfer,LumpSum) and is unknown

Params.r=GEandLumpSum(1);
Params.tau=GEandLumpSum(2);
Params.transfer=GEandLumpSum(3);
Params.LumpSum=GEandLumpSum(4);

% GEPriceParamNames={'r','tau','transfer','LumpSum'}; 
GEPriceParamNames={'r','tau','transfer'}; 

Params.w=(1-Params.alpha)*((Params.r+Params.delta_k)/Params.alpha)^(Params.alpha/(Params.alpha-1));

[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,[],Params,simoptions);

% Vector of GE conditionns
heteroagentoptions.maxiter=0;           % use the prices currently in Params
heteroagentoptions.outputGEstruct=2;    % get GE conditions in a vector (default=1 is structure, =2 is vector)
[~,~,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_FHorz_PType(n_d, n_a, n_z, N_j, N_i, [], pi_z, d_grid, a_grid, z_grid,jequaloneDist, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);

GeneralEqmConditionsVec=real(GeneralEqmConditions);

% Calculate absOmega0minusOmega1
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

Omega0=Params.benchmarkWelfare;
Omega1=Udist_ptype*StationaryDist.ptweights; 
absOmega0minusOmega1=abs(Omega0-Omega1);

EqmCondns=[gather(abs(GeneralEqmConditionsVec)),gather(absOmega0minusOmega1)];

% EqmCondns=sum([1,1,1,1].*(EqmCondns)); % Use the same weights
EqmCondns=sum([1,1,1,10].*(EqmCondns)); % Use different weights

fprintf('Param value of LumpSum: %4.6f \n', Params.LumpSum)
fprintf('Welfare condn: abs(Omega0 - Omega1): %4.6f \n', absOmega0minusOmega1)


end