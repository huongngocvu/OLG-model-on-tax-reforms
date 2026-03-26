function EqmCondns=welfare_objectivefn_TEST(guess_LumpSum,Params,jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,n_d,n_a,n_z,N_j,N_i,pi_z,d_grid,a_grid,z_grid,ReturnFn,FnsToEvaluate,GeneralEqmEqns,GEPriceParamNames,DiscountFactorParamNames,heteroagentoptions,simoptions,vfoptions)
% Welfare with compensation LumpSum in each period of life and new income tax
% guess_LumpSum is a vector of LumpSum and is unknown

Params.LumpSum=guess_LumpSum;

[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,[],Params,simoptions);

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
    Udist_jj(isnan(Udist_jj))=0; % replace nan by 0, otherwise the next line might result in NAN
    Udist_ptype(jj)=sum(sum(sum(Udist_jj)));
end

Omega0=Params.benchmarkWelfare;
Omega1=Udist_ptype*StationaryDist.ptweights; 
absOmega0minusOmega1=abs(Omega0-Omega1);

EqmCondns=gather(absOmega0minusOmega1);

end