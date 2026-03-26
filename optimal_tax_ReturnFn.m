function F=optimal_tax_ReturnFn(s,aprime,a,h,r,sigma,agej,Jr,tau,transfer,alpha,delta_k,g,lambda,LumpSum)
% w is the wage to human capital: earnings = w*h*(1-s)
% LumpSum is not used in baseline (=0 in baseline), is needed for welfare analyis

w=(1-alpha)*((r+delta_k)/alpha)^(alpha/(alpha-1)); % DOUBLE CHECK THIS. SHOULD IT BE (1-alpha)/alpha times the rest
% I checked but the formula of w is unchanged (DOUBLE CHECK!!!)


F=-Inf;

if agej<Jr
    earnings=w*h*(1-s);
    Tax=lambda*statutory_tax_fn_2010(earnings);    % Income tax
    pension=0;
else
    earnings=0;
    pension=transfer;
    Tax=lambda*statutory_tax_fn_2010(pension);    % Taxing pension
end


c=1/(1+tau)*(earnings + (1+r)*a - Tax + pension +LumpSum - aprime*(1+g));

if c>0
    F=(c^(1-sigma))/(1-sigma);
end


% Indifferent about l=1-s when retired
% Not a problem, but gives crazy policy for l during retirement
% Clean it up by making l=0 (equivalently s=1) when retired
if agej>=Jr
    if s<1
        F=-Inf;
    end
end


end