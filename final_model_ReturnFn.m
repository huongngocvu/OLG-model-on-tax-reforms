function F=final_model_ReturnFn(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant)
% w is the wage to human capital: earnings = w*h*(1-s)

F=-Inf;

if agej<Jr
    earnings=constant*w*h*(totalhours-s);
    Tax=statutory_tax_fn_2010(earnings);    % Income tax
    pension=0;
else
    earnings=0;
    pension=transfer;
    Tax=statutory_tax_fn_2010(pension);    % Taxing pension
end


c=1/(1+tau)*(earnings + (1+r)*a - Tax + pension - aprime*(1+g));


if c>0
    F=(c^(1-sigma))/(1-sigma);
end


% Indifferent about l=1-s when retired
% Not a problem, but gives crazy policy for l during retirement
% Clean it up by making l=0 (equivalently s=1) when retired
if agej>=Jr
    if s<totalhours
        F=-Inf;
    end
end

% if agej<Jr && rem(agej,5)==0
%     if s<1
%         F=-Inf;
%     end
% end


end