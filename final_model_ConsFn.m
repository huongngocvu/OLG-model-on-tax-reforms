function c=final_model_ConsFn(s,aprime,a,h,w,r,sigma,agej,Jr,tau,transfer,g,totalhours,constant)
% w is the wage to human capital: earnings = w*h*(1-s)


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



end