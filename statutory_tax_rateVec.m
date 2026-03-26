% Statutory tax rate paid by tax payers according to statutory taxes in NZ from 1/10/2010 to 31/3/2021
% NOTE: this function only helps make a graph on statutory tax rates
% The thesholds below start applied in 2010, but the model is in $2006
% In other tax functions, the thresholds need to be devided by CPI

function tax_rate = statutory_tax_rateVec(income)

tax_rate = zeros(size(income)); % Just to keep matlab happy (one of the if statements will overwrite)

thresholds1 = 14;   
thresholds2 = 48;   
thresholds3 = 70;   

for i = 1:length(income)
    if income(i) <= 0
        tax_rate(i) = 0;
    elseif income(i) > 0 && income(i) <= thresholds1
        tax_rate(i) = 0.105;
    elseif income(i) > thresholds1 && income(i) <= thresholds2
        tax_rate(i) = 0.175;
    elseif income(i) > thresholds2 && income(i) <= thresholds3
        tax_rate(i) = 0.300;
    elseif income(i) > thresholds3
        tax_rate(i) = 0.330;
    end
end

end