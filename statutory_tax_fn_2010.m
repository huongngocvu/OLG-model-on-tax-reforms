% Tax amounts paid by tax payers in NZ from 1/10/2010 to 31/3/2021

function out = statutory_tax_fn_2010(income)

out = 0; % Just to keep matlab happy (one of the if statements will overwrite)

thresholds1 = 14/1.13;   % in2006 prices
thresholds2 = 48/1.13;   % in2006 prices
thresholds3 = 70/1.13;   % in2006 prices

if income <= 0
    out = 0;
elseif income > 0 && income <= thresholds1
    out = 0.105*income;
elseif income > thresholds1 && income <= thresholds2
    out = 0.105*thresholds1 + 0.175*(income-thresholds1);
elseif income > thresholds2 && income <= thresholds3
    out = 0.105*thresholds1 + 0.175*(thresholds2-thresholds1) + 0.300*(income-thresholds2);
elseif income > thresholds3
    out = 0.105*thresholds1 + 0.175*(thresholds2-thresholds1) + 0.300*(thresholds3-thresholds2) + 0.330*(income-thresholds3);
end

end