-----------------------------------------------------------------------------------------------------------------------------------------
Code to solve the baseline model and policy experiments in the paper "Macroeconomic effects of income and consumption tax reforms in NZ"
-----------------------------------------------------------------------------------------------------------------------------------------

Part 1: The baseline model
	The main code is final_model.m which solves the baseline model in the paper.

	Required functions: 	final_model_ReturnFn.m
				final_model_ConsFn.m
				statutory_tax_fn_2010.m

	Required data inputs: 	calib_13.mat (storing all calibrated parameters)
				data_mean.xlsx (mean earnings conditional on age)
				data_gini.xlsx (Gini coefficients conditional on age)
				data_skew.xlsx (skewness as the ratio of mean to median earnings conditional on age) 

Part 2: Optimal tax experiments
	The main code is optimal_tax_MAIN.m which solves the model using different income tax rates (via the value of lambda)

	Required functions: 	optimal_tax_ReturnFn.m
				optimal_tax_ConsFn.m
				statutory_tax_fn_2010.m
				statutory_tax_rateVec.m
				compute_new_welfare.m

	Required data inputs: 	calib_13.mat (storing all calibrated parameters)

Part 3: Compensating variation
	The main code is compensating_variation_MAIN.m which computes the compensating variation

	Required functions: 	compensating_variation_TEST.m
				welfare_objectivefn_TEST.m