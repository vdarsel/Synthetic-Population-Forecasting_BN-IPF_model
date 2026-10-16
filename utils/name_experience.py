alpha_min_default = 0
alpha_max_default = 0.99
n_alpha_default = 20

n_min_default = 0
n_max_default = 1

def get_term(args):
    term="_Projection"
    if args.TPIPF_pre:
        term+="_TPIPF"
    if args.TPBN:
        term+="_TPBN_Hill_Climb"
    if args.static_BN:
        term+="_Static_BN_Hill_Climb"
    if args.TPIPF_post:
        term+="_TPIPF"
    if args.Resampling_marginal:
        term+="_Resampling_marginal"
    if args.Resampling_joint:
        term+="_Resampling_joint_distribution"
    
    
    if ((args.n_min!=n_min_default)|(args.n_max!=n_max_default)):
        term+=f"_n_from{args.n_min}_to_{args.n_max}"
    
    if ((args.alpha_min!=alpha_min_default)|(args.alpha_max!=alpha_max_default)|(args.n_alpha!=n_alpha_default)):
        term+=f"_alphas_{args.n_alpha}_{args.alpha_min}_to_{args.alpha_max}"    
        
        
    year_prediction = args.year_prediction
    
    year_validation = year_prediction-args.time_horizon
    first_year_training_validation  = year_validation-args.n_years-args.time_horizon+1
    
    
    term+=f"_{args.encoding}_from_{first_year_training_validation}_Validation_{year_validation}"
    
    return term