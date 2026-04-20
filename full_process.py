import os
import argparse
import yaml

from conf_fctn import odict,dict2namespace


from models_sampling import projection_process

# from evaluation.evaluation_from_parameters import full_evaluation
# from evaluation.scores_comparison import scores_comparison

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Synthetic Population prediction')
    
    parser.add_argument('--n_variables', type=str, required=False, default=None) # available are
    parser.add_argument('--size_data', type=float, required=False, default=1) # avaialable are {0.03, 1}
    parser.add_argument('--encoding', type=str, required=False, default="tanh") # available are {tanh,log,None}
    
    parser.add_argument('--n_years', type=int, required=False, default=5) 
    parser.add_argument('--time_horizon', type=int, required=False, default=3)
    
    parser.add_argument('--year_prediction', type=int, required=False, default=2021)
    
    parser.add_argument('--alpha_min', type=float, required=False, default=0)
    parser.add_argument('--alpha_max', type=float, required=False, default=0.99)
    parser.add_argument('--n_alpha', type=int, required=False, default=20)
    
    parser.add_argument('--n_min', type=int, required=False, default=0)
    parser.add_argument('--n_max', type=int, required=False, default=1)

    parser.add_argument("--IPF_pre", action='store_true')
    parser.add_argument("--BN", action='store_true')
    parser.add_argument("--IPF_post", action='store_true')
    

    args = odict(vars(parser.parse_args()))
    
    with open(f"conf/conf_variable/{args.n_variables}.yml", "r") as f:
        config_ = yaml.safe_load(f)
    config = dict2namespace(config_)

    if(args.size_data == int(args.size_data)):
        str_float = "_".join(str(args.size_data).split(".0")[0].split("."))
    else:
        str_float = "_".join(str(args.size_data).split("."))
    
    with open(f"conf/conf_size/{str_float}%.yml", "r") as f:
        config_ = yaml.safe_load(f)
    config_2 = dict2namespace(config_)

    for key,val in config_2._get_kwargs():
        setattr(config, key, val)
        
    for key,val in dict2namespace(args)._get_kwargs():
        setattr(config, key, val)
        
    print(config)

    setattr(config,"folder_save",config.folder_save_start+config.folder_save_end)
    setattr(config, "size_data_str", str_float)

    if((not args.BN) and (args.IPF_pre)):
        args.IPF_pre = False
        
    print("\n\n\n***************** Model ********************")
    models_name = []
    
    if args.IPF_pre:
        models_name.append("IPF prefitting (new marginals on training data)")
    
    if args.BN:
        models_name.append("BN (projection of CPTs)")

    if args.IPF_post:
        models_name.append("IPF postfitting (new marginals on generated data)")

    print("  +  ".join(models_name))
    
    print("********************************************\n\n\n")

    projection_process(config)
    



    