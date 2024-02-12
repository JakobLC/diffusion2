

import sys, os
sys.path.append(os.path.abspath('./source/'))
from source.utils import SmartParser,bracket_glob_fix,get_model_name_from_written_args,list_wrap_type,load_old_args
from source.training import DiffusionModelTrainer
import warnings
from pathlib import Path
warnings.filterwarnings('ignore', category=DeprecationWarning)

def get_ckpt_name(s,saves_folder="./saves/",return_multiple_matches=False):
    assert len(s)>0, "name_match_str must be specified"
    if s.find(".pt")>=0 and s.find("/")>=0: #*abc/def.pt or abc/def.pt
        pass 
    elif s.find("/")<0: #*abc or abc
        assert s.find(".pt")<0, "invalid format. If you want to match a .pt file, you must specify the directory. You can always use */your_model_string.pt"
        s += "/ckpt_*.pt"
    else: #*abc/def or abc/def
        s += ".pt" 
    if s.find("*") >= 0:
        matching_paths = list(Path(saves_folder).glob(bracket_glob_fix(s)))
        if len(matching_paths) == 0:
            raise ValueError("No models match the expression: "+s+", consider using a starred expression")
        elif len(matching_paths)==1:
            s = str(matching_paths[0])[len("saves/"):]
        else:
            if return_multiple_matches:
                s = sorted([str(x)[len("saves/"):] for x in matching_paths])
            else:
                raise ValueError("Multiple models match the expression. Be more specific than name_match_str="+s+"\n matches listed below: \n"+str([str(x) for x in matching_paths]))
    else:
        assert (Path(saves_folder)/s).exists(), "model path does not exist. Use starred expressions to search s="+s
        s = str(s)
    return s

def main(**modified_args):
    sample_opts = SmartParser("sample_opts",
                              key_to_type={"name_match_str": lambda x: get_ckpt_name(x,return_multiple_matches=True)}
                              ).get_args(modified_args=modified_args)
    if isinstance(sample_opts,list):
        #sample_opts, modified_args_list = sample_opts
        modified_args_list = sample_opts
        for modified_args in modified_args_list:
            main(**modified_args)
        return
    
    ckpt_name = get_ckpt_name(sample_opts.name_match_str,return_multiple_matches=False)
    print("\nckpt_name:",ckpt_name)
    args_path = str((Path("./saves/")/ckpt_name).parent / "args.json")
    args = load_old_args(args_path)
    if sample_opts.seed>=0:
        args.seed = sample_opts.seed
    args.mode = "gen"
    args.ckpt_name = ckpt_name
    trainer = DiffusionModelTrainer(args)
    gen_tuples = [(sample_opts.gen_setup,sample_opts.__dict__)]
    trainer.generate_samples(gen_tuples)

if __name__ == "__main__":
    main()