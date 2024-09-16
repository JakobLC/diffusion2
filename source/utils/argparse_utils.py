
import argparse
import json
import sys
from pathlib import Path
from functools import partial
from collections import OrderedDict
from source.utils.mixed_utils import load_json_to_dict_list, save_dict_list_to_json, longest_common_substring, bracket_glob_fix
import copy
from shutil import rmtree

special_argkeys = ["deprecated","dynamic","version_backwards_compatability","renamed"]
dont_load_argskeys = [k for k in special_argkeys if k!="dynamic"]

def get_ckpt_name(s,saves_folder="./saves/",return_multiple_matches=False):
    s_orig = copy.copy(s)
    if len(s)==0:
        return s
    assert not s.find("./")>=0, "name_match_str is already relative to saves_folder. Do not use ./ in name_match_str."
    # converts to format:  ver*/*/*.pt
    num_sep = s.count("/")
    if num_sep==0:
        s = "ver-*/"+s+"/ckpt_*.pt"
    elif num_sep==1:
        if s.endswith(".pt"):
            s = "ver-*/"+s
        else:
            s = s+"/ckpt_*.pt"        
    elif num_sep==2:
        if not s.endswith(".pt"):
            s = s+".pt"
    if s.find("*") >= 0:
        matching_paths = list(Path(saves_folder).glob(bracket_glob_fix(s)))
        if len(matching_paths) == 0:
            raise ValueError(f"No models match the expression: {s_orig}, consider using a starred expression. The string was modified to: {s}.")
        elif len(matching_paths)==1:
            s = str(matching_paths[0])
        else:
            if return_multiple_matches:
                s = sorted([str(x) for x in matching_paths])
            else:
                raise ValueError("Multiple models match the expression. Be more specific than name_match_str="+s+"\n matches listed below: \n"+str([str(x) for x in matching_paths]))
    else:
        assert Path(s).exists(), "model path does not exist. Use starred expressions to search s="+s
        s = str(s)
    return s

def list_wrap_type(t):
    def list_wrap(x):
        if isinstance(x,str):
            if x.find(";")>=0:
                return [t(y) for y in x.split(";")]
            else:
                return t(x)
        else:
            return t(x)
    return list_wrap

def load_defaults(idx=0,
                  ordered_dict=False,
                  filename="jsons/args_default.json",
                  return_special_argkey=None,
                  version=None):
    if version is not None:
        assert idx==0, f"version={version} not supported with idx={idx}."
    default_path = Path(__file__).parent.parent.parent/filename
    if ordered_dict:
        args_dicts = json.loads(default_path.read_text(), object_pairs_hook=OrderedDict)    
    else:
        args_dicts = json.loads(default_path.read_text())
    if return_special_argkey is not None:
        assert return_special_argkey in special_argkeys, f"return_special_argkey={return_special_argkey} not supported."
        return args_dicts[return_special_argkey]
    if version is not None:
        assert isinstance(version,str), f"version={version}."
    args_dict = {}
    for k,v in args_dicts.items():
        if isinstance(v,dict):
            if k not in dont_load_argskeys:
                for k2,v2 in v.items():
                    args_dict[k2] = v2[idx]
        else:
            args_dict[k] = v[idx]
    if version is not None:
        for k,v in args_dicts["version_backwards_compatability"].items():
            for k2,v2 in v.items():
                if eval_version_condition(k2,version):
                    args_dict[k] = v2

    return args_dict

def compare_strs(str1,str2,operator):
    if operator=="==":
        out = str1==str2
    elif operator==">=":
        out = str1>=str2
    elif operator=="<=":
        out = str1<=str2
    elif operator=="<" or operator=="<<":
        out = str1<str2
    elif operator==">" or operator==">>":
        out = str1>str2
    else:
        raise ValueError(f"operator={operator} not supported.")
    return out

def eval_version_condition(version_condition=">0.1.2",x="1.0.0"):
    """function which evaluates a version statement, i.e:
    [implicit x]>0.1.2 
    0.1.2<x
    0.1.2<=[implicit x]
    0.0.0<=x<=1.0.0

    5 possible cases:
    #     ver1[op]x[op]ver2, 
    #     ver1[op]x,
    #     x[op]ver1,
    #     [op]ver1,
    #     ver1[op]
    """
    assert isinstance(version_condition,str), f"type(version_condition)={type(version_condition)}."
    #first replace > with >> but only if it is not followed by =, and < with << but only if it is not followed by =
    version_condition = version_condition.replace("=>",">=").replace("=<","<=") #switch order for affected soft inequalities
    version_condition = version_condition.replace(">",">>").replace("<","<<")
    version_condition = version_condition.replace(">>>>",">>").replace("<<<<","<<") #replace if it was already doubled
    version_condition = version_condition.replace(">>=",">=").replace("<<=","<=") #replace for affected soft inequalities
    valid_operators = ["==",">=","<=","<<",">>"]
    for op in valid_operators:
        if version_condition.endswith(op):
            version_condition = version_condition + "x"
        if version_condition.startswith(op):
            version_condition = "x" + version_condition
    #turn into a list of [ver,op,ver,op,...,ver]
    #then evaluate each pair of ver1,op,ver2
    version_condition = version_condition.replace("x",x)
    version_condition_list = []
    for _ in range(len(version_condition)): #simple upper bound on number of iterations
        minimum_idx = len(version_condition)
        for op in valid_operators:
            idx = version_condition.find(op)
            if idx>=0:
                minimum_idx = min(minimum_idx,idx)
        if minimum_idx==len(version_condition):
            version_condition_list.append(version_condition)
            break
        version_condition_list.append(version_condition[:minimum_idx])
        version_condition_list.append(version_condition[minimum_idx:minimum_idx+2])
        version_condition = version_condition[minimum_idx+2:]
    version_condition = version_condition_list
    assert len(version_condition)%2==1, f"expected len(version_condition) to be odd. version_condition={version_condition}."
    num_expr = len(version_condition)//2
    num_dots = x.count(".")
    num_digits_x = [len(y) for y in x.split(".")]
    max_num_digits = num_digits_x
    #assert format is comparable
    for i in range(num_expr+1):
        assert all([y.isdigit() for y in version_condition[2*i].split(".")]), f"version_condition[2*i]={version_condition[2*i]} is not a valid version string."
        num_dots_i = version_condition[2*i].count(".")
        num_digits_i = [len(y) for y in version_condition[2*i].split(".")]
        assert num_dots_i==num_dots, f"Mismatch in number of dots in compared versions. x={x}, version_condition[2*i]={version_condition[2*i]}."
        max_num_digits = [max(max_num_digits[j],num_digits_i[j]) for j in range(num_dots+1)]
    comparable_version_condition = version_condition
    for i in range(num_expr+1):
        y = version_condition[2*i].split(".")
        y = [y[j].zfill(max_num_digits[j]) for j in range(num_dots+1)]
        y = "".join(y)
        comparable_version_condition[2*i] = int(y)
    comparison = []
    for i in range(num_expr):
        ver1 = comparable_version_condition[2*i]
        ver2 = comparable_version_condition[2*i+2]
        op = version_condition[2*i+1]
        comparison.append(compare_strs(ver1,ver2,op))
    return all(comparison)

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, (int,float)):
        return bool(v)
    elif isinstance(v, str):
        if v.lower() in ["yes", "true", "t", "y", "1"]:
            return True
        elif v.lower() in ["no", "false", "f", "n", "0"]:
            return False
        else:
            raise argparse.ArgumentTypeError("Cannot convert string: {} to bool".format(v))
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def str_with_semicolon_version(v):
    """
    Converts strings of the type 
    from       |   to
    -----------------------
    'm[v1;v2]'         -> 'm[v1];m[v2]'
    'm1[v1];m2[v1;v2]' -> 'm1[v1];m2[v1];m2[v2]'
    'm[v1;v2]+p1'      -> 'm[v1]+p1;m[v2]+p1'
    """
    assert v.count("[")==v.count("]"), f"v={v} has mismatched brackets."
    if v.find(";")<0:
        return v
    #replace all ";" inside brackets with ":"
    v2 = ""
    open_brackets = 0
    for s_idx in range(len(v)):
        if v[s_idx]=="[":
            open_brackets += 1
        elif v[s_idx]=="]":
            open_brackets -= 1
        if v[s_idx]==";":
            if open_brackets>0:
                v2 += ":"
            else:
                v2 += ";"
        else:
            v2 += v[s_idx]
    #split by ";"
    v_out = ""
    for s in v2.split(";"):
        s1 = s[:s.find("[")]
        bracket_string = s[s.find("["):s.find("]")+1]
        s3 = s[s.find("]")+1:]
        for s2 in bracket_string[1:-1].split(":"):
            v_out += s1+"["+s2+"]"+s3+";"
    v_out = v_out[:-1]
    return v_out.split(";")

class TieredParser():
    def __init__(self,name="args",
                 tiers_dict={"modified_args": 0,
                                  "commandline": 1,
                                  "name_plus": 2,
                                  "name_versions": 3, 
                                  "name_root": 4, 
                                  "defaults_version": 5, 
                                  "defaults_current": 6},
                    deprecated_argkey="deprecated",
                    backwards_comp_argkey="version_backwards_compatability",
                    dynamic_argkey="dynamic",
                    rename_argkey="renamed",
                    key_to_type={"origin": dict,
                                 "name_match_str": lambda x: get_ckpt_name(x,return_multiple_matches=True),
                                 "model_name": str_with_semicolon_version,
                                 "gen_setup": str_with_semicolon_version}):
        self.tiers_dict = tiers_dict
        assert name in ["args","sample_opts"], f"name={name} not supported."
        self.name_key = {"args": "model_name","sample_opts": "gen_setup"}[name]
        self.id_key = {"args": "model_id","sample_opts": "gen_id"}[name]
        self.version_key = {"args": "model_version","sample_opts": "gen_version"}[name]
        self.filename_def   = "jsons/"+name+"_default.json"
        self.filename_model = "jsons/"+name+"_configs.json"
        self.filename_ids   = "jsons/"+name+"_ids.json"

        self.deprecated_argkey = deprecated_argkey
        self.backwards_comp_argkey = backwards_comp_argkey
        self.dynamic_argkey = dynamic_argkey
        self.rename_argkey = rename_argkey
        self.defaults_func = partial(load_defaults,filename=self.filename_def)
        self.descriptions = self.defaults_func(idx=1)

        self.deprecated_args = self.defaults_func(return_special_argkey="deprecated")
        self.backwards_comp_args = self.defaults_func(return_special_argkey="version_backwards_compatability")
        self.dynamic_args = self.defaults_func(return_special_argkey="dynamic")
        self.renamed_args = self.defaults_func(return_special_argkey="renamed")

        self.parser = argparse.ArgumentParser()
        self.type_dict = {}
        for k, v in self.defaults_func().items():
            v_hat = v
            if k in key_to_type.keys():
                t = key_to_type[k]
            else:
                t = get_type_from_default(v)
            if isinstance(v, str):
                if v.endswith(","):
                    v_hat = v[:-1]
            self.parser.add_argument(f"--{k}", 
                                     default=v_hat, 
                                     type=t, 
                                     help=self.get_description_from_key(k))
            self.type_dict[k] = t

    def update_renamed_args(self,args):
        for k,v in self.renamed_args.items():
            if k in args.keys():
                assert v not in args.keys(), f"Tried to rename {k} to {v}, but both keys are present in args (unexpected)."
                args[v] = args[k]
                del args[k]
        return args

    def construct_args(self,tiers,tiers_dict=None,tiers_for_origin=list(range(4))):
        if tiers_dict is None:
            tiers_dict = self.tiers_dict
        tier_numbers = sorted(list(tiers_dict.values()),reverse=True)
        tiers_dict_inv = {v: k for k,v in tiers_dict.items()}
        origin = {}
        args = {}
        for tier_num in tier_numbers:
            tier_name = tiers_dict_inv[tier_num]
            for k,v in tiers[tier_name].items():
                if k not in self.type_dict.keys():
                    raise ValueError(f"Recieved unrecognized argument k={k} from source: {tier_name}. Closest known matches: {get_closest_matches(k,self.type_dict.keys(),n=3)}")
            args.update(tiers[tier_name])
            origin.update({k: tier_name for k in tiers[tier_name].keys()})
        tfo = [tiers_dict_inv[k] for k in tiers_for_origin]
        args["origin"] = {k: v for k,v in origin.items() if v in tfo}
        return args

    def get_command_line_args(self,alt_parse_args=None,use_parser=False):
        
        if alt_parse_args is None:
            commandline_list = sys.argv[1:]
        else:
            commandline_list = alt_parse_args
        if len(commandline_list)==1:
            if commandline_list[0]=="--help":
                self.parser.print_help()
                sys.exit()
        assert len(commandline_list)%2==0, f"commandline_list={commandline_list} must have an even number of elements."
        assert all([x.startswith("--") for x in commandline_list[::2]]), f"All even elements of commandline_list={commandline_list} must start with --."
        assert all([not x.startswith("--") for x in commandline_list[1::2]]), f"All odd elements of commandline_list={commandline_list} must not start with --."
        if use_parser:
            args = self.parser.parse_args(commandline_list).__dict__
        else:
            args = {k[2:]: v for (k,v) in zip(commandline_list[::2],commandline_list[1::2])}
        return args

    def get_name_based_args(self,name):
        assert isinstance(name,str), f"name={name} not a valid type."
        if name.find(";")>=0:
            return {}, {}, {}
        name_based_args = json.loads((Path(__file__).parent.parent.parent/self.filename_model).read_text())

        if "+" in name:
            plus_names = name.split("+")[1:]
            root_name = name.split("+")[0]
        else:
            plus_names = []
            root_name = name
        ver_names = []
        if ("[" in name) and ("]" in name):
            for _ in range(name.count("[")):
                idx0 = root_name.find("[")
                idx1 = root_name.find("]")
                if idx0<0 or idx1<0:
                    raise ValueError(f"name={name} has mismatched brackets.")
                ver_names.append(root_name[idx0+1:idx1])
                root_name = root_name[:idx0] + root_name[idx1+1:]
        #check that we are not using illegal version names (keys)
        all_keys = self.defaults_func().keys()
        for ver_name in ver_names:
            if ver_name in all_keys:
                raise ValueError(f"ver_name={ver_name} is not a valid version name because it is already a key for args.")
        if root_name in name_based_args.keys():
            root_name_args = name_based_args[root_name]
        else:
            raise ValueError(f"name={root_name} not found in name_based_args")
        ver_name_args = {}
        if len(ver_names)>0:
            assert "versions" in name_based_args[root_name].keys(), f"name={root_name} does not have versions."
            assert all([k in name_based_args[root_name]["versions"].keys() for k in ver_names]), f"ver_names={ver_names} not found in name_based_args[root_name]['versions'].keys()={name_based_args[root_name]['versions'].keys()}"
            for k,v in name_based_args[root_name]["versions"].items():
                if k in ver_names:
                    ver_name_args.update(v)
        plus_name_args = {}
        for pn in plus_names:
            assert "+"+pn in name_based_args.keys(), f"plus_name={pn} not found in name_based_args."
            plus_name_args.update(name_based_args["+"+pn])
        if "versions" in root_name_args.keys():
                del root_name_args["versions"]
        return root_name_args, plus_name_args, ver_name_args

    def get_args(self,alt_parse_args=None,modified_args={}):
        tiers = {k: {} for k in self.tiers_dict.keys()}

        tiers["modified_args"] = modified_args # top default priority
        tiers["commandline"] = self.get_command_line_args(alt_parse_args)
        tiers["defaults_current"] = self.defaults_func()
        name = self.construct_args(tiers)[self.name_key]
        root_name_args, plus_name_args, ver_name_args = self.get_name_based_args(name=name)
        tiers["name_plus"] = plus_name_args
        tiers["name_versions"] = ver_name_args
        tiers["name_root"] = root_name_args
        version = self.construct_args(tiers)[self.version_key]
        tiers["defaults_version"] = self.defaults_func(version=version)
        #find version only after all other args are set
        args = self.construct_args(tiers)
        #map to the correct types
        args = self.parse_types(args)
        if any([isinstance(v,list) for (k,v) in args.__dict__.items()]):
            modified_args_list = []
            num_modified_args = 1
            for k,v in args.__dict__.items():
                if isinstance(v,list):
                    if len(v)>1:
                        num_modified_args *= len(v)
                        if num_modified_args>100:
                            raise ValueError(f"Too many modified args. num_modified_args={num_modified_args}")
                        if len(modified_args_list)==0:
                            modified_args_list.extend([{k: v2} for v2 in v])
                        else:
                            modified_args_list = [{**d, k: v2} for d in modified_args_list for v2 in v]
            if len(modified_args_list)>0:
                return modified_args_list
        setattr(args,self.id_key,self.get_unique_id(args))
        return args
    
    def parse_types(self, args):
        args_dict = {k: v if isinstance(v,list) else self.type_dict[k](v) for k,v in args.items()}
        args = argparse.Namespace(**args_dict)
        return args
    
    def get_description_from_key(self, k):
        if k in self.descriptions.keys():
            return self.descriptions[k]
        else:
            return ""

    def load_and_format_id_dict(self,return_type="dict"):
        id_list = json.loads((Path(__file__).parent.parent.parent/self.filename_ids).read_text())
        if return_type=="list":
            return id_list
        elif return_type=="dict":
            id_dict = {}
        elif return_type=="ordereddict":
            id_dict = OrderedDict()
        else:
            raise ValueError(f"return_type={return_type} not supported. must be 'list', 'dict', or 'ordereddict'.")
        for item in id_list:
            id_of_item = item[self.id_key]
            id_dict[id_of_item] = item
        return id_dict
    
    def is_unique_id(self, id):
        id_dict = self.load_and_format_id_dict()
        return id not in id_dict.keys()

    def get_unique_id(self, args):
        if not (Path(__file__).parent.parent.parent/self.filename_ids).exists():
            (Path(__file__).parent.parent.parent/self.filename_ids).write_text("[]")
        id_dict = self.load_and_format_id_dict()
        id = args.__dict__[self.id_key]
        for k,v in args.__dict__.items():
            id = id.replace(f"[{k}]",str(v))
        if id.find("*")>=0:
            if len(id_dict)==0:
                id = id.replace("*","0")
            else:
                for i in range(len(id_dict)):
                    if id.replace("*",str(i)) not in id_dict.keys():
                        id = id.replace("*",str(i))
                        break
        assert id not in id_dict.keys(), f"id={id} already exists in id_dict.keys(). use a starred expression to get a unique id."
        return id
    
def get_type_from_default(default_v):
    assert isinstance(default_v,(float,int,str,bool)), f"default_v={default_v} is not a valid type."
    if isinstance(default_v, str):
        assert default_v.find(";")<0, f"semicolon not supported in default arguments"
        t2 = lambda x: str(x[:-1]) if x.endswith(",") else str(x)
    else:
        t2 = type(default_v)
    t = list_wrap_type(str2bool if isinstance(default_v, bool) else t2)
    return t

def get_closest_matches(k, list_of_things, n=3):
    """finds the n closest matched between a specified string, k, and the 
    keys of the type_dict. Closeness is measured by intersection 
    (len(intersection) where intersection is the longest common substring) 
    over union (len(k1) + len(k2) - len(intersection))."""
    iou_per_key = {}
    for k2 in list_of_things:
        intersection = longest_common_substring(k,k2)
        iou_per_key[k2] = len(intersection)/(len(k)+len(k2)-len(intersection))
    return [a[0] for a in sorted(iou_per_key.items(), key=lambda x: x[1], reverse=True)[:n]]

def load_existing_args(path_or_id,
                  name_key="args",
                  verify_keys=True,
                  origin_replace_keys=["commandline","modified_args"],
                  use_loaded_dynamic_args=True,
                  renamed_load_keys=True,
                  behavior_on_mismatch="raise"):
    assert behavior_on_mismatch in ["raise","theo","loaded"], f"behavior_on_mismatch={behavior_on_mismatch} must be one of ['raise','theo','loaded']"
    tp = TieredParser(name_key)
    if str(path_or_id).endswith(".json"):
        args_loaded = json.loads(Path(path_or_id).read_text())
        if isinstance(args_loaded,list):
            assert len(args_loaded)==1, f"Expected len(args_loaded)==1, but len(args_loaded)={len(args_loaded)}."
            args_loaded = args_loaded[0]
    else:
        id_dict = tp.load_and_format_id_dict()
        assert path_or_id in id_dict.keys(), f"path_or_id={path_or_id} not found in id_dict.keys(). Closest matches: {get_closest_matches(path_or_id,id_dict.keys(),n=3)}"
        args_loaded = id_dict[path_or_id]
    if not "origin" in args_loaded.keys():
        args_loaded["origin"] = {}
    if not tp.version_key in args_loaded.keys():
        args_loaded[tp.version_key] = "0.0.0"
    args_loaded = tp.update_renamed_args(args_loaded)
    modified_args = {tp.name_key: args_loaded[tp.name_key],
                     tp.version_key: args_loaded[tp.version_key]}
    
    for k,v in args_loaded["origin"].items():
        if v in origin_replace_keys:
            modified_args[k] = args_loaded[k]
    args_theoretical = tp.get_args(alt_parse_args=[],modified_args=modified_args)
    theo_keys = args_theoretical.__dict__.keys()
    load_keys = args_loaded.keys()
    all_keys = set(theo_keys).union(set(load_keys))
    for k in all_keys:
        if k in tp.dynamic_args:
            #dynamic args are allowed to be different
            if (k in theo_keys) and (k in load_keys):
                if use_loaded_dynamic_args:
                    pass
                else:
                    args_loaded[k] = args_theoretical.__dict__[k]
            if (k in theo_keys) and (k not in load_keys):
                args_loaded[k] = args_theoretical.__dict__[k]
        else:
            if (k in theo_keys) and (k in load_keys):
                if (args_theoretical.__dict__[k] != args_loaded[k]) and verify_keys:
                    if behavior_on_mismatch=="raise":
                        raise ValueError(f"args_theoretical.__dict__[{k}]={args_theoretical.__dict__[k]} != args_loaded[{k}]={args_loaded[k]}")
                    elif behavior_on_mismatch=="theo":
                        args_loaded[k] = args_theoretical.__dict__[k]
                    elif behavior_on_mismatch=="loaded":
                        pass
                    else:
                        raise ValueError(f"behavior_on_mismatch={behavior_on_mismatch} must be one of ['raise','theo','loaded']")
            elif (k in theo_keys) and (k not in load_keys):
                args_loaded[k] = args_theoretical.__dict__[k]
            elif (k not in theo_keys) and (k in load_keys):
                if verify_keys and (k not in tp.deprecated_args):
                    raise ValueError(f"args_theoretical.__dict__ does not contain key {k} but args_loaded does.")
            else:
                raise ValueError(f"VERY UNEXPECTED BUG: key {k} not found in either args_theoretical.__dict__ or args_loaded.")
    return argparse.Namespace(**args_loaded)

def add_folder_ids(folder,
                   require_args=True,
                   require_sample_opts=True,
                   atmost_one_sample_opts_file=True,
                   change_save_path=True):
    """
    Adds all ids from a folder which was copied into the saves folder.
    If ids already exist, they are modified to be unique and then added.
    ids means model_id and gen_id.
    """
    if isinstance(folder,str):
        folder = [folder]
    for f in folder:
        path = Path(f)
        if path.is_absolute():
            path = path.relative_to(Path.cwd())
        assert path.exists(), f"{path} does not exist"
        args = None
        
        if (path/"args.json").exists():
            args = load_existing_args(path_or_id=str(path/"args.json"))
        sample_opts_list = None
        sample_opts_paths = list(path.glob("**/sample_opts.json"))
        if len(sample_opts_paths)>0:
            if atmost_one_sample_opts_file:
                assert len(sample_opts_paths)==1, f"len(sample_opts_paths)={len(sample_opts_paths)} must be at most 1."
            sample_opts_list = load_json_to_dict_list(str(sample_opts_paths[0]))
        if require_sample_opts:
            assert sample_opts_list is not None, f"{path/'**/sample_opts.json'} does not exist"
        if require_args:
            assert args is not None, f"{path/'args.json'} does not exist"
        if args is not None:
            tpa = TieredParser()
            if not tpa.is_unique_id(args.model_id):
                args.model_id = args.model_id+"_*"
                args.model_id = tpa.get_unique_id(args)
            if change_save_path:
                actual_save_path = str(path)
                args.save_path = actual_save_path
        if sample_opts_list is not None:
            assert isinstance(sample_opts_list,list), f"sample_opts_list={sample_opts_list} is not a list."
            tpso = TieredParser("sample_opts")
            for i in range(len(sample_opts_list)):
                sample_opts = argparse.Namespace(**sample_opts_list[i])
                if not tpso.is_unique_id(sample_opts.gen_id):
                    sample_opts.gen_id = sample_opts.gen_id+"_*"
                    sample_opts.gen_id = tpso.get_unique_id(sample_opts)
                    if args is not None:
                        sample_opts.model_id = args.model_id
                sample_opts_list[i] = sample_opts
        if args is not None:
            save_args(args)
        if sample_opts_list is not None:
            for sample_opts in sample_opts_list:
                save_args(sample_opts)

def delete_existing_args(id,name_key="sample_opts"):
    tp = TieredParser(name_key)
    id_dict = tp.load_and_format_id_dict()
    assert id in id_dict.keys(), f"id={id} not found in id_dict.keys(). Closest matches: {get_closest_matches(id,id_dict.keys(),n=3)}"
    overwrite_existing_args(argparse.Namespace(**id_dict[id]),delete_instead_of_overwrite=True)

def overwrite_existing_args(args,delete_instead_of_overwrite=False):
    local_path, global_path = save_args(args,dry=True)
    if hasattr(args,"model_name"):
        tp = TieredParser("args")
    elif hasattr(args,"gen_setup"):
        tp = TieredParser("sample_opts")
    else:
        raise ValueError(f"Expected args to contain either model_name or gen_setup.")
    id_dict = tp.load_and_format_id_dict()
    if not args.__dict__[tp.id_key] in id_dict.keys():
        raise ValueError(f"id with name {args.__dict__[tp.id_key]} not found in id_dict.keys().")
    dict_list = load_json_to_dict_list(global_path)
    for i in range(len(dict_list)):
        if dict_list[i][tp.id_key] == args.__dict__[tp.id_key]:
            if delete_instead_of_overwrite:
                del dict_list[i]
            else:
                dict_list[i] = args.__dict__
            break
    save_dict_list_to_json(dict_list,global_path,append=False)
    dict_list = load_json_to_dict_list(local_path)
    for i in range(len(dict_list)):
        if dict_list[i][tp.id_key] == args.__dict__[tp.id_key]:
            if delete_instead_of_overwrite:
                del dict_list[i]
            else:
                dict_list[i] = args.__dict__
            break
    save_dict_list_to_json(dict_list,local_path,append=False)

def save_args(args, local_path=None, global_path=None, dry=False, do_nothing=False):
    if do_nothing:
        return local_path, global_path
    if hasattr(args,"model_name"):
        if local_path is None:
            local_path = args.save_path+"/args.json"
        else:
            assert local_path.endswith(".json"), f"local_path={local_path} must end with .json"
        tp = TieredParser("args")
    elif hasattr(args,"gen_setup"):
        if local_path is None:
            local_path = str(Path(args.default_save_folder)/"sample_opts.json")
        else:
            assert local_path.endswith(".json"), f"local_path={local_path} must end with .json"
        tp = TieredParser("sample_opts")
    else:
        raise ValueError(f"Expected args to contain either model_name or gen_setup.")
    
    if global_path is None:
        global_path = tp.filename_ids
    else:
        assert global_path.endswith(".json"), f"global_path={global_path} must end with .json"
    assert Path(local_path).parent.exists(), f"Path(local_path).parent={Path(local_path).parent} does not exist."
    assert Path(global_path).parent.exists(), f"Path(global_path).parent={Path(global_path).parent} does not exist."
    if not dry:
        save_dict_list_to_json([args.__dict__],str(global_path),append=True)
        save_dict_list_to_json([args.__dict__],str(local_path),append=True)
    return local_path, global_path

def kill_missing_ids(name_key="args",dry=False,keep_criterion=None):
    """Checks ids for sample_opts and and args and removes ids
    which do not have a corresponding folder or files in the
    save folder."""
    if keep_criterion is None:
        if name_key=="args":
            #check if the save_path exists
            keep_criterion = lambda x: any([(Path(x[k]).exists() and len(x[k])>0) for k in ["save_path"]])
        elif name_key=="sample_opts":
            #grid_filename,concat_inter_filename,raw_samples_folder,inter_folder
            s_opt_keys = "grid_filename,concat_inter_filename,raw_samples_folder,inter_folder".split(",")
            keep_criterion = lambda x: any([(Path(x[k]).exists() and len(x[k])>0) for k in s_opt_keys])
    tp = TieredParser(name_key)
    id_dict = tp.load_and_format_id_dict()
    include_keys = []
    for k,v in id_dict.items():
        if keep_criterion(v):
            include_keys.append(k)
        else:
            print(f"Removing id={k} from {name_key}.")
    print(f"Keeping {len(include_keys)} ids out of {len(id_dict)} in {name_key}.")
    if not dry:
        new_dict_list = [id_dict[k] for k in include_keys]
        save_dict_list_to_json(new_dict_list,tp.filename_ids,append=False)

def kill_by_id(id, name_key="sample_opts", dry=False):
    tp = TieredParser(name_key)
    id_dict = tp.load_and_format_id_dict(return_type="ordereddict")
    assert id in id_dict.keys(), f"id={id} not found in id_dict.keys(). Closest matches: {get_closest_matches(id,id_dict.keys(),n=3)}"
    if dry:
        print(f"Would remove id={id} from {name_key}.")
    else:
        if name_key=="args":
            raise NotImplementedError("name_key=args not supported.")
        else:
            assert name_key=="sample_opts", f"name_key={name_key} not supported."
            remove_keys = "grid_filename,light_stats_filename,raw_samples_folder,concat_inter_filename,inter_folder"
            for k in remove_keys.split(","):
                if id_dict[id][k]=="":
                    continue
                elif not Path(id_dict[id][k]).exists():
                    continue
                if k.endswith("_filename"):
                    Path(id_dict[id][k]).unlink()
                else:
                    rmtree(id_dict[id][k])
            delete_existing_args(id,name_key=name_key)

def get_current_default_version():
    tp = TieredParser("args")
    return load_defaults(idx=0)[tp.version_key]

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0: version comparison test")
        #test all possible cases
        version_conditions = ["==0.1.2", "0.1.2<", "0.1.2<=", "0.0.0<=x<=1.0.0<x",">=0.999"]
        x = "1.0.0"
        for version_condition in version_conditions:
            print(f"eval_version_condition({version_condition},{x})={eval_version_condition(version_condition=version_condition,x=x)}")
    elif args.unit_test==1:
        print("UNIT TEST 1: dry kill_missing_ids")
        kill_missing_ids(name_key="args",dry=True)
        kill_missing_ids(name_key="sample_opts",dry=True)
    elif args.unit_test==2:
        print("UNIT TEST 2: kill_missing_ids")
        kill_missing_ids(name_key="args",dry=False)
        kill_missing_ids(name_key="sample_opts",dry=False)
    elif args.unit_test==3:
        print("UNIT TEST 3: get_current_default_version")
        print(get_current_default_version())
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()