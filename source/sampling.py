import torch
import numpy as np
import matplotlib
from argparse import Namespace
from collections import defaultdict
import os
import tqdm
from utils import get_segment_metrics,SmartParser
from plot_utils import plot_grid,plot_inter,concat_inter_plots
#from cont_gaussian_diffusion import DummyDiffusion TODO
"""
def get_named_gen_setup(name):
    opts = get_default_sampler_options()
    valid_sampler_names = ["default","vali","train","gw2","self_cond","self_cond_gw2","ddim"]
    if name=="vali":
        pass
    elif name=="train":
        opts.split = "train"
    elif name=="gw2":
        opts.guidance_weight = 2.0
    elif name=="self_cond":
        opts.self_cond = True
    elif name=="self_cond_gw2":
        opts.guidance_weight = 2.0
        opts.self_cond = True
    elif name=="ddim":
        opts.sampler_type = "ddim"
    else:
        raise ValueError(f"Unknown sampler name: {name}, must be one of {valid_sampler_names}")
    return opts


def get_default_sampler_options(return_dict=True,return_description=False):
    dict_options = dict(
            clip_denoised               = [True,            "Whether to clip denoised samples to [0,1] range. TYPE: bool. DEFAULT: True"],
            num_timesteps               = [100,             "Number of timesteps to use for sampling. TYPE: int. DEFAULT: 100"],
            guidance_weight             = [0.0,             "Weight of classifier free diffusion guidance, 0.0 for No guidance. TYPE: float. DEFAULT: 0.0"],
            guidance_kwargs             = ['',              "Keyword arguments which are passed to the guidance forward pass, empty for none. TYPE: str. DEFAULT: ''"],
            eval_batch_size             = [0,               "Batch size for evaluation. 0 for same as train. TYPE: int. DEFAULT: 0"],
            sampler_type                = ["ddpm",          "Type of sampler to use. One of ['ddpm','ddim']. TYPE: str. DEFAULT: 'ddpm'"],
            self_cond                   = [False,           "Whether to use self conditioning. TYPE: bool. DEFAULT: False"],

            ema_model_rate              = [0,               "EMA model rate to use for sampling. 0 for no EMA (normal model). TYPE: int. DEFAULT: 0"],
            split                       = ["vali",          "Dataset split to use for sampling. One of ['train','vali','test']. TYPE: str. DEFAULT: 'vali'"],
            kwargs_mode                 = ["train",         "How to get kwargs for the model forward pass. One of ['train']. TYPE: str. DEFAULT: 'train'"],
            
            do_agg                      = [True,            "Whether to use matplotlib 'agg' as backend for plotting (to avoid memory leak). TYPE: bool. DEFAULT: True"],
            progress_bar                = [True,            "Whether to show a progress bar for batches. TYPE: bool. DEFAULT: True"],
            progress_bar_timestep       = [False,           "Whether to show a second progress bar running over timesteps during sampling. TYPE: bool. DEFAULT: False"],

            save_raw_inter              = [False,           "Whether to save raw intermediate samples. Needs save_raw_samples==True to have an effect. TYPE: bool. DEFAULT: False"],
            save_raw_samples            = [False,           "Whether to save raw samples. TYPE: bool. DEFAULT: False"],
            return_samples              = [False,           "Whether to return samples. TYPE: bool. DEFAULT: False"],
            remove_old                  = [False,           "Should old plots matching the format and in the same folder of newly produced ones be removed? TYPE: bool. DEFAULT: False"],
            plotting_functions          = ["grid,inter",    "Which plotting functions to use. Comma seperated string of zero or more of ['grid','inter']. TYPE: str. DEFAULT: 'grid,inter'"],

            #how much to sample
            num_votes                   = [5,               "Number of votes (segmentation masks) to use per image when sampling. TYPE: int. DEFAULT: 5"],
            num_samples                 = [8,               "Number of samples (unique images) to produce votes for. TYPE: int. DEFAULT: 8"],
            num_inter_steps             = [10,              "Number of unique intermediate steps to save for saving/plotting (limited by num_timesteps also). TYPE: int. DEFAULT: 10"],
            num_inter_samples           = [8,               "Number of intermediate samples to save for saving/plotting. TYPE: int. DEFAULT: 8"],
            num_grid_samples            = [8,               "Number of samples to use for a grid plot. TYPE: int. DEFAULT: 8"],

            inter_votes_per_sample      = [1,               "How many votes per image (e.g. 1=only first vote) should intermediate steps be saved for. TYPE: int. DEFAULT: 1"],

            #where to save things
            save_plot_grid_path         = ["",              "desc"],
            save_plot_inter_path        = ["",              "desc"],
            save_raw_samples_path       = ["",              "desc"],
            save_concat_plot_inter_path = ["",              "desc"],
            default_save_folder         = ["samples",       "desc"],
            )
    idx = 1 if return_description else 0
    for key in dict_options:
        dict_options[key] = dict_options[key][idx]
    if return_dict:
        return dict_options
    else:
        return Namespace(**dict_options)
"""
class DiffusionSampler(object):
    def __init__(self, trainer, opts=None,
                 ):
        super().__init__()
        self.is_dummy_diffusion = False#isinstance(diffusion,DummyDiffusion) TODO
        if opts is None:
            opts = SmartParser("sample_opts").get_args([])
        self.opts = opts
        self.trainer = trainer
        self.setup_name = "vali"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("WARNING: CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
    
    def load_gen_setup(self,setup_name,modified_args={}):
        self.setup_name = setup_name
        self.opts = SmartParser("sample_opts").get_args(alt_parse_args=["--gen_setup",self.setup_name])
        for k,v in modified_args.items():
            self.opts.__dict__[k] = v

    def prepare_sampling(self,model=None):
        #init variables
        self.samples = []
        self.source_idx = 0
        self.bss = 0
        self.source_batch = None
        self.queue = None
        if self.opts.eval_batch_size==0:
            self.opts.eval_batch_size = self.trainer.args.train_batch_size

        if self.opts.split=="train":
            self.dataloader = self.trainer.train_dl
        elif self.opts.split=="vali":
            self.dataloader = self.trainer.vali_dl
        elif self.opts.split=="test":
            raise NotImplementedError("Test split not implemented.")
        
        if model is None:
            if self.opts.ema_model_rate>0:
                model = self.trainer.get_ema_model(self.opts.ema_model_rate)
            else:
                model = self.trainer.model
        was_training = model.training
        model.eval()

        if self.opts.do_agg:
            old_backend = matplotlib.get_backend()
            matplotlib.use("agg")
        else:
            old_backend = None

        if "grid" in self.opts.plotting_functions.split(",") and self.opts.num_grid_samples>0:
            if self.opts.save_plot_grid_path=="":
                output_folder = os.path.join(self.trainer.args.save_path,self.opts.default_save_folder)
                self.opts.save_plot_grid_path = os.path.join(output_folder,f"plot_grid_{self.setup_name}_{self.trainer.step:06d}.png")
        inter_is_used = ("inter" in self.opts.plotting_functions.split(",") and self.opts.num_inter_samples>0) or (self.opts.save_raw_inter and self.opts.save_raw_samples)
        
        if inter_is_used:
            legal_timesteps = list(range(self.opts.num_timesteps-1, -1, -1))
            idx = np.round(np.linspace(0,len(legal_timesteps)-1,self.opts.num_inter_steps)).astype(int)
            #remove duplicates without changing order
            idx = [idx[i] for i in range(len(idx)) if (i==0) or (idx[i]!=idx[i-1])]
            self.save_i_steps = [legal_timesteps[i] for i in idx]
            if self.opts.save_plot_inter_path=="":
                output_folder = os.path.join(self.trainer.args.save_path,self.opts.default_save_folder)
                self.opts.save_plot_inter_path = os.path.join(output_folder,f"inter")
            if self.opts.save_concat_plot_inter_path=="":
                output_folder = os.path.join(self.trainer.args.save_path,self.opts.default_save_folder)
                self.opts.save_concat_plot_inter_path = os.path.join(output_folder,f"plot_inter_concat_{self.setup_name}_{self.trainer.step:06d}.png")
        else:
            self.save_i_steps = []
        if self.opts.save_raw_samples:
            if self.opts.save_raw_samples_path=="":
                output_folder = os.path.join(self.trainer.args.save_path,self.opts.default_save_folder)
                self.opts.save_raw_samples_path = os.path.join(output_folder,f"raw_samples_{self.setup_name}_{self.trainer.step:06d}.pt")

        if self.opts.save_raw_samples_path!="":
            os.makedirs(self.opts.save_raw_samples_path,exist_ok=True)
        if self.opts.save_plot_inter_path!="":
            os.makedirs(self.opts.save_plot_inter_path,exist_ok=True)
        if self.opts.save_concat_plot_inter_path!="":
            os.makedirs(os.path.dirname(self.opts.save_concat_plot_inter_path),exist_ok=True)
        if self.opts.save_plot_grid_path!="":
            os.makedirs(os.path.dirname(self.opts.save_plot_grid_path),exist_ok=True)

        return model, was_training, old_backend

    def verify_valid_opts(self):
        inter_is_used = ("inter" in self.opts.plotting_functions.split(",") and self.opts.num_inter_samples>0) or (self.opts.save_raw_inter and self.opts.save_raw_samples)
        if inter_is_used:
            assert self.opts.num_samples>=self.opts.num_inter_samples, "num_samples must be at least as large as num_inter_samples."
            assert self.opts.num_votes  >=self.opts.inter_votes_per_sample, "num_votes must be at least as large as inter_votes_per_sample."
            assert self.opts.num_inter_samples>0, "num_inter_samples must be positive."
        if "grid" in self.opts.plotting_functions.split(","):
            assert self.opts.num_samples>=self.opts.num_grid_samples, "num_samples must be at least as large as num_grid_samples."
        assert self.opts.num_votes>0, "num_votes must be positive."
        assert self.opts.num_samples>=0, "num_samples must be non-negative."
        if self.opts.return_samples>64:
            print(f"WARNING: return_samples={self.opts.return_samples} is very large. This may cause memory issues.")
        
        modified_opts_wrt_setup = {}
        ref_opts = SmartParser("sample_opts").get_args(alt_parse_args=["--gen_setup",self.setup_name])
        for k,v in self.opts.__dict__.items():
            if v!=ref_opts.__dict__[k]:
                modified_opts_wrt_setup[k] = v
        return modified_opts_wrt_setup

    def sample(self,model=None,**kwargs):
        self.opts = Namespace(**{**vars(self.opts),**kwargs})
        modified_opts_wrt_setup = self.verify_valid_opts()
        model,was_training,old_backend = self.prepare_sampling(model)
        
        self.queue = None
        metric_list = []
        votes = []
        num_batches = np.ceil(self.opts.num_samples*self.opts.num_votes/self.opts.eval_batch_size).astype(int)
        if num_batches==0:
            print("WARNING: num_batches==0.")
            return None
        if self.opts.progress_bar:
            progress_bar = tqdm.tqdm(range(num_batches), desc="Batch progress.")
        else:
            progress_bar = range(num_batches)
        with torch.no_grad():
            for batch_ite in progress_bar:
                x_true, model_kwargs, info, batch_queue = self.form_next_batch()
                x_true_bit = self.trainer.cgd.ab.int2bit(x_true)
                
                x_init = torch.randn_like(x_true_bit)
                sample_output = self.trainer.cgd.sample_loop(model=model, 
                                            x_init=x_init, 
                                            num_steps=self.opts.num_timesteps, 
                                            sampler_type=self.opts.sampler_type,
                                            clip_x=self.opts.clip_denoised,
                                            model_kwargs=model_kwargs,
                                            guidance_weight=self.opts.guidance_weight,
                                            progress_bar=self.opts.progress_bar_timestep,
                                            save_i_steps=self.save_i_steps,
                                            save_i_idx=[bq["save_inter_steps"] for bq in batch_queue],
                                            self_cond=self.opts.self_cond,
                                            guidance_kwargs=self.opts.guidance_kwargs,
                                            )
                self.run_on_single_batch(sample_output,batch_queue,x_init,x_true_bit,model_kwargs,batch_ite)
                for i in range(sample_output["pred"].shape[0]):
                    votes.append(sample_output["pred"][i])
                    if batch_queue[i]["vote"]==self.opts.num_votes-1:
                        model_kwargs_i = {k: model_kwargs[k][i] for k in model_kwargs.keys() if model_kwargs[k] is not None}
                        metrics = self.run_on_full_votes(votes,x_true[i],x_true_bit[i],info[i],model_kwargs_i,x_init[i])
                        votes = []
                        metric_list.append(metrics)

        sample_output, metric_ouput = self.get_output_dict(metric_list, self.samples)
        self.run_on_finished(output={**sample_output,**metric_ouput})

        if old_backend is not None:
            matplotlib.use(old_backend)
        if was_training:
            model.train()
        
        metric_ouput["setup_name"] = self.setup_name
        metric_ouput["modded_opts"] = str(modified_opts_wrt_setup)


        if not self.opts.return_samples:
            sample_output = None
        return sample_output, metric_ouput
    
    def get_output_dict(self, metric_list, samples):
        sample_output = {}
        metric_ouput = {k: [m[k] for m in metric_list] for k in metric_list[0].keys()}
        if samples is not None:
            for k in samples[0].keys():
                if k=="info":
                    continue
                if isinstance(samples[0][k],dict):
                    for sub_k in samples[0][k].keys():
                        if not torch.is_tensor(samples[0][k][sub_k]):
                            raise ValueError(f"Expected tensor for samples[0][{k}][{sub_k}]. Got {type(samples[0][k][sub_k])}.")
                        sample_output[sub_k] = torch.stack([s[k][sub_k] for s in samples],dim=0)
                elif torch.is_tensor(samples[0][k]):
                    sample_output[k] = torch.stack([s[k] for s in samples],dim=0)
        
        return sample_output, metric_ouput
            
    def run_on_single_batch(self,sample_output,bq,x_init,x_true_bit,model_kwargs,batch_ite):
        sample_output["x"] = x_true_bit      
        if self.opts.save_plot_inter_path!="":
            save_i_idx = [bq_i["save_inter_steps"] for bq_i in bq]
            plot_inter(foldername=self.opts.save_plot_inter_path,
                       sample_output=sample_output,
                       model_kwargs=model_kwargs,
                       ab=self.trainer.cgd.ab,
                       save_i_idx=save_i_idx,
                       plot_text=self.opts.save_concat_plot_inter_path=="")
        if self.opts.save_raw_samples_path!="":
            if not hasattr(self,"raw_samples"):
                self.raw_samples = []
            if not self.opts.save_raw_inter:
                if "inter" in sample_output.keys():
                    del sample_output["inter"]
            sample_output["x_init"] = x_init
            sample_output["batch_queue"] = bq
            sample_output["model_kwargs"] = model_kwargs
            torch.save(sample_output,os.path.join(self.opts.save_raw_samples_path,f"raw_sample_batch{batch_ite:3d}.pt"))
        
    def run_on_full_votes(self,votes,x_true,x_true_bit,info,model_kwargs,x_init):
        x_true = x_true.cpu()
        x_true_bit = x_true_bit.cpu()
        votes = torch.stack(votes,dim=0).cpu()
        votes_int = self.trainer.cgd.ab.bit2int(votes)            
        self.samples.append({"pred_bit": votes,
                             "pred_int": votes_int,
                             "target_bit": x_true_bit,
                             "target_int": x_true,
                             "x_init": x_init,
                             "info": info,
                             "model_kwargs": model_kwargs})
        metrics = defaultdict(list)
        for i in range(len(votes)):            
            metrics_i = get_segment_metrics(votes_int[i],x_true)
            for k in metrics_i.keys():
                metrics[k].append(metrics_i[k])
        return metrics
    
    def run_on_finished(self,output):
        if self.opts.save_plot_grid_path!="":
            assert self.opts.save_plot_grid_path.endswith(".png"), f"filename: {filename}"
            filename = self.opts.save_plot_grid_path
            plot_grid(filename,output,self.trainer.cgd.ab,max_images=32,remove_old=self.opts.remove_old)
        if self.opts.save_concat_plot_inter_path!="":
            assert self.opts.save_concat_plot_inter_path.endswith(".png"), f"filename: {filename}"
            concat_inter_plots(foldername = self.opts.save_plot_inter_path,
                               concat_filename = self.opts.save_concat_plot_inter_path,
                               num_timesteps = self.opts.num_inter_steps,
                               remove_old = self.opts.remove_old)
        

    def get_kwargs(self,batch):
        if self.opts.kwargs_mode=="train":
            assert self.trainer is not None, "self.trainer is None. Set self.trainer to a DiffusionModelTrainer instance or a class with a usable get_kwargs() method."
            x,model_kwargs,info = self.trainer.get_kwargs(batch, gen=True)
            if "points" in model_kwargs.keys():
                if model_kwargs["points"] is not None:
                    model_kwargs["points"] = model_kwargs["points"]*self.trainer.cgd.ab.int2bit(x)
        elif self.opts.kwargs_mode=="none":
            x,info = batch
            x = x.to(self.device)
            model_kwargs = {}
        return x,model_kwargs,info
            
    def form_next_batch(self):
        if self.queue is None:
            
            self.queue = []
            for i in range(self.opts.num_samples):
                for j in range(self.opts.num_votes):
                    save_inter_steps = (i<self.opts.num_inter_samples) and (j<self.opts.inter_votes_per_sample)
                    self.queue.append({"sample":i,"vote":j,"save_inter_steps": save_inter_steps, "save_grid": (i<self.opts.num_grid_samples)})
        
        bs = min(self.opts.eval_batch_size,len(self.queue))
        if self.source_idx >= self.bss:
            self.source_batch = self.get_kwargs(next(self.dataloader))
            self.bss = self.source_batch[0].shape[0]
            self.source_idx = 0
        batch_x = []
        use_kwargs = [k for k,v in self.source_batch[1].items() if v is not None]
        batch_kwargs = {k: [] for k in use_kwargs}
        batch_info = []
        batch_queue = []
        for i in range(bs):
            batch_queue.append(self.queue.pop(0))
            batch_x.append(self.source_batch[0][self.source_idx])
            for k in use_kwargs:
                batch_kwargs[k].append(self.source_batch[1][k][self.source_idx])
            batch_info.append(self.source_batch[2][self.source_idx])
            
            if batch_queue[-1]["vote"]==self.opts.num_votes-1:
                self.source_idx += 1
                if (self.source_idx >= self.bss) and (not len(self.queue)==0):
                    self.source_batch = self.get_kwargs(next(self.dataloader))
                    self.bss = self.source_batch[0].shape[0]
                    self.source_idx = 0

        for k in batch_kwargs.keys():
            if batch_kwargs[k] is not None:
                batch_kwargs[k] = torch.stack(batch_kwargs[k],dim=0)
        batch_x = torch.stack(batch_x,dim=0)
        return batch_x, batch_kwargs, batch_info, batch_queue
    
    
def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0: do sampling")
        from utils import SmartParser
        from training import DiffusionModelTrainer

        #args.mode = "cont"
        #args.model_name = "weak30k[7]"
        alt_parse_args = ["--mode","cont","--model_name","weak30k[7]"]
        args = SmartParser().get_args(alt_parse_args=alt_parse_args)
        trainer = DiffusionModelTrainer(args)
        sampler = DiffusionSampler(trainer)
        sampler.load_gen_setup("gw2")
        sampler.opts.plotting_functions = "grid"
        output = sampler.sample()
    elif args.unit_test==1:
        pass
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()