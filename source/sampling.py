import torch
import numpy as np
import matplotlib
from argparse import Namespace
from collections import defaultdict
import os
import tqdm
from unet import unet_kwarg_to_tensor
from utils import get_segment_metrics,get_time,save_dict_list_to_json,check_keys_are_same
from plot_utils import plot_grid,plot_inter,concat_inter_plots,index_dict_with_bool
from argparse_utils import TieredParser, save_args, overwrite_existing_args
from pathlib import Path
import copy
#from cont_gaussian_diffusion import DummyDiffusion TODO

class DiffusionSampler(object):
    def __init__(self, trainer, opts=None,
                 ):
        super().__init__()
        self.is_dummy_diffusion = False#isinstance(diffusion,DummyDiffusion) TODO
        if opts is None:
            opts = TieredParser("sample_opts").get_args([])
        self.opts = opts
        self.trainer = trainer
        self.opts.seed = self.trainer.args.seed
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("WARNING: CUDA not available. Using CPU.")
            self.device = torch.device("cpu")

    def prepare_sampling(self,model=None):
        #init variables
        self.samples = []
        self.light_stats = []
        self.source_idx = 0
        self.bss = 0
        self.source_batch = None
        self.queue = None
        if self.opts.eval_batch_size==0:
            self.eval_batch_size = self.trainer.args.train_batch_size
        else:
            self.eval_batch_size = self.opts.eval_batch_size
        if len(self.opts.datasets)>0:
            if not isinstance(self.opts.datasets,list):
                self.opts.datasets = [self.opts.datasets]
        if len(self.opts.datasets)>0:
            self.trainer.args.datasets = self.opts.datasets
            self.trainer.create_datasets(self.opts.split)
            lpd = getattr(self.trainer,f"{self.opts.split}_dl").dataloader.dataset.len_per_dataset
            max_num_samples = sum([lpd[dataset] for dataset in self.opts.datasets])
            if self.opts.num_samples<0:
                self.opts.num_samples = max_num_samples
            elif self.opts.num_samples>max_num_samples:
                print(f"WARNING: num_samples={self.opts.num_samples} is larger than the maximum number of samples in the specified datasets: {max_num_samples}. Setting num_samples to the maximum.")
                self.opts.num_samples = max_num_samples
        else:
            if not hasattr(self.trainer,f"{self.opts.split}_dl"):
                self.trainer.create_datasets(self.opts.split)
        self.dataloader = getattr(self.trainer,f"{self.opts.split}_dl")

        if model is None:
            if self.opts.ema_idx>=0:
                model, self.swap_pointers_func = self.trainer.get_ema_model(self.opts.ema_idx)
            else:
                model = self.trainer.model
        was_training = model.training
        model.eval()

        if self.opts.do_agg:
            old_backend = matplotlib.get_backend()
            matplotlib.use("agg")
        else:
            old_backend = None
        if self.opts.default_save_folder=="":
            self.opts.default_save_folder = os.path.join(self.trainer.args.save_path,"samples")
        def_save_name = f"{self.opts.gen_id}_{self.trainer.step:06d}"
        if self.opts.save_light_stats:
            if self.opts.light_stats_filename=="":
                self.opts.light_stats_filename = os.path.join(self.opts.default_save_folder,f"light_stats_{def_save_name}.json")
        if "grid" in self.opts.plotting_functions.split(",") and self.opts.num_grid_samples>0:
            if self.opts.grid_filename=="":
                self.opts.grid_filename = os.path.join(self.opts.default_save_folder,f"grid_{def_save_name}.png")
        inter_is_used = self.opts.num_inter_samples>0
        inter_is_used = inter_is_used and (("inter" in self.opts.plotting_functions.split(",")) or ("concat" in self.opts.plotting_functions.split(",")))
        inter_is_used = inter_is_used or (self.opts.save_raw_inter and self.opts.save_raw_samples)
        if inter_is_used:
            legal_timesteps = list(range(self.opts.num_timesteps-1, -1, -1))
            idx = np.round(np.linspace(0,len(legal_timesteps)-1,self.opts.num_inter_steps)).astype(int)
            #remove duplicates without changing order
            idx = [idx[i] for i in range(len(idx)) if (i==0) or (idx[i]!=idx[i-1])]
            self.save_i_steps = [legal_timesteps[i] for i in idx]
            if self.opts.inter_folder=="":
                self.opts.inter_folder = os.path.join(self.opts.default_save_folder,f"inter_{def_save_name}")
            if "concat" in self.opts.plotting_functions.split(","):
                if self.opts.concat_inter_filename=="":
                    self.opts.concat_inter_filename = os.path.join(self.opts.default_save_folder,f"concat_{def_save_name}.png")
        else:
            self.save_i_steps = []
        if self.opts.save_raw_samples:
            if self.opts.raw_samples_folder=="":
                self.opts.raw_samples_folder = os.path.join(self.opts.default_save_folder,f"raw_samples_{def_save_name}")

        if self.opts.raw_samples_folder!="":
            os.makedirs(self.opts.raw_samples_folder,exist_ok=True)
        if self.opts.inter_folder!="":
            os.makedirs(self.opts.inter_folder,exist_ok=True)
        if self.opts.concat_inter_filename!="":
            os.makedirs(os.path.dirname(self.opts.concat_inter_filename),exist_ok=True)
        if self.opts.grid_filename!="":
            os.makedirs(os.path.dirname(self.opts.grid_filename),exist_ok=True)

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

    def sample(self,model=None,**kwargs):
        self.opts = Namespace(**{**vars(self.opts),**kwargs})
        
        print("Sampling with gen_id:",self.opts.gen_id)
        model,was_training,old_backend = self.prepare_sampling(model)
        self.verify_valid_opts()
        
        self.queue = None
        metric_list = []
        votes = []
        num_batches = np.ceil(self.opts.num_samples*self.opts.num_votes/self.eval_batch_size).astype(int)
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
                self.run_on_single_batch(sample_output,batch_queue,x_init,x_true_bit,model_kwargs,batch_ite,info)
                for i in range(sample_output["pred"].shape[0]):
                    votes.append(sample_output["pred"][i])
                    if batch_queue[i]["vote"]==self.opts.num_votes-1:
                        model_kwargs_i = {k: 
                                          (model_kwargs[k][i] if model_kwargs[k] is not None else None) 
                                          for k in model_kwargs.keys()}
                        metrics = self.run_on_full_votes(votes,x_true[i],x_true_bit[i],info[i],model_kwargs_i,x_init[i],batch_queue[i])
                        votes = []
                        metric_list.append(metrics)

        sample_output, metric_output = self.get_output_dict(metric_list, self.samples)
        self.run_on_finished(output={**sample_output,**metric_output})

        if old_backend is not None:
            matplotlib.use(old_backend)
        if was_training:
            model.train()
        
        metric_output["gen_setup"] = self.opts.gen_setup
        metric_output["gen_id"] = self.opts.gen_id

        if hasattr(self,"swap_pointers_func"):
            self.swap_pointers_func()
            del self.swap_pointers_func
        if not self.opts.return_samples:
            sample_output = None
        return sample_output, metric_output
    
    def get_output_dict(self, metric_list, samples, info_keys_save=["dataset_name","i"]):
        sample_output = {}
        metric_output = {k: [m[k] for m in metric_list] for k in metric_list[0].keys()}
        #check for key conflicts
        if samples is not None:
            assert check_keys_are_same(samples)
            for k in samples[0].keys():
                if k=="info":
                    sample_output[k] = [{sub_k: s[k][sub_k] for sub_k in info_keys_save} for s in samples]
                    continue
                if isinstance(samples[0][k],dict):
                    for sub_k in samples[0][k].keys():
                        assert check_keys_are_same([s[k] for s in samples])
                        #if not torch.is_tensor(samples[0][k][sub_k]):
                        #    raise ValueError(f"Expected tensor for samples[0][{k}][{sub_k}]. Got {type(samples[0][k][sub_k])}.")
                        sample_output[sub_k] = unet_kwarg_to_tensor([s[k][sub_k] for s in samples])
                elif torch.is_tensor(samples[0][k]):
                    sample_output[k] = unet_kwarg_to_tensor([s[k] for s in samples])
        
        return sample_output, metric_output
            
    def run_on_single_batch(self,sample_output,bq,x_init,x_true_bit,model_kwargs,batch_ite,info):
        sample_output = copy.deepcopy(sample_output)
        model_kwargs = copy.deepcopy(model_kwargs)
        info = copy.deepcopy(info)
        sample_output["x"] = x_true_bit
        if self.opts.inter_folder!="":
            save_i_idx = [bq_i["save_inter_steps"] for bq_i in bq]
            plot_inter(foldername=self.opts.inter_folder,
                       sample_output=sample_output,
                       model_kwargs=model_kwargs,
                       ab=self.trainer.cgd.ab,
                       save_i_idx=save_i_idx,
                       plot_text=self.opts.concat_inter_filename=="",
                       imagenet_stats=self.trainer.args.crop_method.startswith("sam"))
        if self.opts.save_raw_samples:
            save_bool = [bq_i["sample"]<self.opts.num_save_raw_samples for bq_i in bq]
            if any(save_bool):
                if not self.opts.save_raw_inter:
                    if "inter" in sample_output.keys():
                        del sample_output["inter"]
                
                sample_output["x_init"] = x_init
                sample_output["batch_queue"] = bq
                sample_output["model_kwargs"] = model_kwargs
                sample_output["info"] = info
                if not all(save_bool):
                    sample_output = index_dict_with_bool(sample_output,save_bool)

                torch.save(sample_output,os.path.join(self.opts.raw_samples_folder,f"raw_sample_batch{batch_ite:03d}.pt"))
            
    def run_on_full_votes(self,votes,x_true,x_true_bit,info,model_kwargs,x_init,bqi):
        x_true = x_true.cpu()
        x_true_bit = x_true_bit.cpu()
        votes = torch.stack(votes,dim=0).cpu()
        votes_int = self.trainer.cgd.ab.bit2int(votes)            
        
        metrics = defaultdict(list)
        for i in range(len(votes)):            
            metrics_i = get_segment_metrics(votes_int[i],x_true)
            for k in metrics_i.keys():
                metrics[k].append(metrics_i[k])
        save_sample = self.opts.return_samples or bqi["save_grid"]
        if save_sample:
            self.samples.append({"pred_bit": votes,
                                "pred_int": votes_int,
                                "target_bit": x_true_bit,
                                "target_int": x_true,
                                "x_init": x_init,
                                "info": info,
                                "model_kwargs": model_kwargs})
        if self.opts.save_light_stats:
            has_raw_sample = self.opts.save_raw_samples and (bqi["sample"]<self.opts.num_save_raw_samples)
            light_stats = {"info": {k: v for k,v in info.items() if k in ["split_idx","i","dataset_name","num_classes"]},
                           "model_kwargs_abs_sum": {k: 
                                                    (v.abs().sum().item() if torch.is_tensor(v) else 0) 
                                                    for k,v in model_kwargs.items()},
                           "metrics": dict(metrics),
                           "has_raw_sample": has_raw_sample}
            self.light_stats.append(light_stats)
        return metrics
    
    def run_on_finished(self,output):
        self.opts.model_id = self.trainer.args.model_id
        self.opts.time = get_time()
        if self.trainer.args.mode!="gen":
            try:
                overwrite_existing_args(self.opts)
            except ValueError:
                save_args(self.opts)
        else:
            save_args(self.opts)
        if "grid" in self.opts.plotting_functions.split(",") and self.opts.num_grid_samples>0:
            assert self.opts.grid_filename.endswith(".png"), f"filename: {filename}"
            filename = self.opts.grid_filename
            max_images = min(self.opts.num_grid_samples,len(self.samples))
            plot_grid(filename,output,self.trainer.cgd.ab,max_images=max_images,remove_old=self.opts.remove_old,sample_names=output["info"],
                      imagenet_stats=self.trainer.args.crop_method.startswith("sam"))
        if "concat" in self.opts.plotting_functions.split(","):
            assert self.opts.concat_inter_filename.endswith(".png"), f"filename: {filename}"
            concat_inter_plots(foldername = self.opts.inter_folder,
                               concat_filename = self.opts.concat_inter_filename,
                               num_timesteps = len(self.save_i_steps),
                               remove_children="inter" not in self.opts.plotting_functions.split(","),
                               remove_old = self.opts.remove_old)
        if self.opts.save_light_stats:
            save_dict_list_to_json(self.light_stats,self.opts.light_stats_filename)

    def get_kwargs(self,batch):
        if self.opts.kwargs_mode=="train":
            assert self.trainer is not None, "self.trainer is None. Set self.trainer to a DiffusionModelTrainer instance or a class with a usable get_kwargs() method."
            x,model_kwargs,info = self.trainer.get_kwargs(batch, gen=True)
            if "points" in model_kwargs.keys():
                if model_kwargs["points"] is None:
                    model_kwargs["points"] = 0
                model_kwargs["points"] = model_kwargs["points"]*self.trainer.cgd.ab.int2bit(x)
        elif self.opts.kwargs_mode=="none":
            x,info = batch
            x = x.to(self.device)
            model_kwargs = {}
        elif self.opts.kwargs_mode=="only_image":
            x,info = batch
            x = x.to(self.device)
            x,model_kwargs,info = self.trainer.get_kwargs(batch, gen=True)
            model_kwargs = {k: v for k,v in model_kwargs.items() if k in ["image","image_features"]}
        return x,model_kwargs,info
            
    def form_next_batch(self):
        if self.queue is None:
            self.queue = []
            for i in range(self.opts.num_samples):
                for j in range(self.opts.num_votes):
                    save_inter_steps = (i<self.opts.num_inter_samples) and (j<self.opts.inter_votes_per_sample)
                    self.queue.append({"sample":i,"vote":j,"save_inter_steps": save_inter_steps, "save_grid": (i<self.opts.num_grid_samples)})
        
        bs = min(self.eval_batch_size,len(self.queue))
        if self.source_idx >= self.bss:
            self.source_batch = self.get_kwargs(next(self.dataloader))
            self.bss = self.source_batch[0].shape[0]
            self.source_idx = 0
        batch_x = []
        #use_kwargs = [k for k,v in self.source_batch[1].items() if v is not None]
        use_kwargs = list(self.source_batch[1].keys())
        batch_kwargs = {k: [] for k in use_kwargs}
        batch_info = []
        batch_queue = []
        for i in range(bs):
            batch_queue.append(self.queue.pop(0))
            batch_x.append(self.source_batch[0][self.source_idx])
            for k in use_kwargs:
                if self.source_batch[1][k] is None:
                    batch_kwargs[k].append(None)
                else:
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
                batch_kwargs[k] = unet_kwarg_to_tensor(batch_kwargs[k])
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