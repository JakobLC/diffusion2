import torch
import numpy as np
import matplotlib
from argparse import Namespace
from collections import defaultdict
import os
import tqdm
from source.utils.mixed_utils import (get_time,save_dict_list_to_json,
                   check_keys_are_same,mask_from_imshape,postprocess_batch,
                   sam_resize_index,apply_mask,unet_kwarg_to_tensor,construct_points,
                   model_arg_is_trivial,nice_split)
from source.utils.metric_and_loss_utils import get_segment_metrics
from source.utils.plot_utils import plot_grid,plot_inter,concat_inter_plots,index_dict_with_bool
from source.utils.argparse_utils import TieredParser, save_args, overwrite_existing_args
from source.models.cond_vit import all_input_keys
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
        self.eval_batch_size = self.opts.eval_batch_size if self.opts.eval_batch_size>0 else self.trainer.args.train_batch_size
        if len(self.opts.split_method)>0:
            assert self.trainer.args.mode=="gen", "split_method can only be specified in sampling mode."
            assert self.opts.split_method in ["random","native_train","native"], f"split_method={self.opts.split_method} is not a valid option."
            self.trainer.args.split_method = self.opts.split_method
        if len(self.opts.datasets)>0:
            if not isinstance(self.opts.datasets,list):
                self.opts.datasets = self.opts.datasets.split(",")
        if len(self.opts.datasets)>0:
            assert self.trainer.args.mode=="gen", "Datasets can only be specified in sampling mode."
            self.trainer.args.datasets = self.opts.datasets
            self.trainer.args.dl_num_workers = 0
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
        if self.opts.semantic_prob>=0:
            self.semantic_prob_old = self.dataloader.dataloader.dataset.semantic_prob
            self.dataloader.dataloader.dataset.semantic_prob = self.opts.semantic_prob
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
        assert self.opts.postprocess in ['none','area0.005'], f"postprocess={self.opts.postprocess} is not a valid option."

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
                                            guidance_kwargs=self.opts.guidance_kwargs,
                                            )
                for k in model_kwargs.keys():
                    model_kwargs[k] = unet_kwarg_to_tensor(model_kwargs[k],key=k)
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
        if hasattr(self,"semantic_prob_old"):
            self.dataloader.dataloader.dataset.semantic_prob = self.semantic_prob_old
            del self.semantic_prob_old
        if not self.opts.return_samples:
            sample_output = None
        return sample_output, metric_output

    def get_output_dict(self, metric_list, samples, info_keys_save=["dataset_name","i"]):
        model_kwargs_keys = []
        for s in samples:
            for k in s["model_kwargs"].keys():
                if k not in model_kwargs_keys:
                    model_kwargs_keys.append(k)
        for k in model_kwargs_keys:
            for i in range(len(samples)):
                if k not in samples[i]["model_kwargs"].keys():
                    samples[i]["model_kwargs"][k] = None

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
                    assert check_keys_are_same([s[k] for s in samples]), f"Key conflict for key={k}"
                    for sub_k in samples[0][k].keys():                        
                        sample_output[sub_k] = unet_kwarg_to_tensor([s[k][sub_k] for s in samples],key=sub_k)
                elif torch.is_tensor(samples[0][k]) or isinstance(samples[0][k],list):
                    sample_output[k] = unet_kwarg_to_tensor([s[k] for s in samples],key=k)
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
                if self.opts.only_save_raw_pred_int:
                    sample_output = {"info": sample_output["info"],
                                     "pred_int": self.trainer.cgd.ab.bit2int(sample_output["pred"])}
                torch.save(sample_output,os.path.join(self.opts.raw_samples_folder,f"raw_sample_batch{batch_ite:03d}.pt"))
            
    def run_on_full_votes(self,votes,x_true,x_true_bit,info,model_kwargs,x_init,bqi):
        x_true = x_true.cpu()
        x_true_bit = x_true_bit.cpu()
        votes = torch.stack(votes,dim=0).cpu()
        votes_int = self.trainer.cgd.ab.bit2int(votes)
        if self.opts.postprocess!="none":
            if self.opts.postprocess=="area0.005":
                votes_int = postprocess_batch(votes_int,seg_kwargs={"mode": "min_area", "min_area": 0.005},list_of_imshape=[info["imshape"][:2]]*votes.shape[0])
            else:
                raise ValueError(f"postprocess={self.opts.postprocess} is not a valid option.")
        
        imsize = x_true.shape[-1]
        metrics = defaultdict(list)
        mask = torch.from_numpy(mask_from_imshape(info["imshape"],imsize,num_dims=3)).to(self.device)
        for i in range(len(votes)):
            if self.opts.pure_eval_mode:
                metrics_i = get_segment_metrics(apply_mask(votes_int[i],info["imshape"]),info)
            else:
                metrics_i = get_segment_metrics(votes_int[i],x_true,mask=mask)
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
            light_stats = {"info": {k: v for k,v in info.items() if k in ["split_idx","i","dataset_name","num_classes","imshape"]},
                           "model_kwargs_abs_sum": {k: 
                                                    (v.abs().sum().item() if torch.is_tensor(v) else 0) 
                                                    for k,v in model_kwargs.items()},
                           "metrics": dict(metrics),
                           "has_raw_sample": has_raw_sample}
            for k,v in model_kwargs.items():
                if not torch.is_tensor(v):
                    import jlc
                    print("model_kwargs for k="+k)
                    jlc.shaprint(v)
                    raise ValueError(f"model_kwargs[{k}] is not a tensor.")

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

    def sampler_get_kwargs(self):
        if self.opts.kwargs_mode in ["train","train_image"]:
            assert self.trainer is not None, "self.trainer is None. Set self.trainer to a DiffusionModelTrainer instance or a class with a usable get_kwargs() method."
            x,model_kwargs,info = self.trainer.get_kwargs(next(self.dataloader),
                                                          force_image=self.opts.kwargs_mode=="train_image")
        else:
            if self.trainer.args.mode=="gen" or self.trainer.args.dl_num_workers==0:
                self.dataloader.dataloader.dataset.gen_mode = True #enables all dynamic cond inputs
            x,model_kwargs,info = self.trainer.get_kwargs(next(self.dataloader), gen=True)
            if self.trainer.args.mode=="gen" or self.trainer.args.dl_num_workers==0:
                self.dataloader.dataloader.dataset.gen_mode = False

            model_kwargs_use = []
            if self.opts.use_image:
                model_kwargs_use.append("image")
                if "image_features" in model_kwargs.keys():
                    model_kwargs_use.append("image_features")

            do_nothing_kwargs_modes = ["none","only_image","image",""]
            special_kwargs_modes = ["all","train","train_image"]+do_nothing_kwargs_modes
            if self.opts.kwargs_mode in do_nothing_kwargs_modes:
                pass
            elif self.opts.kwargs_mode=="all":
                model_kwargs_use.extend(all_input_keys)
            else:
                for k in nice_split(self.opts.kwargs_mode):
                    if k not in all_input_keys:
                        raise ValueError(f"If kwargs_mode is NOT a special value ({special_kwargs_modes})"
                                         f"then it must be a comma-separated list of valid keys from all_input_keys." 
                                         f"Found: {k}. all_input_keys: {all_input_keys}")
                model_kwargs_use.extend(nice_split(self.opts.kwargs_mode))

            if not all([k in model_kwargs.keys() for k in model_kwargs_use]):
                not_found_kwargs = [k for k in model_kwargs_use if k not in model_kwargs.keys()]
                raise ValueError(f"Could not find the following requested kwargs from the dataloader: {not_found_kwargs}")
            model_kwargs = {k: model_kwargs[k] for k in model_kwargs_use}
        if "points" in model_kwargs.keys():
            model_kwargs["points"] = construct_points(model_kwargs["points"],self.trainer.cgd.ab.int2bit(x),as_tensor=False)
            if self.trainer.cgd.ab.shuffle:
                raise NotImplementedError("Shuffling of points is not implemented for sampling with points (since gt will be unmatched to the points).")
        return x,model_kwargs,info
            
    def form_next_batch(self):
        if self.queue is None:
            self.queue = []
            for i in range(self.opts.num_samples):
                for j in range(self.opts.num_votes):
                    save_inter_steps = (i<self.opts.num_inter_samples) and (j<self.opts.inter_votes_per_sample)
                    self.queue.append({"sample":i,
                                       "vote":j,
                                       "save_inter_steps": save_inter_steps, 
                                       "save_grid": (i<self.opts.num_grid_samples)})
        
        bs = min(self.eval_batch_size,len(self.queue))
        if self.source_idx >= self.bss:
            self.source_batch = self.sampler_get_kwargs()
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
                    self.source_batch = self.sampler_get_kwargs()
                    self.bss = self.source_batch[0].shape[0]
                    self.source_idx = 0

        for k in list(batch_kwargs.keys()):
            batch_kwargs[k] = unet_kwarg_to_tensor(batch_kwargs[k],key=k,list_instead=True)
        
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