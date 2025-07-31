import torch
import numpy as np
import matplotlib
from argparse import Namespace
from collections import defaultdict
import os
import warnings
import tqdm
from source.models.unet import all_input_keys, get_sam_image_encoder
from source.utils.mixed import (get_time,save_dict_list_to_json,
                   check_keys_are_same,mask_from_imshape,postprocess_batch,
                   sam_resize_index,apply_mask,unet_kwarg_to_tensor,construct_points,
                   model_arg_is_trivial,nice_split,load_json_to_dict_list,
                   ambiguous_info_from_fn,set_random_seed)
from source.utils.metric_and_loss import get_segment_metrics, get_ambiguous_metrics
from source.utils.dataloading import get_dataset_from_args
from source.utils.plot import plot_grid,plot_inter,concat_inter_plots,index_dict_with_bool
from source.utils.argparsing import TieredParser, save_args, overwrite_existing_args, str2bool
from source.utils.analog_bits import AnalogBits
from pathlib import Path
import copy
#from cont_gaussian_diffusion import DummyDiffusion TODO

import random

class DiffusionSampler(object):
    def __init__(self, trainer, opts=None,
                 ):
        super().__init__()
        self.is_dummy_diffusion = False#isinstance(diffusion,DummyDiffusion) TODO
        if opts is None:
            opts = TieredParser("sample_opts").get_args([])
        self.opts = opts
        self.trainer = trainer
        self.args = copy.deepcopy(self.trainer.args)
        self.ab = AnalogBits(self.args)
        self.opts.seed = self.args.seed
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
        self.eval_batch_size = self.opts.eval_batch_size if self.opts.eval_batch_size>0 else self.args.train_batch_size
        if self.opts.compute_full_ap:
            self.full_infos_for_ap = []
        if len(self.opts.split_method)>0:
            assert self.opts.split_method in ["random","native_train","native"], f"split_method={self.opts.split_method} is not a valid option."
            if self.args.mode!="gen" and not self.opts.ambiguous_mode:#, "split_method can only be specified in sampling mode."
                assert self.args.split_method == self.opts.split_method, f"The training split_method={self.args.split_method} is different from the sampling split_method={self.opts.split_method}."
            else:
                self.args.split_method = self.opts.split_method
                        
        if self.opts.ambiguous_mode:
            assert len(self.opts.datasets.split(","))==1, "Ambiguous mode is only implemented for a single specified dataset. Found: datasets="+str(self.opts.datasets)
        if self.args.mode=="gen" or self.opts.num_workers>=0:
            self.args_restore = copy.deepcopy(self.args)
            if self.opts.num_workers>=0:
                self.args.dl_num_workers = self.opts.num_workers
            else:
                self.args.dl_num_workers = 0
            if len(self.opts.datasets)>0:
                if not isinstance(self.opts.datasets,list):
                    self.opts.datasets = self.opts.datasets.split(",")
                self.args.datasets = self.opts.datasets
            
            if isinstance(self.opts.aug_override,str):
                if self.opts.aug_override.lower()=="none":
                    aug_override = None
                else:
                    aug_override = str2bool(self.opts.aug_override) 
            else:
                aug_override = str2bool(self.opts.aug_override) 

            if self.opts.ambiguous_mode:
                assert len(self.opts.datasets)==1, "Ambiguous mode is only implemented for a single dataset."
                d = self.opts.datasets[0]
                assert d.startswith("lidc"), "Ambiguous mode is only implemented for the lidc dataset."
                assert aug_override is None, "aug_override is not implemented for ambiguous mode."
                #info_path = f"/home/jloch/Desktop/diff/diffusion2/data/{d}/info.jsonl"
                #with relative instead
                info_path = os.path.join(Path(__file__).resolve().parent.parent,"data",d,"info.jsonl")
                info = load_json_to_dict_list(info_path)
                pri_didx = []
                gts_didx = {}
                split2idx = {"train":0,"vali":1,"test":2}
                for i,info_i in enumerate(info):
                    if info_i["split_idx"]==split2idx.get(self.opts.split,self.opts.split):
                        fn = info_i["fn"]
                        amb_info = ambiguous_info_from_fn(fn)
                        if amb_info["m_i"]==0:
                            didx = f"{d}/{info_i['i']}"
                            pri_didx.append(didx)
                            image_ids = [ambiguous_info_from_fn(info_j["fn"])["image_id"] for info_j in info[i:i+amb_info["m_tot"]]]
                            assert len(set(image_ids))==1, f"Found multiple image_ids, but expected only one for fn={fn}. image_ids: {image_ids}"
                            gts_didx[didx] = [f"{d}/{info_j['i']}" for info_j in info[i:i+amb_info["m_tot"]]]
                assert len(pri_didx)>0, f"Found no ambiguous samples in the dataset. split={self.opts.split}"
                #shuffle the list based on the seed
                random.seed(self.opts.seed)
                random.shuffle(pri_didx)
                #set_random_seed(self.opts.seed)
                max_num_samples = len(pri_didx)
                assert self.opts.num_samples < max_num_samples, f"Founds {max_num_samples} ambiguous samples in the dataset. num_samples must be less than this. -1 for the maximum possible"
                assert len(self.opts.pri_didx)==0, f"pri_didx is not implemented for ambiguous mode. Found: {self.opts.pri_didx}"
                if self.opts.num_samples<0:
                    self.opts.num_samples = max_num_samples
                self.gts_didx = gts_didx
                if self.opts.num_samples==64 and self.args.crop_method=="sam_lidc64" and (self.opts.split in ["vali","test"]):
                    self.args.crop_method = "sam_lidc64_fixed64"
                    print("USING sam_lidc64_fixed64")
                self.dataloader = get_dataset_from_args(self.args,
                                                    self.opts.split,
                                                    ambiguous_mode=self.opts.ambiguous_mode,
                                                    mode="pri_didx",
                                                    prioritized_didx=pri_didx)
                
            else:
                self.gts_didx = None
                if len(self.opts.pri_didx)>0:
                    assert isinstance(self.opts.pri_didx,str), f"pri_didx must be a comma-separated string. Found: {self.opts.pri_didx}"
                    if "," not in self.opts.pri_didx:
                        #load from the didx file 
                        named_didx = load_json_to_dict_list("/home/jloch/Desktop/diff/diffusion2/jsons/pri_didx.json")
                        assert self.opts.pri_didx in named_didx.keys(), f"pri_didx={self.opts.pri_didx} not found in the didx file. When no comma is in the string, it is assumed its a key in the didx file."
                        pri_didx = named_didx[self.opts.pri_didx]
                    else:
                        pri_didx = nice_split(self.opts.pri_didx)
                    self.opts.num_samples = len(pri_didx)
                    self.dataloader = get_dataset_from_args(self.args,
                                                        split=self.opts.split,
                                                        prioritized_didx=pri_didx,
                                                        mode="pri_didx",
                                                        aug_override=aug_override)
                else:
                    pri_didx = None
                    self.dataloader = get_dataset_from_args(self.args,
                                                        split=self.opts.split,
                                                        mode="pure_gen",
                                                        aug_override=aug_override)
        else:
            assert hasattr(self.trainer,f"{self.opts.split}_dl"), f"trainer does not have a dataloader for split={self.opts.split}."
            self.dataloader = getattr(self.trainer,f"{self.opts.split}_dl")
        if self.args.image_encoder!="none":
            if hasattr(self.trainer,"image_encoder"):
                pass
            else:
                if self.dataloader.dataloader.dataset.all_samples_have_sfi:
                    self.trainer.image_encoder = None
                else:
                    self.trainer.image_encoder = get_sam_image_encoder(self.args.image_encoder,device=self.device)

        if not self.opts.ambiguous_mode:
            lpd = self.dataloader.dataloader.dataset.len_per_dataset
            datasets = self.args.datasets if isinstance(self.args.datasets,list) else [self.args.datasets]
            max_num_samples = sum([lpd[dataset] for dataset in datasets])
            if self.opts.num_samples<0:
                self.opts.num_samples = max_num_samples
            elif self.opts.num_samples>max_num_samples:
                print(f"WARNING: num_samples={self.opts.num_samples} is larger than the maximum number of samples in the specified datasets: {max_num_samples}. Setting num_samples to the maximum.")
                self.opts.num_samples = max_num_samples

        if model is None:
            if self.opts.ema_idx>=0:
                model, self.swap_pointers_func = self.trainer.get_ema_model(self.opts.ema_idx)
            else:
                model = self.trainer.model
        was_training = model.training
        model.eval()
        #print first 10 params of model

        if self.opts.do_agg:
            old_backend = matplotlib.get_backend()
            matplotlib.use("agg")
        else:
            old_backend = None
        if self.opts.semantic_prob>=0:
            self.semantic_prob_old = self.dataloader.dataloader.dataset.semantic_prob
            self.dataloader.dataloader.dataset.semantic_prob = self.opts.semantic_prob
        if self.opts.default_save_folder=="":
            self.opts.default_save_folder = os.path.join(self.args.save_path,"samples")
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
            assert self.opts.num_samples>=self.opts.num_grid_samples, f"num_samples must be at least as large as num_grid_samples. Found: num_samples={self.opts.num_samples}, num_grid_samples={self.opts.num_grid_samples}"
        assert self.opts.num_votes>0, "num_votes must be positive."
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
        entropy = []
        num_batches = np.ceil(self.opts.num_samples*self.opts.num_votes/self.eval_batch_size).astype(int)
        if num_batches==0:
            warnings.warn("num_batches==0.")
            return None
        if self.opts.progress_bar:
            progress_bar = tqdm.tqdm(range(num_batches), desc="Batch progress.")
        else:
            progress_bar = range(num_batches)
        with torch.no_grad():
            for batch_ite in progress_bar:
                gt_int, model_kwargs, info, batch_queue = self.form_next_batch()
                gt_bit = self.ab.int2bit(gt_int)
                print(gt_int[4,:,-1,48])
                print(gt_bit[4,:,-1,48])
                #exit()
                x_init = torch.randn_like(gt_bit)
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
                                            save_entropy=self.opts.save_entropy,
                                            replace_padding=self.opts.replace_padding,
                                            imshape=[info_i["imshape"] for info_i in info],
                                            )
                for k in model_kwargs.keys():
                    model_kwargs[k] = unet_kwarg_to_tensor(model_kwargs[k],key=k)
                self.run_on_single_batch(sample_output,batch_queue,x_init,gt_bit,model_kwargs,batch_ite,info)
                for i in range(sample_output["pred_bit"].shape[0]):
                    votes.append(sample_output["pred_bit"][i])
                    if "entropy" in sample_output.keys():
                        entropy.append(sample_output["entropy"][i])
                    if batch_queue[i]["vote"]==self.opts.num_votes-1:
                        model_kwargs_i = {k: 
                                          (model_kwargs[k][i] if model_kwargs[k] is not None else None) 
                                          for k in model_kwargs.keys()}
                        metrics = self.run_on_full_votes(votes,gt_int[i],gt_bit[i],info[i],model_kwargs_i,x_init[i],batch_queue[i],entropy)
                        votes = []
                        entropy = []
                        metric_list.append(metrics)

        sample_output, metric_output = self.get_output_dict(metric_list, self.samples)
        self.run_on_finished(output={**sample_output,**metric_output})

        if old_backend is not None:
            matplotlib.use(old_backend)
        if hasattr(self,"args_restore"):
            self.args = copy.deepcopy(self.args_restore)
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

    def get_output_dict(self, metric_list, samples, info_keys_save=["dataset_name","i","gts_didx"]):
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
                    #sample_output[k] = [{sub_k: s[k][sub_k] for sub_k in info_keys_save} for s in samples]
                    sample_output[k] = [{sub_k: s[k][sub_k] for sub_k in s[k].keys() if sub_k in info_keys_save} for s in samples]
                    continue
                if isinstance(samples[0][k],dict):
                    assert check_keys_are_same([s[k] for s in samples]), f"Key conflict for key={k}"
                    for sub_k in samples[0][k].keys():                        
                        sample_output[sub_k] = unet_kwarg_to_tensor([s[k][sub_k] for s in samples],key=sub_k)
                elif torch.is_tensor(samples[0][k]) or isinstance(samples[0][k],list):
                    sample_output[k] = unet_kwarg_to_tensor([s[k] for s in samples],key=k)
        return sample_output, metric_output
            
    def run_on_single_batch(self,sample_output,bq,x_init,gt_bit,model_kwargs,batch_ite,info):
        sample_output = copy.deepcopy(sample_output)
        model_kwargs = copy.deepcopy(model_kwargs)
        info = copy.deepcopy(info)
        sample_output["gt_bit"] = gt_bit
        if self.opts.inter_folder!="":
            save_i_idx = [bq_i["save_inter_steps"] for bq_i in bq]
            plot_inter(foldername=self.opts.inter_folder,
                       sample_output=sample_output,
                       model_kwargs=model_kwargs,
                       save_i_idx=save_i_idx,
                       plot_text=self.opts.concat_inter_filename=="",
                       imagenet_stats=self.args.crop_method.startswith("sam"),
                       ab=self.ab)
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
                if self.opts.only_save_raw_pred:
                    sample_output = {"info": [{"dataset_name": info_i["dataset_name"], "i": info_i["i"]} for info_i in info],
                                     "pred_int": self.ab.bit2int(sample_output["pred_bit"],info=info),}
                torch.save(sample_output,os.path.join(self.opts.raw_samples_folder,f"raw_sample_batch{batch_ite:03d}.pt"))
            
    def run_on_full_votes(self,votes,gt_int,gt_bit,info,model_kwargs,x_init,bqi,entropy):
        gt_int = gt_int.cpu()
        gt_bit = gt_bit.cpu()
        votes = torch.stack(votes,dim=0).cpu()
        votes_int = self.ab.bit2int(votes,[info]*votes.shape[0])
        if self.opts.postprocess!="none":
            if self.opts.postprocess.startswith("min_area"):
                area = float(self.opts.postprocess.split("min_area")[-1])
                votes_int = postprocess_batch(votes_int,seg_kwargs={"mode": "min_area", "min_area": area},list_of_imshape=[info["imshape"][:2]]*votes.shape[0])
            elif self.opts.postprocess=="rel_area0.5":
                votes_int = postprocess_batch(votes_int,seg_kwargs={"mode": "min_rel_area", "min_area": 0.5},list_of_imshape=[info["imshape"][:2]]*votes.shape[0])
            else:
                raise ValueError(f"postprocess={self.opts.postprocess} is not a valid option.")
        if self.opts.ambiguous_mode:
            info["gts_didx"] = self.gts_didx[f"{info['dataset_name']}/{info['i']}"]
        imsize = gt_int.shape[-1]
        mask = torch.from_numpy(mask_from_imshape(info["imshape"],imsize,num_dims=3)).to(self.device)
        ign0 = not self.args.agnostic
        if self.opts.ambiguous_mode:
            metrics = get_ambiguous_metrics(apply_mask(votes_int,info["imshape"]).squeeze(1).permute(1,2,0).cpu().numpy(),info,
                                            reduce_to_mean=True)
            if self.opts.add_amb_postprocess_metrics:
                areas = [votes_int[i].float().mean().item() for i in range(votes_int.shape[0])]
                max_area = max(areas)
                votes_int_pp = torch.stack([v_i if a>0.5*max_area else torch.zeros_like(v_i) for v_i,a in zip(votes_int,areas)])
                metrics_pp = get_ambiguous_metrics(apply_mask(votes_int_pp,info["imshape"]).squeeze(1).permute(1,2,0).cpu().numpy(),info,
                                            reduce_to_mean=True)
                metrics.update({f"{k}_pp": v for k,v in metrics_pp.items()})
        else:
            metrics = []
            for i in range(len(votes)):
                if self.opts.pure_eval_mode:
                    metrics_i = get_segment_metrics(apply_mask(votes_int[i],info["imshape"]),info,ignore_zero=ign0)
                else:
                    metrics_i = get_segment_metrics(votes_int[i],gt_int,mask=mask,ignore_zero=ign0)
                metrics.append(metrics_i)
            metrics = {k: [m[k] for m in metrics] for k in metrics[0].keys()}
        if len(entropy)>0:
            metrics["entropy"] = entropy
        save_sample = self.opts.return_samples or bqi["save_grid"]
        if save_sample:
            self.samples.append({"pred_bit": votes,
                                "pred_int": votes_int,
                                "gt_bit": gt_bit,
                                "gt_int": gt_int,
                                "x_init": x_init,
                                "info": copy.deepcopy(info),
                                "model_kwargs": model_kwargs})
        if self.opts.compute_full_ap:
            pass
            #self.full_infos_for_ap.append(ap_entity()[1]["full_info"])
        if self.opts.save_light_stats:
            has_raw_sample = self.opts.save_raw_samples and (bqi["sample"]<self.opts.num_save_raw_samples)
            light_stats = {"info": {k: v for k,v in info.items() if k in ["split_idx","i","dataset_name","num_classes","imshape","gts_didx"]},
                           "model_kwargs_abs_sum": {k: 
                                                    (v.abs().sum().item() if torch.is_tensor(v) else 0) 
                                                    for k,v in model_kwargs.items()},
                           "metrics": metrics,
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
        self.opts.model_id = self.args.model_id
        self.opts.time = get_time()
        if self.args.mode!="gen":
            try:
                overwrite_existing_args(self.opts)
            except:
                if not Path(self.opts.default_save_folder).exists():
                    Path(self.opts.default_save_folder).mkdir(parents=False, exist_ok=True)
                save_args(self.opts)
        else:
            save_args(self.opts)
        if "grid" in self.opts.plotting_functions.split(",") and self.opts.num_grid_samples>0:
            assert self.opts.grid_filename.endswith(".png"), f"filename: {filename}"
            filename = self.opts.grid_filename
            max_images = min(self.opts.num_grid_samples,len(self.samples))
            plot_grid(filename,output,max_images=max_images,remove_old=self.opts.remove_old,sample_names=output["info"],
                      imagenet_stats=self.args.crop_method.startswith("sam"),ab=self.ab)
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
            gt_int,model_kwargs,info = self.trainer.get_kwargs(next(self.dataloader),
                                                          force_image=self.opts.kwargs_mode=="train_image")
        else:
            if self.args.mode=="gen" or self.args.dl_num_workers==0:
                self.dataloader.dataloader.dataset.gen_mode = True #enables all dynamic cond inputs
            gt_int,model_kwargs,info = self.trainer.get_kwargs(next(self.dataloader), gen=True)
            if self.args.mode=="gen" or self.args.dl_num_workers==0:
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
        if "num_labels" in model_kwargs.keys():
            assert isinstance(model_kwargs["num_labels"],list), f"Labels must be a list. Found: {model_kwargs['num_labels']}"
            for i in range(len(model_kwargs["num_labels"])):
                if self.opts.cond_num_labels>0:
                    model_kwargs["num_labels"][i].fill_(self.opts.cond_num_labels)
        if "points" in model_kwargs.keys():
            model_kwargs["points"] = construct_points(model_kwargs["points"],self.ab.int2bit(gt_int),as_tensor=False)
        return gt_int,model_kwargs,info
            
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