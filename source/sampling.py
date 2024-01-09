import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import get_segment_metrics
import jlc
from argparse import Namespace
import copy
from collections import defaultdict
import os
import tqdm
from plot_utils import plot_grid,plot_inter,concat_inter_plots
#from cont_gaussian_diffusion import DummyDiffusion

def get_default_sampler_options():
    dict_options = dict(clip_denoised=True,
                        num_timesteps=100,
                        num_samples=8,
                        guidance_weight=0.0,
                        num_votes=1, 
                        eval_batch_size=8,
                        progress_bar=True,
                        sampler_type="ddpm",
                        progress_bar_timestep=False,
                        save_plot_grid_path=None,
                        save_plot_inter_path=None,
                        save_raw_samples_path=None,#TODO
                        save_concat_plot_inter_path=None,
                        save_raw_inter=False,#TODO
                        num_inter_steps=10,
                        num_inter_samples=0,
                        inter_votes_per_sample=1,
                        )
    return Namespace(**dict_options)

class DiffusionSampler(object):
    def __init__(self, diffusion, model, dataloader, step=0, 
                 opts=get_default_sampler_options(),do_agg=True):
        super().__init__()
        self.cgd = diffusion
        self.is_dummy_diffusion = False#isinstance(diffusion,DummyDiffusion)
        self.model = model
        self.dataloader = dataloader
        self.opts = opts
        self.step = step
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("WARNING: CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
        if do_agg:
            matplotlib.use("agg")
        
    def reset(self,store_opts=False,restore_opts=False):
        self.source_batch = None
        self.queue = None
        if restore_opts:
            self.opts = self.original_opts
        if store_opts:
            self.original_opts = copy.copy(self.opts)
        self.samples = []
        self.source_idx = 0
        self.bss = 0

    def sample(self,**kwargs):
        if self.model.training:
            was_training = True
            self.model.eval()
        else:
            was_training = False
        self.reset(store_opts=True)
        self.opts = Namespace(**{**vars(self.opts),**kwargs})
            
        if self.opts.save_raw_samples_path is not None:
            os.makedirs(self.opts.save_raw_samples_path,exist_ok=True)
        self.queue = None
        metric_list = []
        votes = []
        num_batches = np.ceil(self.opts.num_samples*self.opts.num_votes/self.opts.eval_batch_size).astype(int)
        if num_batches==0:
            return [], {}, [], None
        if self.opts.progress_bar:
            progress_bar = tqdm.tqdm(range(num_batches), desc="Batch progress.")
        else:
            progress_bar = range(num_batches)
        with torch.no_grad():
            for batch_ite in progress_bar:
                x_true, model_kwargs, info, batch_queue = self.form_next_batch()
                x_true_bit = self.cgd.ab.int2bit(x_true)
                
                x_init = torch.randn_like(x_true_bit)
                sample_output = self.cgd.sample_loop(model=self.model, 
                                            x_init=x_init, 
                                            num_steps=self.opts.num_timesteps, 
                                            sampler_type=self.opts.sampler_type,
                                            clip_x=self.opts.clip_denoised,
                                            model_kwargs=model_kwargs,
                                            guidance_weight=self.opts.guidance_weight,
                                            progress_bar=self.opts.progress_bar_timestep,
                                            save_i_steps=self.save_i_steps,
                                            save_i_idx=[bq["save_inter_steps"] for bq in batch_queue]
                                            )
                self.run_on_single_batch(sample_output,batch_queue,x_init,x_true_bit,model_kwargs,batch_ite)
                for i in range(sample_output["pred"].shape[0]):
                    votes.append(sample_output["pred"][i])
                    if batch_queue[i]["vote"]==self.opts.num_votes-1:
                        model_kwargs_i = {k: model_kwargs[k][i] for k in model_kwargs.keys() if model_kwargs[k] is not None}
                        metrics = self.run_on_full_votes(votes,x_true[i],x_true_bit[i],info[i],model_kwargs_i,x_init[i])
                        votes = []
                        metric_list.append(metrics)
        
        metric_list = {k: [m[k] for m in metric_list] for k in metric_list[0].keys()}
        mean_metrics = {k: np.mean(metric_list[k]) for k in metric_list.keys()} 
        
        output = self.get_output_dict(metric_list, mean_metrics, self.samples)
        self.run_on_finished(metric_list,output)
        self.reset(restore_opts=True)
        if was_training:
            self.model.train()
        return output
    
    def get_output_dict(self,metric_list, mean_metrics, samples):
        output = {}
        for k in metric_list.keys():
            output[k] = metric_list[k]
        if samples is not None:
            for k in samples[0].keys():
                output[k] = [s[k] for s in samples]
                if torch.is_tensor(output[k][0]):
                    output[k] = torch.stack(output[k],dim=0)
        return output
            
    def run_on_single_batch(self,sample_output,bq,x_init,x_true_bit,model_kwargs,batch_ite):
        sample_output["x"] = x_true_bit      
        if self.opts.save_plot_inter_path is not None:
            save_i_idx = [bq_i["save_inter_steps"] for bq_i in bq]
            plot_inter(foldername=self.opts.save_plot_inter_path,
                       sample_output=sample_output,
                       model_kwargs=model_kwargs,
                       ab=self.cgd.ab,
                       save_i_idx=save_i_idx,
                       plot_text=self.opts.save_concat_plot_inter_path is None)
        if self.opts.save_raw_samples_path is not None:
            if not hasattr(self,"raw_samples"):
                self.raw_samples = []
            if not os.path.exists(self.opts.save_raw_samples_path):
                os.makedirs(self.opts.save_raw_samples_path)
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
        votes_int = self.cgd.ab.bit2int(votes)            
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
    
    def run_on_finished(self,metric_list,output):
        if self.opts.save_plot_grid_path is not None:
            assert self.opts.save_plot_grid_path.endswith(".png"), f"filename: {filename}"
            filename = self.opts.save_plot_grid_path
            plot_grid(filename,output,self.cgd.ab,max_images=32,remove_old=True)
        if self.opts.save_concat_plot_inter_path is not None:
            assert self.opts.save_concat_plot_inter_path.endswith(".png"), f"filename: {filename}"
            concat_inter_plots(foldername = self.opts.save_plot_inter_path,
                               concat_filename = self.opts.save_concat_plot_inter_path,
                               num_timesteps = self.opts.num_timesteps)
    def get_kwargs(self,batch):
        x,info = batch
        x = x.to(self.device)
        model_kwargs = {}
        return x,model_kwargs,info
            
    def form_next_batch(self):
        if self.queue is None:
            if (0<self.opts.num_inter_samples) and (0<self.opts.inter_votes_per_sample):
                if (not self.opts.save_raw_inter) and (not self.opts.save_plot_inter_path):
                    raise ValueError("self.opts.save_raw_inter and self.opts.save_plot_inter_path are both False. No need to compute intermediate steps.")
                legal_timesteps = list(range(self.opts.num_timesteps-1, -1, -1))
                idx = np.round(np.linspace(0,len(legal_timesteps)-1,self.opts.num_inter_steps)).astype(int)
                #remove duplicates without changing order
                idx = [idx[i] for i in range(len(idx)) if (i==0) or (idx[i]!=idx[i-1])]
                self.save_i_steps = [legal_timesteps[i] for i in idx]
            self.queue = []
            for i in range(self.opts.num_samples):
                for j in range(self.opts.num_votes):
                    save_inter_steps = (i<self.opts.num_inter_samples) and (j<self.opts.inter_votes_per_sample)
                    self.queue.append({"sample":i,"vote":j,"save_inter_steps": save_inter_steps})
        
        bs = min(self.opts.eval_batch_size,len(self.queue))
        if self.source_idx >= self.bss:
            self.source_batch = self.get_kwargs(next(self.dataloader))
            self.bss = self.source_batch[0].shape[0]
            self.source_idx = 0
        batch_x = []
        batch_kwargs = {}
        batch_info = []
        batch_queue = []
        for i in range(bs):
            batch_queue.append(self.queue.pop(0))
            batch_x.append(self.source_batch[0][self.source_idx])
            for k in self.source_batch[1].keys():
                batch_kwargs[k].append
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
        print("UNIT TEST 0: generate images and show them")
        from utils import SmartParser
        from training import DiffusionModelTrainer
        
        args = SmartParser().get_args(do_parse_args=False)
        args.model_name = "test_trained"
        args.save_path = "./saves/test_trained/"
        trainer = DiffusionModelTrainer(args)
        jlc.num_of_params(trainer.model)
        sampler = DiffusionSampler(diffusion=trainer.cgd,
                                   model=trainer.model,
                                   dataloader=trainer.vali_dl,
                                   step=trainer.step,
                                   do_agg=False)
        sampler.opts.num_samples = 4
        output = sampler.sample(num_timesteps=10)
        images = []
        key_show = "pred_bit"
        im = output[key_show]
        if len(im.shape)==3:
            im = im.permute(1,2,0).numpy()
            images.append(im)
        elif len(im.shape)==4:
            im = im.permute(0,2,3,1).numpy()
            images.extend([im[i] for i in range(im.shape[0])])
        else:
            assert len(im.shape)==5, f"im.shape: {im.shape}"
            #collapse first 2 dims
            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    images.append(im[i,j].permute(1,2,0).numpy())
        jlc.montage(images,return_im=True)
        plt.show()
    if args.unit_test==1:
        print("UNIT TEST 1: plot grid of generated images")
        from utils import SmartParser
        from training import DiffusionModelTrainer
        
        args = SmartParser().get_args(do_parse_args=False)
        args.model_name = "test_trained"
        args.save_path = "./saves/test_trained/"
        trainer = DiffusionModelTrainer(args)
        sampler = DiffusionSampler(diffusion=trainer.cgd,
                                   model=trainer.model,
                                   dataloader=trainer.vali_dl,
                                   step=trainer.step,
                                   do_agg=False)
        sampler.opts.num_samples = 2
        sampler.opts.num_votes = 3
        output = sampler.sample(num_timesteps=10)
        filename = os.path.join(args.save_path,"test_123.png")
        plot_grid(filename,output,trainer.cgd.ab,max_images=32,remove_old=False)
    if args.unit_test==2:
        print("UNIT TEST 2: plot intermediate generated images")
        from utils import SmartParser
        from training import DiffusionModelTrainer
        
        args = SmartParser().get_args(do_parse_args=False)
        args.model_name = "test_trained"
        args.save_path = "./saves/test_trained/"
        trainer = DiffusionModelTrainer(args)
        sampler = DiffusionSampler(diffusion=trainer.cgd,
                                   model=trainer.model,
                                   dataloader=trainer.vali_dl,
                                   step=trainer.step,
                                   do_agg=False)
        sampler.opts.num_samples = 2
        sampler.opts.num_votes = 3
        sampler.opts.num_inter_samples = 2
        sampler.opts.num_inter_steps = 10
        sampler.opts.save_plot_inter_path = os.path.join(args.save_path,"inter")
        
        output = sampler.sample(num_timesteps=10)
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()