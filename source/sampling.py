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
#from cont_gaussian_diffusion import DummyDiffusion

def get_default_sampler_options():
    dict_options = dict(clip_denoised=True,
                        num_timesteps=100,
                        num_samples=8,
                        guidance_weight=0.0,
                        num_votes=1, 
                        eval_batch_size=8, 
                        inter_steps=10,
                        inter_per_sample=1,
                        save_grid=False,
                        save_inter=False,
                        return_sample=False,
                        save_raw_samples_path=None,
                        return_raw_samples=False,
                        log=True,
                        progress_bar=True,
                        calc_full_stats=False,
                        sampler_type="ddpm",
                        progress_timestep=True,
                        return_inter=False,
                        save_text=True)
    return Namespace(**dict_options)

class DiffusionSampler(object):
    def __init__(self, diffusion, model, dataloader, output_folder=None, step=0, 
                 opts=get_default_sampler_options(),do_agg=True):
        super().__init__()
        self.cgd = diffusion
        self.is_dummy_diffusion = False#isinstance(diffusion,DummyDiffusion)
        self.model = model
        self.dataloader = dataloader
        self.opts = opts
        self.output_folder = output_folder
        self.step = step
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.log("WARNING: CUDA not available. Using CPU.")
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
        if self.output_folder is not None:
            os.makedirs(self.output_folder, exist_ok=True)
        self.opts = Namespace(**{**vars(self.opts),**kwargs})
        
        if self.output_folder is None:
            assert len(self.sample_function)==0,"output_folder must be specified if sample_function is not empty."
        
        if self.opts.save_raw_samples_path is not None:
            os.makedirs(self.opts.save_raw_samples_path,exist_ok=True)
        self.queue = None
        if self.opts.return_inter:
            self.inter_for_return = []
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
                x_pred = self.cgd.sample_loop(self.model, 
                                        x_init, 
                                        self.opts.num_timesteps, 
                                        self.opts.sampler_type,
                                        clip_x=self.opts.clip_denoised,
                                        model_kwargs=model_kwargs,
                                        guidance_weight=self.opts.guidance_weight)
                self.run_on_single_batch()
                for i in range(x_pred.shape[0]):
                    votes.append(x_pred[i])
                    if batch_queue[i]["vote"]==self.opts.num_votes-1:
                        metrics = self.run_on_full_votes(votes,x_init[i],x_true[i],x_true_bit[i],info[i])
                        votes = []
                        metric_list.append(metrics)

            metric_list = {k: [m[k] for m in metric_list] for k in metric_list[0].keys()}
            mean_metrics = {k: np.mean(metric_list[k]) for k in metric_list.keys()} 
            self.run_on_finished(metric_list)
            
        if self.opts.log:
            pass
        if self.opts.return_sample:
            samples = self.samples
        else:
            samples = None
        self.reset(restore_opts=True)
        if was_training:
            self.model.train()
            
        output = self.post_process(metric_list, mean_metrics, samples)
        return output
    
    def post_process(self,metric_list, mean_metrics, samples):
        output = {}
        for k in metric_list.keys():
            output[k] = metric_list[k]
        for k in samples[0]:
            output[k] = [s[k] for s in samples]
            if torch.is_tensor(output[k][0]):
                output[k] = torch.stack(output[k],dim=0)
        return output
            
    def run_on_single_batch(self):
        pass

    def run_on_full_votes(self,votes,x_init,x_true,x_true_bit,info):
        x_true = x_true.cpu()
        x_true_bit = x_true_bit.cpu()
        votes = torch.stack(votes,dim=0).cpu()
        votes_int = self.cgd.ab.bit2int(votes)
        self.samples.append({"pred_bit": votes,
                             "pred_int": votes_int,
                             "x_init": x_init.cpu(), 
                             "target_bit": x_true_bit,
                             "target_int": x_true, 
                             "info": info})
        metrics = defaultdict(list)
        for i in range(len(votes)):            
            metrics_i = get_segment_metrics(votes_int[i],x_true)
            for k in metrics_i.keys():
                metrics[k].append(metrics_i[k])
        return metrics
    
    def run_on_finished(self,metric_list):
        pass
    
    def get_kwargs(self,batch):
        x,info = batch
        x = x.to(self.device)
        model_kwargs = {}
        return x,model_kwargs,info
    
    def form_next_batch(self):
        if self.queue is None:
            self.queue = []
            for i in range(self.opts.num_samples):
                for j in range(self.opts.num_votes):
                    save_inter = False
                    self.queue.append({"sample":i,"vote":j,"save_inter": save_inter})

        bs = min(self.opts.eval_batch_size,len(self.queue))
        if self.source_idx >= self.bss:
            self.source_batch = self.get_kwargs(next(self.dataloader))
            self.bss = self.source_batch[0].shape[0]
            self.source_idx = 0
        batch_x = []
        batch_kwargs = defaultdict(list)
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
                                   output_folder=args.save_path,
                                   step=trainer.step,
                                   do_agg=False)
        sampler.opts.num_samples = 4
        output = sampler.sample(return_sample=True,num_timesteps=10)
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
        from plot_utils import plot_grid
        from utils import SmartParser
        from training import DiffusionModelTrainer
        
        args = SmartParser().get_args(do_parse_args=False)
        args.model_name = "test_trained"
        args.save_path = "./saves/test_trained/"
        trainer = DiffusionModelTrainer(args)
        sampler = DiffusionSampler(diffusion=trainer.cgd,
                                   model=trainer.model,
                                   dataloader=trainer.vali_dl,
                                   output_folder=args.save_path,
                                   step=trainer.step,
                                   do_agg=False)
        sampler.opts.num_samples = 2
        sampler.opts.num_votes = 3
        output = sampler.sample(return_sample=True,num_timesteps=10)
        filename = os.path.join(args.save_path,"test_123.png")
        plot_grid(filename,output,trainer.cgd.ab,max_images=32,remove_old=False)
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()