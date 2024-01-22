

import sys, os
sys.path.append(os.path.abspath('./source/'))
from source.utils import SmartParser
from source.training import DiffusionModelTrainer
from source.sampling import DiffusionSampler
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model_name to load")
    parser.add_argument("--gen_setups", type=str, default=None, help="generation setup to use")
    parser.add_argument("--num_timesteps", type=int, default=None, help="number of timesteps to sample")
    parser.add_argument("--num_samples", type=int, default=None, help="number of samples to generate")
    parser.add_argument("--num_grid_samples", type=int, default=None, help="number of grid samples to generate")
    parser.add_argument("--num_inter_samples", type=int, default=None, help="number of inter samples to generate")
    sample_args = parser.parse_args()
    """
    clip_denoised=True,
    num_timesteps=100, TODO
    num_samples=8, TODO
    guidance_weight=0.0, TODO
    guidance_kwargs='',
    num_votes=5, 
    eval_batch_size=0,
    progress_bar=True,
    sampler_type="ddpm",
    progress_bar_timestep=False,
    save_plot_grid_path=None,
    save_plot_inter_path=None,
    save_raw_samples_path=None,
    save_concat_plot_inter_path=None,
    save_raw_inter=False,
    num_inter_steps=10,
    num_inter_samples=8,
    inter_votes_per_sample=1,
    kwargs_mode="train",
    self_cond=False,
    return_metrics=True,
    return_samples=False,
    remove_old=False,
    do_agg=True,
    ema_model_rate=0,
    sample_function="grid,inter",
    split="vali"
    """

    alt_parse_args = ["--mode","cont","--model_name",sample_args.model_name,"--gen_setups",sample_args.gen_setups]
    args = SmartParser().get_args(do_parse_args=False,alt_parse_args=alt_parse_args)
    trainer = DiffusionModelTrainer(args)
    sampler = DiffusionSampler(trainer)
        
    trainer = DiffusionModelTrainer(args)
    trainer.train_loop()
    
if __name__ == "__main__":
    main()