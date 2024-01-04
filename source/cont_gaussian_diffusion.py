#Continuous Guassian Diffusion implemented by Jakob Loenborg Christensen (JakobLC github) jloch@dtu.dk

import enum
import numpy as np
import torch
from datasets import AnalogBits
from utils import normal_kl,mse_loss
import tqdm

def add_(coefs,x,batch_dim=0,flat=False):
    """broadcast and add coefs to x"""
    if isinstance(coefs,np.ndarray):
        coefs = torch.from_numpy(coefs)
    else:
        assert torch.is_tensor(coefs)
    assert torch.is_tensor(x)
    if flat:
        not_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
        return (coefs_(coefs, x.shape, dtype=x.dtype, device=x.device)+x).mean(not_batch_dims)
    else:
        return coefs_(coefs, x.shape, dtype=x.dtype, device=x.device)+x
    
def mult_(coefs,x,batch_dim=0,flat=False):
    """broacast and multiply coefs with x"""
    if isinstance(coefs,np.ndarray):
        coefs = torch.from_numpy(coefs)
    else:
        assert torch.is_tensor(coefs)
    assert torch.is_tensor(x)
    if flat:
        not_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
        return (coefs_(coefs, x.shape, dtype=x.dtype, device=x.device)*x).mean(not_batch_dims)
    else:
        return coefs_(coefs, x.shape, dtype=x.dtype, device=x.device)*x
    
def coefs_(coefs, shape, dtype=torch.float32, device="cuda", batch_dim=0):
    view_shape = [1 for _ in range(len(shape))]
    view_shape[batch_dim] = -1
    return coefs.view(view_shape).to(device).to(dtype)

def get_named_gamma_schedule(schedule_name,b,clip_min=1e-9):
    float64 = lambda x: torch.tensor(float(x),dtype=torch.float64)
    if schedule_name=="linear":
        gamma = lambda t: torch.sigmoid(-torch.log(torch.expm1(1e-4+10*t*t)))
    elif schedule_name.startswith("cosine"):
        num_params = len(schedule_name.split("_"))-1
        #format: cosine_start_end_tau
        tau = float64(1) if num_params<3 else float64(schedule_name.split("_")[3])
        end = float64(0) if num_params<2 else float64(schedule_name.split("_")[2])
        start = float64(1) if num_params<1 else float64(schedule_name.split("_")[1])
        def gamma(t):
            v_start = torch.cos(start * torch.pi / 2) ** (2 * tau)
            v_end = torch.cos(end * torch.pi / 2) ** (2 * tau)
            output = torch.cos((t * (end - start) + start) * torch.pi / 2) ** (2 * tau)
            output = (v_end - output) / (v_end - v_start)
            return output
        
    elif schedule_name.startswith("sigmoid"):
        num_params = len(schedule_name.split("_"))-1
        #format: cosine_start_end_tau
        sigmoid = lambda x: 1.0/(1.0+torch.exp(-x))
        tau = float64(1) if num_params<3 else float64(schedule_name.split("_")[3])
        end = float64(3) if num_params<2 else float64(schedule_name.split("_")[2])
        start = float64(-3) if num_params<1 else float64(schedule_name.split("_")[1])
        def gamma(t):
            v_start = sigmoid(start / tau)
            v_end = sigmoid(end / tau)
            output = sigmoid((t * (end - start) + start) / tau)
            output = (v_end - output) / (v_end - v_start)
            return output
    elif schedule_name=="linear_simple":
        gamma = lambda t: 1-t
    else:
        raise ValueError(f"Unknown schedule name: {schedule_name}, must be one of ['linear', 'cosine_[start]_[end]_[tau]', 'sigmoid_[start]_[end]_[tau]', 'linear_simple']")
    
    b = (b if torch.is_tensor(b) else torch.tensor(b)).to(torch.float64)
    input_scaling = (b-1.0).abs().item()>1e-9
    if input_scaling:
        gamma_input_scaled = lambda t: b*gamma(t)/((b-1)*gamma(t)+1)
    else:
        gamma_input_scaled = gamma
        
    gamma_wrapped = lambda t: gamma_input_scaled((t if torch.is_tensor(t) else torch.tensor(t)).to(torch.float64)).clamp_min(clip_min)
    return gamma_wrapped

def inter_save_map(x,save_i_idx):
    if torch.is_tensor(x):
        if x.numel()==1:
            return torch.tensor(x.item())
        else:
            return x[save_i_idx].cpu()
    elif x is None:
        return None
    else:
        assert isinstance(x,(float,int)), f"x={x}"
        return torch.tensor(x)
    
def type_from_maybe_str(s,class_type):
    if isinstance(s,class_type):
        return s
    list_of_attribute_strings = [a for a in dir(class_type) if not a.startswith("__")]
    list_of_attribute_strings_lower = [a.lower() for a in list_of_attribute_strings]
    if s.lower() in list_of_attribute_strings_lower:
        s_maybe_not_lower = list_of_attribute_strings[list_of_attribute_strings_lower.index(s.lower())]
        return class_type[s_maybe_not_lower]
    raise ValueError(f"Unknown type: {s}, must be one of {list_of_attribute_strings}")
    
class ModelPredType(enum.Enum):
    """Which type of output the model predicts."""
    EPS = enum.auto()  
    X = enum.auto() 
    V = enum.auto()
    BOTH = enum.auto()

class WeightsType(enum.Enum):
    """Which type of output the model predicts."""
    SNR = enum.auto()
    SNR_plus1 = enum.auto()
    SNR_trunc = enum.auto()
    uniform = enum.auto()

class TimeCondType(enum.Enum):
    """Time condition type the model uses"""
    logSNR = enum.auto()
    t = enum.auto()

class VarType(enum.Enum):
    """Time condition type the model uses"""
    small = enum.auto()
    large = enum.auto()

class SamplerType(enum.Enum):
    """How to sample timesteps for training"""
    uniform = enum.auto()
    low_discrepency = enum.auto()
    uniform_low_discrepency = enum.auto()
class ContinuousGaussianDiffusion():
    def __init__(self, 
                 analog_bits,
                 schedule_name, 
                 input_scale, 
                 model_pred_type, 
                 weights_type, 
                 time_cond_type, 
                 sampler_type,
                 var_type, 
                 clip_min=1e-9):
        """class to handle the diffusion process"""
        self.ab = analog_bits
        self.gamma = get_named_gamma_schedule(schedule_name,input_scale,clip_min=clip_min)
        self.model_pred_type = type_from_maybe_str(model_pred_type,ModelPredType)
        self.time_cond_type = type_from_maybe_str(time_cond_type,TimeCondType)
        self.var_type = type_from_maybe_str(var_type,VarType)
        self.weights_type = type_from_maybe_str(weights_type,WeightsType)
        self.sampler_type = type_from_maybe_str(sampler_type,SamplerType)
        
    def snr(self,t):
        """returns the signal to noise ratio"""
        return self.gamma(t)/(1-self.gamma(t))
    
    def alpha(self,t):
        """returns the signal coeffecient"""
        return torch.sqrt(self.gamma(t))
    
    def sigma(self,t):
        """returns the noise coeffecient"""
        return torch.sqrt(1-self.gamma(t))
    
    def logsnr(self,t):
        """returns the log signal-to-noise ratio"""
        return torch.log(self.snr(t))
    
    def loss_weights(self, t, use_float64=True):
        snr = self.snr(t)
        if self.weights_type==WeightsType.SNR:
            weights = snr
        elif self.weights_type==WeightsType.SNR_plus1:
            weights = 1+snr
        elif self.weights_type==WeightsType.SNR_trunc:
            weights = torch.maximum(snr,torch.ones_like(snr))
        elif self.weights_type==WeightsType.uniform:
            weights = torch.ones_like(snr)
        return weights
    
    def train_loss_step(self, model, x, model_kwargs={}, eps=None):
        """compute one training step and return the loss"""
        if self.ab is not None:
            assert x.shape[self.ab.bit_dim]==1, f"analog bit dimension, {self.ab.bit_dim}, must have size 1, got {x.shape[self.ab.bit_dim]}"
            x = self.ab.int2bit(x)
        bs = x.shape[0]
        if self.sampler_type==SamplerType.uniform:
            t = torch.rand(bs).to(x.device)
        elif self.sampler_type==SamplerType.low_discrepency:
            t0 = torch.rand()/bs
            t = (torch.arange(bs)/bs+t0).to(x.device)
            t = t[torch.randperm(bs)]
        elif self.sampler_type==SamplerType.uniform_low_discrepency:
            t = ((torch.arange(bs)[torch.randperm(bs)]+torch.rand(bs))/bs).to(x.device)
        if eps is None:
            eps = torch.randn_like(x)
        
        loss_weights = self.loss_weights(t)
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        x_t = mult_(alpha_t,x) + mult_(sigma_t,eps)
        output = model(x_t, t, **model_kwargs)
        
        pred_x, pred_eps = self.get_predictions(output,x_t,alpha_t,sigma_t)
        losses = mult_(loss_weights,mse_loss(pred_x,x))
        loss = torch.mean(losses)
        out =  {"loss_weights": loss_weights,
                "loss": loss,
                "losses": losses, 
                "pred_x": pred_x, 
                "x": x,
                "pred_eps": pred_eps,
                "x_t": x_t,
                "eps": eps,
                "raw_model_output": output}
        return out
    
    def get_x_from_eps(self,eps,x_t,alpha_t,sigma_t):
        """returns the predicted x from eps"""
        #return (1/alpha_t)*(x_t-sigma_t*eps)
        return mult_(1/alpha_t,x_t) - mult_(sigma_t/alpha_t,eps)
    
    def get_eps_from_x(self,x,x_t,alpha_t,sigma_t):
        """returns the predicted eps from x"""
        #return (1/sigma_t)*(x_t-alpha_t*x)
        return mult_(1/sigma_t,x_t) - mult_(alpha_t/sigma_t,x)
    
    def get_predictions(self, output, x_t, alpha_t, sigma_t, clip_x=False,guidance_weight=None,model_output_guidance=None):
        """returns predictions based on the equation x_t = alpha_t*x + sigma_t*eps"""
        if self.model_pred_type==ModelPredType.EPS:
            pred_eps = output
            if guidance_weight is None:
                pred_x = self.get_x_from_eps(pred_eps,x_t,alpha_t,sigma_t)
        elif self.model_pred_type==ModelPredType.X:
            pred_x = output
            pred_eps = self.get_eps_from_x(pred_x,x_t,alpha_t,sigma_t)
        elif self.model_pred_type==ModelPredType.BOTH:
            pred_eps, pred_x = torch.split(output, output.shape[1]//2, dim=1)
            #reconsiles the two predictions (parameterized by eps and by direct prediction):
            pred_x = alpha_t*pred_x+sigma_t*self.get_x_from_eps(pred_eps,x_t,alpha_t,sigma_t)
        elif self.model_pred_type==ModelPredType.V:
            #V = alpha*eps-sigma*x
            v = output
            pred_x = alpha_t*x_t - sigma_t*v
            pred_eps = self.get_eps_from_x(pred_x,x_t,alpha_t,sigma_t)
        
        if guidance_weight is not None:
            pred_eps = (1+guidance_weight)*pred_eps - guidance_weight*self.get_predictions(model_output_guidance,x_t,alpha_t,sigma_t,clip_x=False)[1]
            pred_x = self.get_x_from_eps(pred_eps,x_t,alpha_t,sigma_t)
        if clip_x:
            assert not pred_x.requires_grad
            pred_x = torch.clamp(pred_x,-1,1)
            #pred_eps = (1/sigma_t)*(x_t-alpha_t*pred_x) Should this be done? TODO
        return pred_x, pred_eps
        
    def ddim_step(self, i, pred_x, pred_eps, num_steps):
        logsnr_s = self.logsnr(torch.tensor(i / num_steps))
        stdv_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        x_s_pred = alpha_s * pred_x + stdv_s * pred_eps
        if i==0:
            return pred_x
        else:
            return x_s_pred

    def ddpm_step(self, i, pred_x, x_t, num_steps):
        t = torch.tensor((i + 1.) / num_steps).to(pred_x.dtype)
        s = torch.tensor(i / num_steps).to(pred_x.dtype)
        x_s_dist = self.p_distribution(
            x_t=x_t,
            pred_x=pred_x,
            logsnr_t=self.logsnr(t),
            logsnr_s=self.logsnr(s))
        if i==0:
            return x_s_dist['pred_x']
        else:
            return x_s_dist['mean'] + x_s_dist['std'] * torch.randn_like(x_t)
        
    def transform_guidance_weight(self, gw, x):
        if gw is None:
            return None
        else:
            bs = x.shape[0]
            device = x.device
            dtype = x.dtype
            w = torch.tensor(gw, dtype=dtype, device=device) if not torch.is_tensor(gw) else gw
            if w.numel() != bs:
                assert w.numel() == 1, f"guidance_weight must be a scalar or batch_size={bs} got {str(w.numel())}"
                if abs(w)<1e-9:
                    return None
                w = w.repeat(bs)
            assert w.numel() == bs, f"guidance_weight must be a scalar or batch_size={bs} got {str(w.numel())}"
            w = w.view(bs,1,1,1)
            return w
        
    def sample_loop(self, model, x_init, num_steps, sampler_type, clip_x=False, model_kwargs={},
                    guidance_weight=0.0, self_cond=False, progress_bar=False, save_i_steps=[], save_i_idx=[]):
        if sampler_type == 'ddim':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddim_step(i, pred_x, pred_eps, num_steps)
        elif sampler_type == 'ddpm':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddpm_step(i, pred_x, x_t, num_steps)
        else:
            raise NotImplementedError(sampler_type)
        
        guidance_weight = self.transform_guidance_weight(guidance_weight,x_init)
        if self.ab is not None:
            assert x_init.shape[self.ab.bit_dim]==self.ab.num_bits, f"analog bit dimension, {self.ab.bit_dim}, must have size {self.ab.num_bits}, got {x_init.shape[self.ab.bit_dim]}"
        
        if progress_bar:
            trange = tqdm.tqdm(range(num_steps-1, -1, -1), desc="Batch progress.")
        else:
            trange = range(num_steps-1, -1, -1)
            
        sample_output = {}
        if len(save_i_steps)>0 and len(save_i_idx)>0:
            intermediate_save = True
            inter_keys = ["x_t","pred_x","pred_eps","model_output","model_output_guidance","i","t"]
            sample_output["inter"] = {k: [] for k in inter_keys}
        else:
            intermediate_save = False
        x_t = x_init
        
        for i in trange:
            t = torch.tensor((i + 1.) / num_steps)
            alpha_t, sigma_t = self.alpha(t), self.sigma(t)
            t_cond = self.to_t_cond(t).to(x_t.dtype).to(x_t.device)
            
            if guidance_weight is not None:
                model_output_guidance = model(x_t, t_cond, **{k: v for (k,v) in model_kwargs if k in self.guidance_kwargs})
            else:
                model_output_guidance = None
            
            model_output = model(x_t, t_cond, **model_kwargs)
            pred_x, pred_eps = self.get_predictions(output=model_output,
                                                    x_t=x_t,
                                                    alpha_t=alpha_t,
                                                    sigma_t=sigma_t,
                                                    clip_x=clip_x,
                                                    guidance_weight=guidance_weight,
                                                    model_output_guidance=model_output_guidance)
            if intermediate_save:
                if i in save_i_steps:
                    for key,value in zip(inter_keys,[pred_x,pred_eps,model_output,model_output_guidance,i,t]):
                        sample_output["inter"][key].append(inter_save_map(value,save_i_idx))
            
            if self_cond:
                model_kwargs['self_cond'] = x_t
            
            x_t = body_fun(i, pred_x, pred_eps, x_t)

        assert x_t.shape == x_init.shape and x_t.dtype == x_init.dtype
        
        sample_output["pred"] = x_t
        return sample_output

    def to_t_cond(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        if self.time_cond_type==TimeCondType.logSNR:
            return self.logsnr(t)
        elif self.time_cond_type==TimeCondType.t:
            return t
    
    def vb(self, x, x_t, logsnr_t, logsnr_s, model_output):
        assert x.shape == x_t.shape
        assert logsnr_t.shape == logsnr_s.shape == (x_t.shape[0],)
        q_dist = self.q_distribution(x=x,x_t=x_t,logsnr_t=logsnr_t,logsnr_s=logsnr_s,x_logvar='small')
        p_dist = self.p_distribution(x_t=x_t, logsnr_t=logsnr_t,logsnr_s=logsnr_s)
        kl = normal_kl(
            mean1=q_dist['mean'], logvar1=q_dist['logvar'],
            mean2=p_dist['mean'], logvar2=p_dist['logvar'])
        return kl.mean((1,2,3)) / torch.log(2.)
    
    def p_distribution(self, x_t, pred_x, logsnr_t, logsnr_s):
        """computes p(x_s | x_t)."""
        if self.var_type==VarType.small:
            x_logvar = "small"
        else: 
            assert self.var_type==VarType.large
            x_logvar = "large"
            
        out = self.q_distribution(
            x_t=x_t, logsnr_t=logsnr_t, logsnr_s=logsnr_s,
            x=pred_x, x_logvar=x_logvar)
        out['pred_x'] = pred_x
        return out
    
    def q_distribution(self, x, x_t, logsnr_s, logsnr_t, x_logvar):
        """computes q(x_s | x_t, x) (requires logsnr_s > logsnr_t (i.e. s < t))."""
        alpha_st = torch.sqrt((1. + torch.exp(-logsnr_t)) / (1. + torch.exp(-logsnr_s)))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        r = torch.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
        one_minus_r = -torch.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
        log_one_minus_r = torch.log1p(-torch.exp(logsnr_s - logsnr_t))  # log(1-SNR(t)/SNR(s))

        mean = mult_(r * alpha_st, x_t) + mult_(one_minus_r * alpha_s, x)
        
        if x_logvar == 'small':
            # same as setting x_logvar to -infinity
            var = one_minus_r * torch.sigmoid(-logsnr_s)
            logvar = log_one_minus_r + torch.nn.LogSigmoid()(-logsnr_s)
        elif x_logvar == 'large':
            # same as setting x_logvar to nn.LogSigmoid()(-logsnr_t)
            var = one_minus_r * torch.sigmoid(-logsnr_t)
            logvar = log_one_minus_r + torch.nn.LogSigmoid()(-logsnr_t)
        else:        
            raise NotImplementedError(x_logvar)
        return {'mean': mean, 'std': torch.sqrt(var), 'var': var, 'logvar': logvar}
    
def create_diffusion_from_args(args):
    num_bits = np.ceil(np.log2(args.max_num_classes)).astype(int)
    ab = AnalogBits(num_bits=num_bits)

    cgd = ContinuousGaussianDiffusion(analog_bits=ab,
                                    schedule_name=args.noise_schedule,
                                    input_scale=args.input_scale,
                                    model_pred_type=args.predict,
                                    weights_type=args.loss_weights,
                                    time_cond_type=args.time_cond_type,
                                    sampler_type=args.schedule_sampler,
                                    var_type="small" if args.sigma_small else "large",
                                    clip_min=args.gamma_clip_min)
    return cgd
    
def main():
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    def dummy_model(x,t,noise_level=0.01):
            return x+noise_level*torch.randn_like(x)
    def dummy_data(bs=16,imsize=32,upscale=4):
        x_small = torch.randn(bs,3,imsize//upscale,imsize//upscale)
        x_big = torch.nn.functional.interpolate(x_small,scale_factor=upscale,mode="bilinear")
        return x_big
    if args.unit_test==0:
        print("UNIT TEST: basic gamma functions")
        
        schedule_names = ["cosine","sigmoid","linear","linear_simple"]
        for schedule_name in schedule_names:
            gamma = get_named_gamma_schedule(schedule_name,1.0)
            t = torch.linspace(0,1,1000)
            plt.plot(t,gamma(t),label=schedule_name)
        plt.legend()
        plt.show()

    elif args.unit_test==1:
        print("UNIT TEST: gamma functions with parameters")
        schedule_names = ["sigmoid_0_3_0.3","sigmoid_0_3_0.5","cosine_0.2_1_2","cosine_0.2_1_3"]
        for schedule_name in schedule_names:
            gamma = get_named_gamma_schedule(schedule_name,1.0)
            t = torch.linspace(0,1,1000)
            plt.subplot(1,2,1)
            plt.plot(t,gamma(t),label=schedule_name)
            logsnr = torch.log(gamma(t)/(1-gamma(t)))
            plt.subplot(1,2,2)
            plt.plot(t,logsnr,label=schedule_name)
        plt.legend()
        plt.subplot(1,2,1)
        plt.ylabel("gamma(t)")
        plt.subplot(1,2,2)
        plt.ylabel("logSNR(t)")
        plt.ylim(-22,22)
        plt.show()
    elif args.unit_test==2:
        print("UNIT TEST: simple_linear gamma function with different b scales")
        b_vec = [0.0001,0.01,1.0]
        for b in b_vec:
            gamma = get_named_gamma_schedule("linear_simple",b)
            t = torch.linspace(0,1,1000)
            plt.subplot(1,2,1)
            plt.plot(t,gamma(t),label=f"b={b}")
            logsnr = torch.log(gamma(t)/(1-gamma(t)))
            plt.subplot(1,2,2)
            plt.plot(t,logsnr,label=f"b={b}")
        plt.subplot(1,2,1)
        plt.legend()
        plt.ylabel("gamma(t)")
        plt.subplot(1,2,2)
        plt.legend()
        plt.ylabel("logSNR(t)")
        plt.ylim(-22,22)
        plt.show()
    elif args.unit_test==3:
        print("UNIT TEST: unit_test 2 and 3 together")
        b_vec = [0.0001,0.01,1.0]
        schedule_names = ["sigmoid_0_3_0.3","sigmoid_0_3_0.5","cosine_0.2_1_2","cosine_0.2_1_3"]
        for b in b_vec:
            gamma = get_named_gamma_schedule("linear_simple",b)
            t = torch.linspace(0,1,1000)
            plt.subplot(1,2,1)
            plt.plot(t,gamma(t),label=f"b={b}")
            logsnr = torch.log(gamma(t)/(1-gamma(t)))
            plt.subplot(1,2,2)
            plt.plot(t,logsnr,label=f"b={b}")
        for schedule_name in schedule_names:
            gamma = get_named_gamma_schedule(schedule_name,1.0)
            t = torch.linspace(0,1,1000)
            plt.subplot(1,2,1)
            plt.plot(t,gamma(t),label=schedule_name)
            logsnr = torch.log(gamma(t)/(1-gamma(t)))
            plt.subplot(1,2,2)
            plt.plot(t,logsnr,label=schedule_name)
        plt.subplot(1,2,1)
        plt.legend()
        plt.ylabel("gamma(t)")
        plt.subplot(1,2,2)
        plt.legend()
        plt.ylabel("logSNR(t)")
        plt.ylim(-22,22)
        plt.show()
    elif args.unit_test==4:
        print("UNIT TEST: test train_loss_step")
        x = dummy_data()
        model = lambda x,t: dummy_model(x,t,noise_level=0.01)
        cgd = ContinuousGaussianDiffusion(AnalogBits(),
                                                "linear_simple",
                                                1.0,
                                                ModelPredType.X,
                                                WeightsType.SNR,
                                                TimeCondType.t,
                                                SamplerType.uniform,
                                                VarType.small)
        
        out = cgd.train_loss_step(model,x)
        print(out["loss"])
        print(out["loss_weights"].shape)
        print(out.keys())
    elif args.unit_test==5:
        print("UNIT TEST: show images from train_loss_step")
        x = dummy_data()
        model = lambda x,t: dummy_model(x,t,noise_level=0.01)
        
        cgd = ContinuousGaussianDiffusion(AnalogBits(),"linear_simple",1.0,
                                                ModelPredType.X,
                                                WeightsType.SNR,
                                                TimeCondType.t,
                                                SamplerType.uniform,
                                                VarType.small)
        
        out = cgd.train_loss_step(model,x)
        
        image_keys = ["x_t","pred_x","x","pred_eps","eps"]
        for key in image_keys:
            im = out[key][0].permute(1,2,0).detach().cpu().numpy()
            im = np.clip(im*0.5+0.5,0,1)
            plt.subplot(1,len(image_keys),image_keys.index(key)+1)
            plt.imshow(im)
            plt.title(key)
        plt.show()
        
    elif args.unit_test==6:
        print("UNIT TEST: test sample_loop")
        x_t = dummy_data()
        model = lambda x,t: dummy_model(x,t,noise_level=0.01)
        cgd = ContinuousGaussianDiffusion(AnalogBits(),"linear_simple",1.0,
                                                ModelPredType.X,
                                                WeightsType.SNR,
                                                TimeCondType.t,
                                                SamplerType.uniform,
                                                VarType.small)  
        pred_x = cgd.sample_loop(model,x_t,10,"ddim",clip_x=1.0)
        print(pred_x.shape)
        
        plt.subplot(1,2,1)
        im = x_t[0].permute(1,2,0).detach().cpu().numpy()
        im = np.clip(im*0.5+0.5,0,1)
        plt.imshow(im)
        plt.title("x_t")
        plt.subplot(1,2,2)
        im = pred_x[0].permute(1,2,0).detach().cpu().numpy()
        im = np.clip(im*0.5+0.5,0,1)
        plt.imshow(im)
        plt.title("pred_x")
        plt.show()
    elif args.unit_test==7:
        print("UNIT TEST: compare loss weight functions")
        loss_weight_types = [WeightsType.SNR,WeightsType.SNR_plus1,WeightsType.SNR_trunc]
        t = torch.linspace(0,1,1000)
        for loss_weight_type in loss_weight_types:
            cgd = ContinuousGaussianDiffusion(AnalogBits(),"cosine",1.0,
                                                ModelPredType.X,
                                                loss_weight_type,
                                                TimeCondType.t,
                                                SamplerType.uniform,
                                                VarType.small)
            
            w = cgd.loss_weights(t)
            
            #w *= 1 - uniform_prob
            #w += uniform_prob / len(w)
            #w /= w.sum()
            print(w[:5],w[-5:])
            print(w.min(),w.max())
            logw = torch.log(w)
            print(logw.min(),logw.max())
            
            plt.subplot(2,2,1)
            plt.plot(cgd.logsnr(t),logw,label=f"{loss_weight_type.name}")
            plt.subplot(2,2,2)
            plt.plot(cgd.logsnr(t),w,label=f"{loss_weight_type.name}")
            plt.subplot(2,2,3)
            plt.plot(t,logw,label=f"{loss_weight_type.name}")
            plt.subplot(2,2,4)
            plt.plot(t,w,label=f"{loss_weight_type.name}")
        plt.subplot(2,2,1)
        plt.legend()
        plt.ylabel("log(loss_weight)")
        plt.xlabel("logSNR")
        #plt.xlim(-6,6)
        plt.subplot(2,2,2)
        plt.legend()
        plt.ylabel("loss_weight")
        plt.xlabel("logSNR")
        plt.xlim(-6,6)
        plt.ylim(0,10)
        plt.subplot(2,2,3)
        plt.legend()
        plt.ylabel("log(loss_weight)")
        plt.xlabel("t")
        plt.subplot(2,2,4)
        plt.legend()
        plt.ylabel("loss_weight")
        plt.xlabel("t")
        
        plt.show()
    elif args.unit_test==8:
        print("UNIT TEST: compare loss alpha,sigma,SNR,logSNR")
        schedule_name = "cosine"
        
        cgd = ContinuousGaussianDiffusion(AnalogBits(),schedule_name,1.0,
                                                ModelPredType.X,
                                                WeightsType.SNR,
                                                TimeCondType.t,
                                                SamplerType.uniform,
                                                VarType.small)
        t = torch.linspace(0,1,1000)
        plt.plot(t,cgd.snr(t),label="SNR")
        plt.plot(t,cgd.alpha(t),label="alpha")
        plt.plot(t,cgd.sigma(t),label="sigma")
        plt.plot(t,cgd.logsnr(t),label="logSNR")
        plt.ylim(-0.1,1.1)
        plt.legend()
        plt.show()
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()