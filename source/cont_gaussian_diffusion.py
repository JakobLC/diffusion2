#Continuous Guassian Diffusion implemented by Jakob Loenborg Christensen (JakobLC github) jloch@dtu.dk

import enum
import math
import numpy as np
import torch
from utils import normal_kl
from . import nn
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
    view_shape = [1 for _ in range(len(len(shape)))]
    view_shape[batch_dim] = -1
    return torch.from_numpy(coefs).view(view_shape).to(dtype).to(device)

def get_named_gamma_schedule(schedule_name,b,clip_min=1e-9):
    float64 = lambda x: np.float64(float(x))
    if schedule_name=="linear":
        return lambda t: -np.log(np.expm1(1e-4+10*t*t))
    elif schedule_name.startswith("cosine"):
        num_params = len(schedule_name.split("_"))-1
        #format: cosine_start_end_tau
        tau = 1.0 if num_params<3 else float64(schedule_name.split("_")[2])
        end = 0 if num_params<2 else float64(schedule_name.split("_")[1])
        start = 1 if num_params<1 else float64(schedule_name.split("_")[0])
        def gamma(t):
            v_start = math.cos(start * math.pi / 2) ** (2 * tau)
            v_end = math.cos(end * math.pi / 2) ** (2 * tau)
            output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
            output = (v_end - output) / (v_end - v_start)
            return output
        
    elif schedule_name.startswith("sigmoid"):
        num_params = len(schedule_name.split("_"))-1
        #format: cosine_start_end_tau
        sigmoid = lambda x: 1.0/(1.0+np.exp(-x))
        tau = 1.0 if num_params<3 else float64(schedule_name.split("_")[2])
        end = 3 if num_params<2 else float64(schedule_name.split("_")[1])
        start = -3 if num_params<1 else float64(schedule_name.split("_")[0])
        def gamma(t):
            v_start = sigmoid(start / tau)
            v_end = sigmoid(end / tau)
            output = sigmoid((t * (end - start) + start) / tau)
            output = (v_end - output) / (v_end - v_start)
    elif schedule_name=="linear_simple":
        gamma = lambda t: 1-t
    else:
        raise ValueError(f"Unknown schedule name: {schedule_name}, must be one of ['linear', 'cosine_[start]_[end]_[tau]', 'sigmoid_[start]_[end]_[tau]', 'linear_simple']")
    def wrap_with_high_precision_clip_scale(t):
        t = t.astype(np.float64)
        b = b.astype(np.float64)
        return lambda t: np.clip(gamma(t)/((b*b-1)(1-t)+1),clip_min,1.0)
    return wrap_with_high_precision_clip_scale
    
class ModelPredType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    EPS = enum.auto()  
    X = enum.auto() 
    V = enum.auto()
    BOTH = enum.auto()

class WeightsType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    SNR = enum.auto()
    SNR_plus1 = enum.auto()
    SNR_trunc = enum.auto()
    uniform = enum.auto()

class TCondType(enum.Enum):
    logSNR = enum.auto()
    unit_interval = enum.auto()

class ContinuousGaussianDiffusion():
    def __init__(self, schedule_name, input_scale, model_pred_type, weights_type, t_cond_type, clip_min=1e-9):
        """class to handle the diffusion process"""
        self.gamma = get_named_gamma_schedule(schedule_name,input_scale,clip_min=clip_min)
        self.weights_type = weights_type
        self.model_pred_type = model_pred_type
        self.t_cond_type = t_cond_type
        
    def snr(self,t):
        """returns the signal to noise ratio"""
        return self.gamma(t)/(1-self.gamma(t))
    
    def alpha(self,t):
        """returns the signal coeffecient"""
        return np.sqrt(self.gamma(t))
    
    def sigma(self,t):
        """returns the noise coeffecient"""
        return np.sqrt(1-self.gamma(t))
    
    def logsnr(self,t):
        """returns the log signal-to-noise ratio"""
        return np.log(self.snr(t))
    
    def loss_weights(self, t, use_float64=True):
        snr = self.snr(t)
        if self.weights_type==WeightsType.SNR:
            weights = snr
        elif self.weights_type==WeightsType.SNR_plus1:
            weights = 1+snr
        elif self.weights_type==WeightsType.SNR_trunc:
            weights = np.maximum(1,snr)
        return weights
    
    def train_loss_step(self, model, x, model_kwargs={}, eps=None):
        """compute one training step and return the loss"""
        bs = x.shape[0]
        t = np.random.rand(bs,1,1,1)
        if eps is None:
            eps = torch.randn_like(x)
        
        loss_weights = self.loss_weights(t)
        t = torch.from_numpy(t).to(x.device)
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        x_t = mult_(alpha_t,x) + mult_(sigma_t,eps)
        output = model(x_t, t, **model_kwargs)
        
        pred_x, pred_eps = self.get_predictions(output,x_t,alpha_t,sigma_t)
        
        losses = mult_(loss_weights,self.mse_loss(pred_x,x))
        loss = torch.mean(losses)
        out =  {"loss_weights": loss_weights,
                "loss": loss,
                "losses": losses, 
                "pred_x": pred_x, 
                "pred_eps": pred_eps,
                "x_t": x_t,
                "eps": eps}
        return out

    def mse_loss(self, pred_x, x, batch_dim=0):
        """mean squared error loss reduced over all dimensions except batch"""
        non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
        return torch.mean((pred_x-x)**2, dim=non_batch_dims)
    
    def get_predictions(self, output, x_t, alpha_t, sigma_t):
        """returns predictions based on the equation x_t = alpha_t*x + sigma_t*eps"""
        if self.model_pred_type==ModelPredType.EPS:
            pred_eps = output
            pred_x = (1/alpha_t)*(x_t-sigma_t*pred_eps)
        elif self.model_pred_type==ModelPredType.X:
            pred_x = output
            pred_eps = (1/sigma_t)*(x_t-alpha_t*pred_x)
        elif self.model_pred_type==ModelPredType.BOTH:
            pred_eps, pred_x = torch.split(output, output.shape[1]//2, dim=1)
            pred_x_from_eps = (1/alpha_t)*(x_t-sigma_t*pred_eps)
            #reconsiles the two predictions (parameterized by eps and by direct prediction):
            pred_x = alpha_t*pred_x+sigma_t*pred_x_from_eps
        elif self.model_pred_type==ModelPredType.V:
            #V = alpha*eps-sigma*x
            v = output
            pred_x = alpha_t*x_t - sigma_t*v
            pred_eps = (1/sigma_t)*(x_t-alpha_t*pred_x)
        return pred_x, pred_eps
        
    def ddim_step(self, i, pred_x, pred_eps, num_steps):
        logsnr_s = self.logsnr(i.to(pred_x.dtype) / num_steps)
        stdv_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        x_s_pred = alpha_s * pred_x + stdv_s * pred_eps
        return torch.where(i == 0, pred_x, x_s_pred)

    def ddpm_step(self, i, pred_x, pred_eps, x_t, num_steps, clip_x):
        dtype = pred_x.dtype
        logsnr_t = self.logsnr((i + 1.).to(dtype) / num_steps)
        logsnr_s = self.logsnr(i.to(dtype) / num_steps)
        x_s_dist = self.p_distribution(
            x_t=pred_x,
            logsnr_t=logsnr_t,
            logsnr_s=logsnr_s,
            clip_x=clip_x)
        if i==0:
            x_s = x_s_dist['pred_x']
        else:
            x_s = x_s_dist['mean'] + x_s_dist['std'] * torch.randn_like(x_t)
        return x_s

    def sample_loop(self, model, x_init, num_steps, sampler_type, clip_x, model_kwargs={}):
        if sampler_type == 'ddim':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddim_step(i, pred_x, pred_eps, x_t, num_steps, clip_x)
        elif sampler_type == 'ddpm':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddpm_step(i, pred_x, pred_eps, x_t, num_steps, clip_x)
        else:
            raise NotImplementedError(sampler_type)

        x_t = x_init
        for i in range(num_steps-1, -1, -1):
            t = (i + 1.) / num_steps
            pred_x, pred_eps = self.get_predictions(model(x_t, self.to_t_cond(t), **model_kwargs))
            x_t = body_fun(i, pred_x, pred_eps, x_t)
            #update model_kwargs TODO

        assert x_t.shape == x_init.shape and x_t.dtype == x_init.dtype
        return x_t

    def to_t_cond(self, t):
        if self.t_cond_type==TCondType.logSNR:
            return self.logsnr(t)
        elif self.t_cond_type==TCondType.unit_interval:
            return t
    
    def vb(self, *, x, x_t, logsnr_t, logsnr_s, model_output):
        assert x.shape == x_t.shape
        assert logsnr_t.shape == logsnr_s.shape == (x_t.shape[0],)
        q_dist = self.q_distribution(x=x,x_t=x_t,logsnr_t=logsnr_t,logsnr_s=logsnr_s,x_logvar='small')
        p_dist = self.p_distribution(x_t=x_t, logsnr_t=logsnr_t,logsnr_s=logsnr_s,model_output=model_output)
        kl = normal_kl(
            mean1=q_dist['mean'], logvar1=q_dist['logvar'],
            mean2=p_dist['mean'], logvar2=p_dist['logvar'])
        return kl.mean((1,2,3)) / np.log(2.)
    
    def p_distribution(self, x_t, pred_x, logsnr_t, logsnr_s):
        """computes p(x_s | x_t)."""
        assert logsnr_t.shape == logsnr_s.shape == (x_t.shape[0],)
        if self.logvar_type == 'fixed_small':
            pred_x_logvar = 'small'
        elif self.logvar_type == 'fixed_large':
            pred_x_logvar = 'large'
        else:
            raise NotImplementedError(self.logvar_type)

        out = self.q_distribution(
            x_t=x_t, logsnr_t=logsnr_t, logsnr_s=logsnr_s,
            x=pred_x, x_logvar=pred_x_logvar)
        out['pred_x'] = pred_x
        return out
    
    def q_distribution(self, x, x_t, logsnr_s, logsnr_t, x_logvar):
        """computes q(x_s | x_t, x) (requires logsnr_s > logsnr_t (i.e. s < t))."""
        alpha_st = torch.sqrt((1. + torch.exp(-logsnr_t)) / (1. + torch.exp(-logsnr_s)))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        r = torch.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
        one_minus_r = -torch.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
        log_one_minus_r = torch.log1p(-torch.exp(logsnr_s - logsnr_t))  # log(1-SNR(t)/SNR(s))

        mean = r * alpha_st * x_t + one_minus_r * alpha_s * x
        
        if x_logvar == 'small':
            # same as setting x_logvar to -infinity
            var = one_minus_r * torch.sigmoid(-logsnr_s)
            logvar = log_one_minus_r + nn.LogSigmoid()(-logsnr_s)
        elif x_logvar == 'large':
            # same as setting x_logvar to nn.LogSigmoid()(-logsnr_t)
            var = one_minus_r * torch.sigmoid(-logsnr_t)
            logvar = log_one_minus_r + nn.LogSigmoid()(-logsnr_t)
        else:        
            raise NotImplementedError(x_logvar)
        return {'mean': mean, 'std': torch.sqrt(var), 'var': var, 'logvar': logvar}