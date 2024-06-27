#Continuous Guassian Diffusion implemented by Jakob Loenborg Christensen (JakobLC github) jloch@dtu.dk

import enum
import numpy as np
import torch
from datasets import AnalogBits
from source.utils.utils import normal_kl,mse_loss,ce1_loss,ce2_loss,ce2_logits_loss
import tqdm
from source.models.cond_vit import dynamic_image_keys

def cond_kwargs_int2bit(kwargs,ab,keys=dynamic_image_keys):
    """loops over dynamic image keys and converts the label part to bits"""
    pesent_dynamic_keys = set(kwargs.keys()).intersection(set(keys))
    for key in pesent_dynamic_keys:
        #x should be a tuple of (label, image). verify this
        assert all([isinstance(x,(tuple,list)) for x in kwargs[key]]), f"expected a tuple of (label,image) for key={key}. got type(x[0])={type(kwargs[key][0])}"
        assert all([len(x)==2 for x in kwargs[key]]), f"expected a tuple of (label,image), i.e. len 2 for key={key}. got len(x[0])={len(kwargs[key][0])}"
        kwargs[key] = [(torch.cat([ab.int2bit(x[0][None])[0],x[1]]) 
                        if (x is not None) else None) 
                       for x in kwargs[key]]
        
    return kwargs

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

def get_named_gamma_schedule(schedule_name,b,logsnr_min=-20.0,logsnr_max=20.0):
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
    elif schedule_name=="parabola":
        #gamma = lambda t: (1-t**2)**2 expanded:
        gamma = lambda t: 1-2*t**2+t**4
    else:
        raise ValueError(f"Unknown schedule name: {schedule_name}, must be one of ['linear', 'cosine_[start]_[end]_[tau]', 'sigmoid_[start]_[end]_[tau]', 'linear_simple']")
    
    b = (b if torch.is_tensor(b) else torch.tensor(b)).to(torch.float64)
    gamma_wrap1 = input_scaling_wrap(gamma,b)
    slope,bias = logsnr_wrap(gamma_wrap1,logsnr_min,logsnr_max)
    gamma_wrap2 = lambda t: gamma_wrap1((t if torch.is_tensor(t) else torch.tensor(t)).to(torch.float64))*slope+bias
    return gamma_wrap2

def input_scaling_wrap(gamma,b=1.0):
    input_scaling = (b-1.0).abs().item()>1e-9
    if input_scaling:
        gamma_input_scaled = lambda t: b*gamma(t)/((b-1)*gamma(t)+1)
    else:
        gamma_input_scaled = gamma
    return gamma_input_scaled

def logsnr_wrap(gamma,logsnr_min=-10,logsnr_max=10,dtype=torch.float64):
    if dtype==torch.float64:
        assert logsnr_max<=36, "numerical issues are reached with logsnr_max>36 for float64"
    assert logsnr_min<logsnr_max, "expected logsnr_min<logsnr_max"
    g1_old = gamma(torch.tensor(1,dtype=dtype))
    g0_old = gamma(torch.tensor(0,dtype=dtype))
    g0_new = 1/(1+torch.exp(-torch.tensor(logsnr_max,dtype=dtype)))
    g1_new = 1/(1+torch.exp(-torch.tensor(logsnr_min,dtype=dtype)))
    slope = (g0_new-g1_new)/(g0_old-g1_old)
    bias = g1_new-g1_old*slope
    return slope,bias

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
    P = enum.auto()
    P_logits = enum.auto()

class LossType(enum.Enum):
    """Which type of loss the model uses."""
    MSE = enum.auto()  
    CE1 = enum.auto()
    CE2 = enum.auto()

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
    uniform_low_d = enum.auto()


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
                 loss_type,
                 logsnr_min=-10.0,
                 logsnr_max=10.0):
        """class to handle the diffusion process"""
        self.ab = analog_bits
        self.loss_type = type_from_maybe_str(loss_type,LossType)
        self.gamma = get_named_gamma_schedule(schedule_name,b=input_scale,logsnr_min=logsnr_min,logsnr_max=logsnr_max)
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
    
    def sample_t(self,bs):
        if self.sampler_type==SamplerType.uniform:
            t = torch.rand(bs)
        elif self.sampler_type==SamplerType.low_discrepency:
            t0 = torch.rand()/bs
            t = (torch.arange(bs)/bs+t0)
            t = t[torch.randperm(bs)]
        elif self.sampler_type==SamplerType.uniform_low_d:
            t = ((torch.arange(bs)[torch.randperm(bs)]+torch.rand(bs))/bs)
        else:
            raise NotImplementedError(self.sampler_type)
        return t
    
    def train_loss_step(self, model, x, model_kwargs={}, eps=None, t=None, self_cond=False):
        """compute one training step and return the loss"""
        if isinstance(self_cond,bool):
            self_cond = [self_cond for _ in range(len(x))]
        assert isinstance(self_cond,list)
        self_cond = torch.tensor(self_cond).view(-1,1,1,1).to(x.device)

        if self.ab is not None:
            assert x.shape[self.ab.bit_dim]==1, f"analog bit dimension, {self.ab.bit_dim}, must be 1, got {x.shape[self.ab.bit_dim]}"
            loss_mask = torch.logical_not(self.ab.int2pad(x).cpu()).float()
            x = self.ab.int2bit(x)
        if "points" in model_kwargs.keys():
            if model_kwargs["points"] is not None:
                model_kwargs["points"] = model_kwargs["points"]*x
        if t is None:
            t = self.sample_t(x.shape[0]).to(x.device)
        if eps is None:
            eps = torch.randn_like(x)
        
        loss_weights = self.loss_weights(t)
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        x_t = mult_(alpha_t,x) + mult_(sigma_t,eps)
        
        if any(self_cond):
            with torch.no_grad():
                output = model(x_t, t, **model_kwargs)
                pred_x, pred_eps = self.get_predictions(output,x_t,alpha_t,sigma_t)
                model_kwargs['self_cond'] = pred_x*self_cond #not the most effecient way to do this, but easy to implement TODO
        else:
            model_kwargs['self_cond'] = None
    
        output = model(x_t, t, **model_kwargs)
        
        pred_x, pred_eps = self.get_predictions(output,x_t,alpha_t,sigma_t)
        if self.loss_type==LossType.MSE:
            losses = mult_(loss_weights,mse_loss(pred_x,x,loss_mask))
        elif self.loss_type==LossType.CE1:
            losses = mult_(loss_weights,ce1_loss(pred_x,x,loss_mask))
        elif self.loss_type==LossType.CE2:
            if self.model_pred_type==ModelPredType.P_logits:
                losses = mult_(loss_weights,ce2_logits_loss(output,x,loss_mask))
            else:
                losses = mult_(loss_weights,ce2_loss(pred_x,x,loss_mask))
        loss = torch.mean(losses)
        out =  {"loss_weights": loss_weights,
                "loss_mask": loss_mask,
                "loss": loss,
                "losses": losses, 
                "pred_x": pred_x,
                "t": t, 
                "x": x,
                "pred_eps": pred_eps,
                "x_t": x_t,
                "eps": eps,
                "raw_model_output": output,
                "self_cond": model_kwargs['self_cond']}
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
            pred_x = mult_(alpha_t,pred_x)+mult_(sigma_t,self.get_x_from_eps(pred_eps,x_t,alpha_t,sigma_t))
        elif self.model_pred_type==ModelPredType.V:
            #V = alpha*eps-sigma*x
            v = output
            pred_x = mult_(alpha_t,x_t) - mult_(sigma_t,v)
            pred_eps = self.get_eps_from_x(pred_x,x_t,alpha_t,sigma_t)
        elif self.model_pred_type==ModelPredType.P:
            pred_x = output*2-1
            pred_eps = self.get_eps_from_x(pred_x,x_t,alpha_t,sigma_t)
        elif self.model_pred_type==ModelPredType.P_logits:
            pred_x = torch.sigmoid(output)*2-1
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
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        x_s_pred = alpha_s * pred_x + sigma_s * pred_eps
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
        
    def sample_loop(self, model, x_init, num_steps, sampler_type, clip_x=False, model_kwargs={},
                    guidance_weight=0.0, self_cond=False, progress_bar=False, save_i_steps=[], save_i_idx=[],
                    guidance_kwargs="",save_entropy_score=False):
        if sampler_type == 'ddim':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddim_step(i, pred_x, pred_eps, num_steps)
        elif sampler_type == 'ddpm':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddpm_step(i, pred_x, x_t, num_steps)
        else:
            raise NotImplementedError(sampler_type)
        
        guidance_weight = transform_guidance_weight(guidance_weight,x_init)
        if self.ab is not None:
            if self.ab.onehot:
                assert x_init.shape[self.ab.bit_dim]==self.ab.num_classes, f"analog bit dimension, {self.ab.bit_dim}, must have size {self.ab.num_classes}, got {x_init.shape[self.ab.bit_dim]}"
            else:
                assert x_init.shape[self.ab.bit_dim]==self.ab.num_bits, f"analog bit dimension, {self.ab.bit_dim}, must have size {self.ab.num_bits}, got {x_init.shape[self.ab.bit_dim]}"
        
        if progress_bar:
            trange = tqdm.tqdm(range(num_steps-1, -1, -1), desc="Batch progress.")
        else:
            trange = range(num_steps-1, -1, -1)
            
        sample_output = {}
        intermediate_save = len(save_i_steps)>0 and len(save_i_idx)>0
        if intermediate_save:
            inter_keys = ["x_t","pred_x","pred_eps","model_output","model_output_guidance","i","t"]
            sample_output["inter"] = {k: [] for k in inter_keys}
        
        if save_entropy_score:
            sample_output["entropy_score"] = []

        x_t = x_init
        
        for i in trange:
            t = torch.tensor((i + 1.) / num_steps)
            alpha_t, sigma_t = self.alpha(t), self.sigma(t)
            t_cond = self.to_t_cond(t).to(x_t.dtype).to(x_t.device)
            
            if guidance_weight is not None:
                model_output_guidance = model(x_t, t_cond, **{k: v for (k,v) in model_kwargs.items() if k in guidance_kwargs.split(",")})
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
                    for key,value in zip(inter_keys,[x_t,pred_x,pred_eps,model_output,model_output_guidance,i,t]):
                        sample_output["inter"][key].append(inter_save_map(value,save_i_idx))

            if save_entropy_score:
                sample_output["entropy_score"].append(entropy_score_from_predx(pred_x))

            if self_cond:
                model_kwargs['self_cond'] = pred_x
            
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

def transform_guidance_weight(gw, x):
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
        else:
            if (abs(w)<1e-9).all():
                return None
        assert w.numel() == bs, f"guidance_weight must be a scalar or batch_size={bs} got {str(w.numel())}"
        w = w.view(bs,1,1,1)
        return w

def create_diffusion_from_args(args):
    num_bits = np.ceil(np.log2(args.max_num_classes)).astype(int)
    ab = AnalogBits(num_bits=num_bits,
                    onehot=args.onehot)

    cgd = ContinuousGaussianDiffusion(analog_bits=ab,
                                    schedule_name=args.noise_schedule,
                                    input_scale=args.input_scale,
                                    model_pred_type=args.predict,
                                    weights_type=args.loss_weights,
                                    time_cond_type=args.time_cond_type,
                                    sampler_type=args.schedule_sampler,
                                    var_type="small" if args.sigma_small else "large",
                                    loss_type=args.loss_type,
                                    logsnr_min=args.logsnr_min,
                                    logsnr_max=args.logsnr_max)
    return cgd

def entropy_score_from_predx(predx,mean_reduce=True):
    num_bits = predx.shape[1]
    entropy = entropy_from_predx(predx,mean_reduce=False,as_onehot=False).sum(1)
    entropy_score = 1-entropy*torch.log(-2**num_bits)
    if mean_reduce:
        entropy_score = torch.mean(entropy_score)
    return entropy_score

def entropy_from_predx(predx,mean_reduce=True,as_onehot=False):
    num_bits = predx.shape[1]
    if as_onehot:
        probs = AnalogBits(num_bits=num_bits).bit2prob(predx)
    else:
        #each number is a probability, so we must add the complementary probability
        probs = predx.unsqueeze(1)*0.5+0.5
        probs = torch.cat([probs,1-probs],axis=1)
    entropy = -torch.sum(probs*torch.log(probs+1e-9),dim=1)
    if mean_reduce:
        entropy = torch.mean(entropy)
    return entropy

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
    elif args.unit_test==9:
        print("UNIT TEST: simple_linear gamma function with different b scales, showing alpha and sigma")
        b_vec = [0.01,0.02,0.03]
        for b in b_vec:
            gamma = get_named_gamma_schedule("linear_simple",b)
            alpha = lambda t: torch.sqrt(gamma(t))
            sigma = lambda t: torch.sqrt(1-gamma(t))
            t = torch.linspace(0,1,1000)
            plt.subplot(1,2,1)
            plt.plot(t,gamma(t),label=f"b={b}")
            plt.subplot(1,2,2)
            plt.plot(t,alpha(t),"-",label=f"alpha, b={b}")
            plt.plot(t,sigma(t),"--",label=f"sigma, b={b}")
        
        plt.subplot(1,2,1)
        plt.legend()
        plt.ylabel("gamma(t)")
        plt.subplot(1,2,2)
        for s in ["linear","cosine","parabola"]:
            gamma = get_named_gamma_schedule(s,1.0)
            alpha = lambda t: torch.sqrt(gamma(t))
            sigma = lambda t: torch.sqrt(1-gamma(t))
            t = torch.linspace(0,1,1000)
            plt.plot(t,alpha(t),"-",label=s)
            plt.plot(t,sigma(t),"--",label=s)
        plt.legend()
        plt.ylabel("logSNR(t)")
        plt.ylim(0,1)
        plt.show()
    elif args.unit_test==10:
        print("UNIT TEST: logSNR vs t with different clip_min and clip_max, b=0.01")
        clip_vec = [1e-2,1e-3,1e-4,1e-5]
        for clip in clip_vec:
            gamma = get_named_gamma_schedule("cosine",1.0,clip_min=clip,clip_max_delta=clip)
            logsnr = lambda t: torch.log(gamma(t)/(1-gamma(t)))
            t = torch.linspace(0,1,1000)
            plt.plot(t,logsnr(t),label=f"clip={clip}")
        plt.legend()
        plt.ylabel("logSNR(t)")
        plt.xlabel("t")
        plt.show()
        
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()