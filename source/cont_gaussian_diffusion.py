#Continuous Guassian Diffusion implemented by Jakob Loenborg Christensen (JakobLC github) jloch@dtu.dk

import enum
import numpy as np
import torch
from source.utils.mixed import normal_kl, construct_points, nice_split, get_padding_slices
from source.utils.metric_and_loss import mse_loss,ce1_loss,ce2_loss,ce2_logits_loss
from source.utils.argparsing import compare_strs
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

def get_named_gamma_schedule(schedule_name,b,logsnr_min=-20.0,logsnr_max=20.0):
    float64 = lambda x: torch.tensor(float(x),dtype=torch.float64)
    if schedule_name=="linear":
        gamma = lambda t: torch.sigmoid(-torch.log(torch.expm1(1e-4+10*t*t)))
    elif schedule_name=="cosine":
        gamma = lambda t: torch.cos(t*torch.pi/2)**2
    elif schedule_name=="linear_simple":
        gamma = lambda t: 1-t
    elif schedule_name=="parabola":
        gamma = lambda t: 1-2*t**2+t**4 #(1-t**2)**2 expanded
    else:
        raise NotImplementedError(schedule_name)
    
    b = (b if torch.is_tensor(b) else torch.tensor(b)).to(torch.float64)
    gamma_wrap1 = input_scaling_wrap(gamma,b)
    slope,bias = logsnr_wrap(gamma_wrap1,logsnr_min,logsnr_max)
    gamma_wrap2 = lambda t: gamma_wrap1((t if torch.is_tensor(t) else torch.tensor(t)).to(torch.float64))*slope+bias
    return gamma_wrap2

def input_scaling_wrap(gamma,b=1.0):
    input_scaling = (b-1.0).abs().item()>1e-9
    if input_scaling:
        gamma_input_scaled = lambda t: b*b*gamma(t)/((b*b-1)*gamma(t)+1)
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
    gamma = enum.auto()

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
                 schedule_name, 
                 input_scale, 
                 model_pred_type, 
                 weights_type, 
                 time_cond_type,
                 sampler_type,
                 var_type,
                 loss_type,
                 logsnr_min=-10.0,
                 logsnr_max=10.0,
                 decouple_loss_weights=False):
        """class to handle the diffusion process"""
        self.loss_type = type_from_maybe_str(loss_type,LossType)
        self.gamma = get_named_gamma_schedule(schedule_name,b=input_scale,logsnr_min=logsnr_min,logsnr_max=logsnr_max)
        self.model_pred_type = type_from_maybe_str(model_pred_type,ModelPredType)
        self.time_cond_type = type_from_maybe_str(time_cond_type,TimeCondType)
        self.var_type = type_from_maybe_str(var_type,VarType)
        self.weights_type = type_from_maybe_str(weights_type,WeightsType)
        self.sampler_type = type_from_maybe_str(sampler_type,SamplerType)
        self.decouple_loss_weights = decouple_loss_weights
        
    def snr(self,t):
        """returns the signal to noise ratio, aka alpha^2/sigma^2"""
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

    def diff_logsnr(self,t):
        """returns the derivative of the log signal-to-noise ratio"""
        t_req_grad = torch.autograd.Variable(t, requires_grad = True)
        with torch.enable_grad():
            t_grad = torch.autograd.grad(self.logsnr(t_req_grad).sum(),t_req_grad,create_graph=True)[0]
        return t_grad

    def loss_weights(self, t):
        snr = self.snr(t)
        if self.weights_type==WeightsType.SNR:
            weights = snr
        elif self.weights_type==WeightsType.SNR_plus1:
            weights = 1+snr
        elif self.weights_type==WeightsType.SNR_trunc:
            weights = torch.maximum(snr,torch.ones_like(snr))
        elif self.weights_type==WeightsType.uniform:
            weights = torch.ones_like(snr)
        elif self.weights_type==WeightsType.gamma: # aka sigmoid loss from simpler diffusion/VDM++
            weights = self.gamma(t) 
        if self.decouple_loss_weights:
            weights *= -self.diff_logsnr(t)
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
    
    def convert_self_cond(self,model_kwargs,bs):
        if "self_cond" in model_kwargs.keys():
            self_cond = model_kwargs["self_cond"]
            del model_kwargs["self_cond"]
        else:
            self_cond = False
        if isinstance(self_cond,bool):
            self_cond = [self_cond for _ in range(bs)]
        assert isinstance(self_cond,list) or torch.is_tensor(self_cond)
        assert len(self_cond)==bs
        return self_cond

    def train_loss_step(self, model, x, loss_mask=None, model_kwargs={}, eps=None, t=None):
        """compute one training step and return the loss"""
        self_cond = self.convert_self_cond(model_kwargs,x.shape[0])

        if "points" in model_kwargs.keys():
            if model_kwargs["points"] is not None:
                model_kwargs["points"] = construct_points(model_kwargs["points"],x)
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
                model_kwargs['self_cond'] = [(pred_x[i] if self_cond[i] else None) for i in range(len(x))]
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
        output =  {"loss_weights": loss_weights,
                "loss_mask": loss_mask,
                "loss": loss,
                "losses": losses, 

                "pred_bit": pred_x,
                "pred_eps": pred_eps,


                "gt_bit": x,
                "gt_eps": eps,

                "x_t": x_t,
                "t": t, 
                
                "self_cond": model_kwargs['self_cond']}
        
        return output
    
    def get_x_from_eps(self,eps,x_t,alpha_t,sigma_t):
        """returns the predicted x from eps"""
        #return (1/alpha_t)*(x_t-sigma_t*eps)
        return mult_(1/alpha_t,x_t) - mult_(sigma_t/alpha_t,eps)
    
    def get_eps_from_x(self,x,x_t,alpha_t,sigma_t):
        """returns the predicted eps from x"""
        #return (1/sigma_t)*(x_t-alpha_t*x)
        return mult_(1/sigma_t,x_t) - mult_(alpha_t/sigma_t,x)
    
    def get_predictions(self, output, x_t, alpha_t, sigma_t, clip_x=False,
                        guidance_weight=None,model_output_guidance=None,
                        replace_padding=False,imshape=None):
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
            pred_eps = (1+guidance_weight)*pred_eps - guidance_weight*self.get_predictions(model_output_guidance,x_t,
                                                                                           alpha_t,sigma_t,
                                                                                           clip_x=False,
                                                                                           replace_padding=False)[1]
            pred_x = self.get_x_from_eps(pred_eps,x_t,alpha_t,sigma_t)
        if clip_x:
            assert not pred_x.requires_grad
            pred_x = torch.clamp(pred_x,-1,1)
            #pred_eps = (1/sigma_t)*(x_t-alpha_t*pred_x) Should this be done? TODO
        if replace_padding:
            assert len(imshape)==len(pred_x), f"len(imshape)={len(imshape)}, len(pred_x.shape)={len(pred_x)}"
            for i in range(len(pred_x)):
                s = get_padding_slices(pred_x[i],imshape[i])
                pred_x[i][s] = -1
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
            return x_s_dist['pred_bit']
        else:
            return x_s_dist['mean'] + x_s_dist['std'] * torch.randn_like(x_t)
        
    def sample_loop(self, model, x_init, num_steps, sampler_type, clip_x=False, model_kwargs={},
                    guidance_weight=0.0, progress_bar=False, save_i_steps=[], save_i_idx=[],
                    guidance_kwargs="",save_entropy=False, replace_padding=False, imshape=None):
        self_cond = self.convert_self_cond(model_kwargs,x_init.shape[0])
        if replace_padding:
            assert imshape is not None, "imshape (list of image shapes before reshaping) must be provided if replace_padding is True"
        if sampler_type == 'ddim':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddim_step(i, pred_x, pred_eps, num_steps)
        elif sampler_type == 'ddpm':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddpm_step(i, pred_x, x_t, num_steps)
        else:
            raise NotImplementedError(sampler_type)
        
        guidance_weight = transform_guidance_weight(guidance_weight,x_init)
        if progress_bar:
            trange = tqdm.tqdm(range(num_steps-1, -1, -1), desc="Batch progress.")
        else:
            trange = range(num_steps-1, -1, -1)
            
        sample_output = {}
        intermediate_save = len(save_i_steps)>0 and len(save_i_idx)>0
        if intermediate_save:
            inter_keys = ["x_t","pred_bit","pred_eps","model_output","model_output_guidance","i","t"]
            sample_output["inter"] = {k: [] for k in inter_keys}
        
        if save_entropy:
            sample_output["entropy"] = []

        x_t = x_init
        
        for i in trange:
            t = torch.tensor((i + 1.) / num_steps)
            alpha_t, sigma_t = self.alpha(t), self.sigma(t)
            t_cond = self.to_t_cond(t).to(x_t.dtype).to(x_t.device)
            
            if guidance_weight is not None:
                model_output_guidance = model(x_t, t_cond, **{k: v for (k,v) in model_kwargs.items() if k in nice_split(guidance_kwargs)})
            else:
                model_output_guidance = None
            
            model_output = model(x_t, t_cond, **model_kwargs)
            pred_x, pred_eps = self.get_predictions(output=model_output,
                                                    x_t=x_t,
                                                    alpha_t=alpha_t,
                                                    sigma_t=sigma_t,
                                                    clip_x=clip_x,
                                                    guidance_weight=guidance_weight,
                                                    model_output_guidance=model_output_guidance,
                                                    replace_padding=replace_padding,
                                                    imshape=imshape)
            if intermediate_save:
                if i in save_i_steps:
                    for key,value in zip(inter_keys,[x_t,pred_x,pred_eps,model_output,model_output_guidance,i,t]):
                        sample_output["inter"][key].append(inter_save_map(value,save_i_idx))

            if save_entropy:
                sample_output["entropy"].append(entropy_from_predx(pred_x).tolist())

            if any(self_cond):
                model_kwargs['self_cond'] = [(pred_x[i] if self_cond[i] else None) for i in range(len(x_t))]
            
            x_t = body_fun(i, pred_x, pred_eps, x_t)

        assert x_t.shape == x_init.shape and x_t.dtype == x_init.dtype
        
        if save_entropy:
            sample_output["entropy"] = torch.tensor(sample_output["entropy"]).T.tolist()
        sample_output["pred_bit"] = x_t
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
        out['pred_bit'] = pred_x
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
    if compare_strs(args.model_version,"1.4.0","<"):
        exponent = 0.5
    else:
        exponent = 1
    cgd = ContinuousGaussianDiffusion(schedule_name=args.noise_schedule,
                                    input_scale=args.input_scale**exponent,
                                    model_pred_type=args.predict,
                                    weights_type=args.loss_weights,
                                    time_cond_type=args.time_cond_type,
                                    sampler_type=args.schedule_sampler,
                                    var_type="small" if args.sigma_small else "large",
                                    loss_type=args.loss_type,
                                    logsnr_min=args.logsnr_min,
                                    logsnr_max=args.logsnr_max,
                                    decouple_loss_weights=args.decouple_loss_weights)
    return cgd

def entropy_from_predx(predx,reduce_spatial=True,reduce_bits=True,reduce_batch=False,onehot=False):
    assert len(predx.shape)==4, f"expected 4D tensor, got {predx.shape}"
    if onehot:
        probs = predx
    else:
        #each number is a probability, so we must add the complementary probability
        probs = predx.unsqueeze(1)*0.5+0.5
        probs = torch.cat([probs,1-probs],axis=1)
    entropy = -(probs*torch.log(probs+1e-12)).sum(1)
    if reduce_spatial:
        entropy = entropy.mean((2,3))
    if reduce_bits:
        entropy = entropy.mean(1)
    if reduce_batch:
        entropy = entropy.mean(0)
    return entropy
