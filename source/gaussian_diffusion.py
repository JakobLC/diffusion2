"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .utils import smooth_threshold,TemporarilyDeterministic

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name.startswith("ideal"):
        from scipy.special import erfinv
        min_success = 0.51
        max_success = 0.9999
        #linear map that maps 0 to 0.51 and 1 to 0.9999
        if len(schedule_name)==5:
            image_size = 4
        else:
            image_size = int(schedule_name[5:])

        P = image_size**2
        def get_coef_from_t(t):
            success_prob = (max_success-min_success)*(1-t) + min_success
            output = 1/(np.sqrt(P)/(erfinv(2*success_prob-1))+1)**2
            t_original = int(round(t*num_diffusion_timesteps))
            if t_original==0:
                return 1
            elif t_original==1:
                return 0.99
            else:
                return output 
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            get_coef_from_t,
        )
    elif schedule_name.startswith("corner"):
        from scipy.ndimage import gaussian_filter
        def get_linear_func_from_points(p1,p2):
            x1,y1 = p1
            x2,y2 = p2
            a = (y2-y1)/(x2-x1)
            b = y1-a*x1
            return lambda x: a*x+b
        if len(schedule_name)==6:
            corner_point = [0.15,0.075]
        else:
            assert len(schedule_name.split("_"))>1, "corner schedule name must be in the form of corner_x2_y2 or corner"
            corner_point = [float(x) for x in schedule_name.split("_")[1:]]
        f = get_linear_func_from_points([0,1],corner_point)
        g = get_linear_func_from_points(corner_point,[1,0])
        T = num_diffusion_timesteps
        t_vals = np.linspace(-1,2,T*3,endpoint=False)
        vals = np.maximum(f(t_vals),g(t_vals))
        t_vals = t_vals[T:-T+1]
        vals = gaussian_filter(vals,T/20)[T:-T+1]
        vals -= vals.min()
        vals /= vals.max()
        return betas_for_alpha_bar(
                    num_diffusion_timesteps,
            lambda t: vals[np.argmin(abs(t_vals-t))]**1,
        )
    elif schedule_name.startswith("smooth"):
        if len(schedule_name)==6:
            height = 0.25
        else:
            assert len(schedule_name.split("_"))==2, "smooth schedule name must be in the form of smooth_[height] e.g smooth_0.3"
            height  = float(schedule_name.split("_")[1])
        g = smooth_threshold([0,1.0],[0.3,height])
        g2 = smooth_threshold([0,1.0],[0.3,0.0])
        f = lambda t: 1-t
        T = num_diffusion_timesteps
        def get_variance(t):
            return (f(t)*g(t)*(1-g2(t))+g(t)*g2(t))**2
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            get_variance,
        )
    elif schedule_name.startswith("filtered_smooth"):
        if len(schedule_name)==len("filtered_smooth"):
            height = 0.1
        else:
            assert len(schedule_name.split("_"))==3, "smooth schedule name must be in the form of filtered_smooth_[height] e.g filtered_smooth__0.3"
            height = float(schedule_name.split("_")[2])
        gaussian = lambda x,mean,std: np.exp(-(x-mean)**2/(2*std**2))
        def gaussian_kernel(v, t, sigma_per_t):
            assert len(v)==len(sigma_per_t)
            assert len(v)==len(t)
            v2 = v.copy()
            for i in range(len(v)):
                if sigma_per_t[i]>0:
                    weights = gaussian(t,t[i],sigma_per_t[i])
                    weights /= weights.sum()
                    v2[i] = (v*weights).sum()
            return v2

        def gaussian_filter_vector(v,sigma,multiplier_exponent=2):
            multiplier_func = lambda x: 1-abs(2*x-1)**multiplier_exponent
            t = np.linspace(0,1,len(v))
            sigma_per_t = sigma*multiplier_func(t)
            v_filtered = gaussian_kernel(v,t,sigma_per_t)
            return v_filtered
        
        exponent = 2
        sigma = 0.1
        d = 0.2
        if schedule_name.startswith("filtered_smooth2"):
            g = smooth_threshold([0,1.0],[d,height],0)
        else:
            g = smooth_threshold([0,1.0],[d,height])
        g2 = smooth_threshold([0,1.0],[d,0.0])
        f = lambda t: 1-t
        t_vals = np.linspace(0,1,num_diffusion_timesteps+1)
        v = f(t_vals)*g(t_vals)*(1-g2(t_vals))+g(t_vals)*g2(t_vals)

        v_filtered = gaussian_filter_vector(v,sigma,exponent)
        return betas_for_alpha_bar(
                    num_diffusion_timesteps,
            lambda t: v_filtered[np.argmin(abs(t_vals-t))]**2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    BCE = enum.auto()  # use binary cross-entropy loss
    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class LossDomain(enum.Enum):
    PREVIOUS_X = enum.auto()
    EPSILON = enum.auto()
    START_X = enum.auto()

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        loss_domain=None,
    ):
        self.seed_for_noise = None
        self.seed_translate = 0
        self.guidance_kwargs = "conditioned_image"
        self.model_mean_type = model_mean_type
        if loss_domain is None:
            if self.model_mean_type==ModelMeanType.START_X:
                self.loss_domain = LossDomain.START_X
            elif self.model_mean_type==ModelMeanType.EPSILON:
                self.loss_domain = LossDomain.EPSILON
            else:
                raise NotImplementedError(self.model_mean_type)
        else:
            self.loss_domain = loss_domain
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        if self.loss_type==LossType.BCE:
            self.maybe_bce = lambda x: th.sigmoid(x)*2-1
            self.maybe_bce_gt = lambda x: (x*0.5+0.5).clamp(0,1)
            assert self.model_mean_type==ModelMeanType.START_X, "BCE loss is only supported for ModelMeanType.START_X"
            assert self.loss_domain==LossDomain.START_X, "BCE loss is only supported for LossDomain.START_X"
        else:
            self.maybe_bce_gt = lambda x: x
            self.maybe_bce = lambda x: x
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def randn_like(self, *args, **kwargs):
        if self.seed_for_noise is not None:
            with TemporarilyDeterministic(self.seed_for_noise+self.seed_translate):
                return th.randn_like(*args, **kwargs)
        else:
            return th.randn_like(*args, **kwargs)
    
    def randn(self, *args, **kwargs):
        if self.seed_for_noise is not None:
            with TemporarilyDeterministic(self.seed_for_noise+self.seed_translate):
                return th.randn(*args, **kwargs)
        else:
            return th.randn(*args, **kwargs)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            self.seed_translate = 10000
            noise = self.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def get_pred(self, x_t, t, model_output, prefix="pred_"):
        if self.model_mean_type == ModelMeanType.START_X:
            xstart = self.maybe_bce(model_output)
            eps = self._predict_eps_from_xstart(x_t=x_t, t=t, pred_xstart=xstart)
        if self.model_mean_type == ModelMeanType.EPSILON:
            eps = model_output
            xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps)
        return {prefix+"xstart":xstart,
                prefix+"eps":eps}
    
    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, guidance_weight=None, forced_xstart=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        batch_size = x.shape[0]

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        model_pred = self.get_pred(x_t=x, t=t, model_output=model_output)
        
        use_guidance = False
        if guidance_weight is not None:
            w = th.tensor(guidance_weight, dtype=x.dtype, device=x.device)
            if w.numel() != batch_size:
                assert w.numel() == 1, f"guidance_weight must be a scalar or batch_size={batch_size} got {str(w.numel())}"
                w = w.repeat(batch_size)
            assert w.numel() == batch_size, f"guidance_weight must be a scalar or batch_size={batch_size} got {str(w.numel())}"
            w = w.view(batch_size,1,1,1)
            if (w.abs()>1e-12).any():
                use_guidance = True
                for g in self.guidance_kwargs.split(","):
                    assert g in ["bbox","points","conditioned_image","labels",""]
                model_kwargs_guidance = {k:v for k,v in model_kwargs.items() if k in self.guidance_kwargs.split(",")}
                model_output_guidance = model(x, self._scale_timesteps(t), **model_kwargs_guidance)
                guidance_pred = self.get_pred(x_t=x, t=t, model_output=model_output)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        if use_guidance:
            eps = (1+w)*model_pred["pred_eps"] - w*guidance_pred["pred_eps"]
        else:
            eps = model_pred["pred_eps"]
        pred_xstart = self._process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=eps),denoised_fn,clip_denoised)
        pred_xstart = self._process_forced_xstart(pred_xstart, forced_xstart)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
                
        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
        
    def _process_forced_xstart(self, pred_xstart, forced_xstart):        
        if forced_xstart is not None:
            assert forced_xstart["mask"].shape==pred_xstart.shape, f"expected size {pred_xstart.shape} got mask shape {forced_xstart['mask'].shape}"
            assert forced_xstart["xstart"].shape==pred_xstart.shape, f"expected size {pred_xstart.shape} got xstart shape {forced_xstart['xstart'].shape}"
            if forced_xstart["mask"].dtype==th.bool:
                pred_xstart[forced_xstart["mask"]] = forced_xstart["xstart"][forced_xstart["mask"]]
            else:
                assert forced_xstart["mask"].dtype==th.float32
                pred_xstart = pred_xstart*(1-forced_xstart["mask"]) + forced_xstart["xstart"]*forced_xstart["mask"]
        return pred_xstart
    
    def _process_xstart(self,x,denoised_fn,clip_denoised):
        if denoised_fn is not None:
            x = denoised_fn(x)
        if clip_denoised:
            return x.clamp(-1, 1)
        return x
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, guidance_weight=None, forced_xstart=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            guidance_weight=guidance_weight,
            forced_xstart=forced_xstart
        )
        noise = self.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "noise": noise}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        guidance_weight=None,
        forced_xstart=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            guidance_weight=guidance_weight,
            forced_xstart=forced_xstart,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        guidance_weight=None,
        forced_xstart=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            self.seed_translate = 40000
            img = self.randn(*shape).to(device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            self.seed_translate = i
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    guidance_weight=guidance_weight,
                    forced_xstart=forced_xstart,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        guidance_weight=None,
        forced_xstart=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            guidance_weight=guidance_weight,
            forced_xstart=forced_xstart
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = self.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        guidance_weight=None,
        forced_xstart=None,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            guidance_weight=guidance_weight,
            forced_xstart=forced_xstart,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        guidance_weight=None,
        forced_xstart=None
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            guidance_weight=guidance_weight,
            forced_xstart=forced_xstart,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        guidance_weight=None,
        forced_xstart=None
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            self.seed_translate = 50000
            img = self.randn(*shape).to(device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            self.seed_translate = i
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    guidance_weight=guidance_weight,
                    forced_xstart=forced_xstart
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None, guidance_weight=None, forced_xstart=None,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, guidance_weight=guidance_weight, forced_xstart=forced_xstart
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, return_all_outputs=False):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            self.seed_translate = 20000
            noise = self.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            vb = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            model_pred = {"pred_xstart": vb["pred_xstart"],
                          "pred_eps": self._predict_eps_from_xstart(x_t, t, vb["pred_xstart"])}
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        else:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            if (((self.loss_type == LossType.MSE) or (self.loss_type == LossType.RESCALED_MSE)) and
                (self.model_var_type in [ModelVarType.LEARNED,ModelVarType.LEARNED_RANGE])):
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                vb = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )
                terms["vb"] = vb["output"]
                model_pred = {"pred_xstart": vb["pred_xstart"],
                              "pred_eps": self._predict_eps_from_xstart(x_t, t, vb["pred_xstart"])}
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0
            else:
                model_pred = self.get_pred(x_t=x_t, t=t, model_output=model_output)

            target = {LossDomain.START_X: self.maybe_bce_gt(x_start),
                      LossDomain.EPSILON: noise
                      }[self.loss_domain]
            pred = {LossDomain.START_X: model_pred["pred_xstart"],
                    LossDomain.EPSILON: model_pred["pred_eps"]
                    }[self.loss_domain]
            assert model_output.shape == target.shape == x_start.shape == pred.shape
            
            terms["mse"] = mean_flat((target - pred) ** 2)
            terms["sum"] = (target - pred).pow(2).sum(dim=(1, 2, 3))
            terms["xstartmse"] = mean_flat((x_start - model_pred["pred_xstart"]) ** 2)
            terms["epsmse"] = mean_flat((noise - model_pred["pred_eps"]) ** 2)
            
        if self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = terms["vb"] / self.num_timesteps
        elif self.loss_type == LossType.KL:
            terms["loss"] = terms["vb"]
        elif self.loss_type == LossType.MSE:
            if self.model_var_type in [ModelVarType.LEARNED,ModelVarType.LEARNED_RANGE]:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["sum"]
        elif self.loss_type == LossType.RESCALED_MSE:
            if self.model_var_type in [ModelVarType.LEARNED,ModelVarType.LEARNED_RANGE]:
                terms["loss"] = terms["mse"] + terms["vb"]/1000.0
            else:
                terms["loss"] = terms["sum"]  
        elif self.loss_type == LossType.BCE:
            terms["bce"] = mean_flat(th.nn.functional.binary_cross_entropy_with_logits(model_output, target, reduction="none"))
            terms["loss"] = terms["bce"]
        else:
            raise NotImplementedError(self.loss_type)

        if return_all_outputs:
            terms["x_t"] = x_t
            terms["eps"] = noise
            terms["xstart"] = x_start
            terms["pred_xstart"] = model_pred["pred_xstart"]
            terms["pred_eps"] = model_pred["pred_eps"]
            terms["model_output_type"] = {ModelMeanType.EPSILON: "eps",
                                          ModelMeanType.START_X: "xstart"}[self.model_mean_type]
            terms["model_loss_domain"] = {LossDomain.START_X: "xstart",
                                          LossDomain.EPSILON: "eps"}[self.loss_domain]
            terms["t"] = t
            if not (self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL):
                terms["err"] = target - pred
                terms["err_sq"] = terms["err"].pow(2)
            if "bbox" in model_kwargs.keys():
                if model_kwargs["bbox"] is not None:
                    terms["bbox"] = model_kwargs["bbox"]
            if "points" in model_kwargs.keys():
                if model_kwargs["points"] is not None:
                    terms["points"] = model_kwargs["points"]
            if "labels" in model_kwargs.keys():
                if model_kwargs["labels"] is not None:
                    terms["labels"] = model_kwargs["labels"]
        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            self.seed_translate = 30000
            noise = self.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class DummyDiffusion:
    def __init__(
        self,
        loss_type,
        num_timesteps,
        loss_domain="same",
    ):
        self.loss_type = loss_type
        self.num_timesteps = num_timesteps
        self.seed_for_noise = None

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        guidance_weight=None,
        forced_xstart=None,
    ):
        assert "conditioned_image" in model_kwargs.keys(), "image must be in model_kwargs when using a normal UNet"
        assert model_kwargs["conditioned_image"] is not None, "image must be in model_kwargs when using a normal UNet"
        if device is None:
            device = next(model.parameters()).device
        noise = th.randn(*shape).to(device=device)
        t = th.tensor([0] * shape[0], device=device)
        model_output = model(noise, self._scale_timesteps(t), **model_kwargs)
        if self.loss_type==LossType.MSE:
            output = {"sample": model_output}
        elif self.loss_type==LossType.BCE:
            output = {"sample": 2*th.sigmoid(model_output)-1}
        return [output]

    def p_sample_loop(self,*args,**kwargs):
        return self.p_sample_loop_progressive(*args,**kwargs)[0]["sample"]

    def ddim_sample_loop(self,*args,**kwargs):
        return self.p_sample_loop_progressive(*args,**kwargs)[0]["sample"]

    def ddim_sample_loop_progressive(self,*args,**kwargs):
        return self.p_sample_loop_progressive(*args,**kwargs)
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, return_all_outputs=False):
        assert "conditioned_image" in model_kwargs.keys(), "image must be in model_kwargs when using a normal UNet"
        assert model_kwargs["conditioned_image"] is not None, "image must be in model_kwargs when using a normal UNet"
        device = next(model.parameters()).device
        terms = {}
        shape = (x_start.shape[0],1,x_start.shape[2],x_start.shape[3])
        noise = th.randn(*shape).to(device=device)
        t = th.tensor([0] * shape[0], device=device)
        model_output = model(noise, self._scale_timesteps(t), **model_kwargs)
        
        
        if self.loss_type == LossType.MSE:
            target = x_start
            terms["mse"] = mean_flat((target - model_output) ** 2)
            terms["loss"] = terms["mse"]
        elif self.loss_type == LossType.BCE:
            target = (x_start*0.5+0.5).clamp(0,1)
            bce = th.nn.functional.binary_cross_entropy_with_logits
            terms["loss"] = mean_flat(bce(model_output, target, reduction="none"))
            terms["mse"] = mean_flat((x_start - (th.sigmoid(model_output)*2-1)) ** 2)
        else:
            raise NotImplementedError(self.loss_type)

        if return_all_outputs:
            terms["x_t"] = th.zeros_like(x_start)
            terms["eps"] = th.zeros_like(x_start)
            terms["xstart"] = x_start
            terms["pred_eps"] = th.zeros_like(x_start)
            terms["model_output_type"] = "xstart"
            if self.loss_type == LossType.MSE:
                terms["pred_xstart"] = model_output
                terms["err"] = target - model_output
            elif self.loss_type == LossType.BCE:
                terms["pred_xstart"] = th.sigmoid(model_output)*2-1
                terms["err"] = bce(model_output, target, reduction="none")
                
            terms["err_sq"] = terms["err"].pow(2)
            terms["t"] = t
            if "bbox" in model_kwargs.keys():
                if model_kwargs["bbox"] is not None:
                    terms["bbox"] = model_kwargs["bbox"]
            if "points" in model_kwargs.keys():
                if model_kwargs["points"] is not None:
                    terms["points"] = model_kwargs["points"]
            if "labels" in model_kwargs.keys():
                if model_kwargs["labels"] is not None:
                    terms["labels"] = model_kwargs["labels"]
                    #"image","gt","eps","x_t","pred_xstart","pred_eps","err","t"
        return terms

    def _scale_timesteps(self, t):
        return t