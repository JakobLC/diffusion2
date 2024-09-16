import numpy as np
import torch
import os
from PIL import Image
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix, pair_confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from skimage.morphology import binary_dilation,disk

def get_all_metrics(output,ab=None):
    assert isinstance(output,dict), "output must be an output dict"
    assert "pred_x" in output.keys(), "output must have a pred_x key"
    assert "x" in output.keys(), "output must have an x key"
    mask = output["loss_mask"] if "loss_mask" in output.keys() else None
    metrics = {**get_segment_metrics(output["pred_x"],output["x"],mask=mask,ab=ab),
               **get_mse_metrics(output)}
    metrics["likelihood"] = get_likelihood(output["pred_x"],output["x"],output["loss_mask"],ab)[1]
    return metrics

def get_likelihood(pred,target,mask,ab,outside_mask_fill_value=0.0,clamp=True):
    assert isinstance(pred,torch.Tensor), "pred must be a torch tensor"
    assert isinstance(target,torch.Tensor), "target must be a torch tensor"
    assert len(pred.shape)==len(target.shape), "pred and target must be 3D or 4D torch tensors. got pred.shape: "+str(pred.shape)+", target.shape: "+str(target.shape)
    if len(pred.shape)==3:
        was_single = True
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    else:
        was_single = False
    if mask is None:
        mask = torch.ones_like(pred)
    else:
        mask = mask.to(pred.device)
    bs = pred.shape[0]
    likelihood_images = ab.likelihood(pred,target)
    if clamp:
        likelihood_images = likelihood_images.clamp(min=0.0,max=1.0)
    likelihood_images = likelihood_images*mask + outside_mask_fill_value*(1-mask)
    likelihood = []
    for i in range(bs):
        lh = likelihood_images[i][mask[i]>0].mean().item()
        likelihood.append(lh)
    if was_single:
        likelihood_images = likelihood_images[0]
    return likelihood_images, likelihood

def get_mse_metrics(output):
    metrics = {}
    if ("pred_x" in output.keys()) and ("x" in output.keys()):
        metrics["mse_x"] = mse_loss(output["pred_x"],output["x"],output["loss_mask"]).tolist()
    if ("pred_eps" in output.keys()) and ("eps" in output.keys()):
        metrics["mse_eps"] = mse_loss(output["pred_eps"],output["eps"],output["loss_mask"]).tolist()
    return metrics

def mse_loss(pred_x, x, loss_mask=None, batch_dim=0):
    """mean squared error loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(x)*(1/torch.numel(x[0])).to(pred_x.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred_x.device)
    return torch.sum(loss_mask*(pred_x-x)**2, dim=non_batch_dims)

def ce1_loss(pred_x, x, loss_mask=None, batch_dim=0):
    """crossentropy loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(x)*(1/torch.numel(x[0])).to(pred_x.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred_x.device)
    likelihood = torch.prod(1-0.5*torch.abs(pred_x-x),axis=1,keepdims=True)
    return -torch.sum(loss_mask*torch.log(likelihood), dim=non_batch_dims)

def ce2_loss(pred_x, x, loss_mask=None, batch_dim=0):
    """crossentropy loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(x)*(1/torch.numel(x[0])).to(pred_x.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred_x.device)
    likelihood = 1-0.5*torch.abs(pred_x-x)
    return -torch.sum(loss_mask*torch.log(likelihood), dim=non_batch_dims)

def ce2_logits_loss(logits, x, loss_mask=None, batch_dim=0):
    """BCEWithLogits loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(x)*(1/torch.numel(x[0])).to(logits.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(logits.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    return torch.mean(bce(logits, (x.clone()>0.0).float(), reduction="none")*loss_mask, dim=non_batch_dims)

def load_raw_label(x):
    if isinstance(x,dict):
        assert "dataset_name" in x and "i" in x, "x must be a dictionary with the fields 'dataset_name' and 'i'"
        dataset_name = x["dataset_name"]
        i = x["i"]
    elif isinstance(x,str):
        assert len(x.split("/"))==2, "x must be a string formatted as '{dataset_name}/{i}'"
        dataset_name = x.split("/")[0]
        i = int(x.split("/")[1])
    label_filename = os.path.join(str(Path(__file__).parent.parent.parent / "data"),dataset_name,"f"+str(i//1000),f"{i}_la.png")
    return np.array(Image.open(label_filename))

def mean_iou(results, gt_seg_maps, num_classes, ignore_index,
            label_map=dict(), reduce_zero_label=False):
    total_intersect, total_union, _, _ = np.zeros((num_classes,)), np.zeros((num_classes,)), np.zeros((num_classes,)), np.zeros((num_classes,))
    
    for i in range(len(results)):
        pred_label, label = results[i], gt_seg_maps[i]

        if label_map:
            label[label == label_map[0]] = label_map[1]

        if reduce_zero_label:
            label[label == 0] = 255
            label -= 1
            label[label == 254] = 255

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
        area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
        area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
        area_union = area_pred_label + area_label - area_intersect

        total_intersect += area_intersect
        total_union += area_union

    iou = total_intersect / total_union
    all_acc = total_intersect.sum() / total_union.sum()

    return all_acc, iou

def get_segment_metrics(pred,target,mask=None,metrics=["iou","hiou","ari","mi"],ab=None,reduce_to_mean=True,acceptable_ratio_diff=0.1):
    if isinstance(target,(dict,str)):
        #we are in pure evaluation mode, i.e compare with the same target for any method in the native resolution
        #load raw target and reshape pred
        assert (len(pred.shape)==4 and pred.shape[1]==1) or len(pred.shape)==3, "pure evaluation mode expects non-batched 3D or 4D tensors"
        assert mask is None, "mask must be None in pure evaluation mode. Manually apply your mask to the target."
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred)
        if len(pred.shape)==3:
            pred = pred.unsqueeze(0)
        target = torch.tensor(load_raw_label(target),device=pred.device).unsqueeze(0).unsqueeze(0)
        h,w = target.shape[-2:]
        h1,w1 = pred.shape[-2:]
        if h1!=h or w1!=w:
            ratio_diff = min(abs(h1/w1-h/w),abs(w1/h1-w/h))
            assert ratio_diff<acceptable_ratio_diff, f"pred and target aspect ratios deviate too much. found pred.shape: {pred.shape}, target.shape: {target.shape}"
            pred = torch.nn.functional.interpolate(pred,(h,w),mode="nearest")

    assert len(pred.shape)==len(target.shape)==3 or len(pred.shape)==len(target.shape)==4, "pred and target must be 3D or 4D torch tensors. found pred.shape: "+str(pred.shape)+", target.shape: "+str(target.shape)
    assert pred.shape[-1]==target.shape[-1] and pred.shape[-2]==target.shape[-2], "pred and target must have the same spatial dimensions. found pred.shape: "+str(pred.shape)+", target.shape: "+str(target.shape)
    if isinstance(pred,torch.Tensor):
        assert isinstance(target,torch.Tensor), "target must be a torch tensor"
        assert (mask is None) or isinstance(mask,torch.Tensor), "mask must be a torch tensor"
    else:
        assert isinstance(pred,np.ndarray), "pred must be a numpy array"
        assert isinstance(target,np.ndarray), "target must be a numpy array"
        assert (mask is None) or isinstance(mask,np.ndarray), "mask must be a numpy array"
        pred = torch.tensor(pred)
        target = torch.tensor(target)
        if mask is not None:
            mask = torch.tensor(mask)
    if len(pred.shape)==len(target.shape)==3:
        was_single = True
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    else:
        was_single = False
    if not pred.shape[1]==target.shape[1]==1:
        if ab is None:
            raise ValueError("ab must be specified if pred and target are not 1-channel. Use Analog bits to get a 1-channel output.")
        pred = ab.bit2int(pred)
        target = ab.bit2int(target)
    assert len(pred.shape)==len(target.shape)==4, "batched_metrics expects 3D or 4D torch tensors"
    bs = pred.shape[0]
    if not isinstance(metrics,list):
        metrics = [metrics]
    metric_dict = {"iou": standard_iou,
                   "hiou": hungarian_iou,
                   "ari": adjusted_rand_score_stable,
                   "mi": adjusted_mutual_info_score,}
    #has to be defined inline for ab to be implicitly passed


    #metric_dict = {k: handle_empty(v) for k,v in metric_dict.items()}
    out = {metric: [] for metric in metrics}
    for i in range(bs):
        pred_i,target_i = metric_preprocess(pred[i],target[i],mask=mask[i] if mask is not None else None)
        for metric in metrics:
            out[metric].append(metric_dict[metric](pred_i,target_i))
    if was_single:
        for metric in metrics:
            out[metric] = out[metric][0]
    if reduce_to_mean:
        for metric in metrics:
            out[metric] = np.mean(out[metric])
    return out

def adjusted_rand_score_stable(target,pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(target.astype(np.uint64),pred.astype(np.uint64))
    tn,fp,fn,tp = np.float64(tn),np.float64(fp),np.float64(fn),np.float64(tp)
    if fp==0 and fn==0:
        return 1.0
    else:
        return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

def handle_empty(metric_func):
    def wrapped(target,pred,*args,**kwargs):
        if len(target)==0 and len(pred)==0:
            return 1.0
        elif len(target)==0 or len(pred)==0:
            return 0.0
        else:
            return metric_func(target,pred,*args,**kwargs)
    return wrapped

def metric_preprocess(target,pred,mask=None):
    assert isinstance(target,np.ndarray) or isinstance(target,torch.Tensor), "target must be a torch tensor or numpy array"
    assert isinstance(pred,np.ndarray) or isinstance(pred,torch.Tensor), "pred must be a torch tensor or numpy array"
    if isinstance(target,torch.Tensor):
        target = target.cpu().detach().numpy()
    if isinstance(pred,torch.Tensor):
        pred = pred.cpu().detach().numpy()
    if mask is None:
        target = target.flatten()
        pred = pred.flatten()
    else:
        if isinstance(mask,torch.Tensor):
            mask = mask.cpu().detach().numpy()>0.5
        target = target[mask]
        pred = pred[mask]
    return target,pred

def extend_shorter_vector(vec1,vec2,fill_value=0):
    if len(vec1)<len(vec2):
        vec1 = np.concatenate([vec1,(fill_value*np.ones(len(vec2)-len(vec1))).astype(vec1.dtype)])
    elif len(vec2)<len(vec1):
        vec2 = np.concatenate([vec2,(fill_value*np.ones(len(vec1)-len(vec2))).astype(vec2.dtype)])
    return vec1,vec2

def hungarian_iou(target,pred,ignore_idx=0,return_assignment=False):
    if ignore_idx is None:
        ignore_idx = []
    if isinstance(ignore_idx,list):
        assert all([isinstance(idx,int) for idx in ignore_idx]), "ignore_idx must be None, int or list[int]"
    else:
        assert isinstance(ignore_idx,int), "ignore_idx must be None, int or list[int]"
        ignore_idx = [ignore_idx]
    
    uq_target,target,conf_rowsum = np.unique(target,return_counts=True,return_inverse=True)
    uq_pred,pred,conf_colsum = np.unique(pred,return_counts=True,return_inverse=True)
    conf_rowsum,conf_colsum = extend_shorter_vector(conf_rowsum,conf_colsum)
    uq_target,uq_pred = extend_shorter_vector(uq_target,uq_pred,fill_value=-1)
    
    conf_rowsum,conf_colsum = conf_rowsum[:,None],conf_colsum[None,:]
    intersection = confusion_matrix(target, pred)

    union = conf_rowsum + conf_colsum - intersection
    iou_hungarian_mat = intersection / union

    mask_pred = np.isin(uq_pred,ignore_idx)
    mask_target = np.isin(uq_target,ignore_idx)
    #handle edge cases
    if all(mask_pred) and all(mask_target):
        val = 1.0
        assign_pred = np.array([],dtype=int)
        assign_target = np.array([],dtype=int)
        iou_per_assignment = np.array([],dtype=float)
    elif all(mask_pred) or all(mask_target):
        val = 0.0
        assign_pred = np.array([],dtype=int)
        assign_target = np.array([],dtype=int)
        iou_per_assignment = np.array([],dtype=float)
    else:
        #force optimal assignment to match ignore_idx with ignore_idx
        iou_hungarian_mat[mask_target,:] = 0
        iou_hungarian_mat[:,mask_pred] = 0
        iou_hungarian_mat += mask_target[:,None]*mask_pred[None,:]

        assignment = linear_sum_assignment(iou_hungarian_mat, maximize=True)

        assign_target = uq_target[assignment[0]]
        assign_pred = uq_pred[assignment[1]]
        iou_per_assignment = iou_hungarian_mat[assignment[0],assignment[1]]
        
        #remove matches which have ignore_idx or dummy (-1) as both target and pred
        ignore_idx.append(-1)
        mask = np.logical_or(~np.isin(assign_pred,ignore_idx),~np.isin(assign_target,ignore_idx))
        assign_target,assign_pred,iou_per_assignment = assign_target[mask],assign_pred[mask], iou_per_assignment[mask]
        
        val = np.mean(iou_per_assignment)

    if return_assignment:
        return val, assign_target, assign_pred, iou_per_assignment
    else:
        return val
    
def standard_iou(target,pred,ignore_idx=0,reduce_classes=True):
    num_classes = max(target.max(),pred.max())+1
    if num_classes==1:
        return 1.0
    intersection = np.histogram(target[pred==target], bins=np.arange(num_classes + 1))[0]
    area_pred = np.histogram(pred, bins=np.arange(num_classes + 1))[0]
    area_target = np.histogram(target, bins=np.arange(num_classes + 1))[0]
    union = area_pred + area_target - intersection
    if ignore_idx is not None:
        union[ignore_idx] = 0
    iou = intersection[union>0] / union[union>0]
    if reduce_classes:
        iou = np.mean(iou)
    return iou

def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
	"""
	Compute mean,recall and decay from per-frame evaluation.
	Calculates precision/recall for boundaries between foreground_mask and
	gt_mask using morphological operators to speed it up.

	Arguments:
		foreground_mask (ndarray): binary segmentation image.
		gt_mask         (ndarray): binary annotated image.

	Returns:
		F (float): boundaries F-measure
		P (float): boundaries precision
		R (float): boundaries recall
	"""
	assert np.atleast_3d(foreground_mask).shape[2] == 1

	bound_pix = bound_th if bound_th >= 1 else \
			np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

	# Get the pixel boundaries of both masks
	fg_boundary = seg2bmap(foreground_mask);
	gt_boundary = seg2bmap(gt_mask);

	fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
	gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

	# Get the intersection
	gt_match = gt_boundary * fg_dil
	fg_match = fg_boundary * gt_dil

	# Area of the intersection
	n_fg     = np.sum(fg_boundary)
	n_gt     = np.sum(gt_boundary)

	#% Compute precision and recall
	if n_fg == 0 and  n_gt > 0:
		precision = 1
		recall = 0
	elif n_fg > 0 and n_gt == 0:
		precision = 0
		recall = 1
	elif n_fg == 0  and n_gt == 0:
		precision = 1
		recall = 1
	else:
		precision = np.sum(fg_match)/float(n_fg)
		recall    = np.sum(gt_match)/float(n_gt)

	# Compute F measure
	if precision + recall == 0:
		F = 0
	else:
		F = 2*precision*recall/(precision+recall);

	return F

def seg2bmap(seg,width=None,height=None):
	"""
	From a segmentation, compute a binary boundary map with 1 pixel wide
	boundaries.  The boundary pixels are offset by 1/2 pixel towards the
	origin from the actual segment boundary.

	Arguments:
		seg     : Segments labeled from 1..k.
		width	  :	Width of desired bmap  <= seg.shape[1]
		height  :	Height of desired bmap <= seg.shape[0]

	Returns:
		bmap (ndarray):	Binary boundary map.

	 David Martin <dmartin@eecs.berkeley.edu>
	 January 2003
 """

	seg = seg.astype(bool)
	seg[seg>0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width  = seg.shape[1] if width  is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e  = np.zeros_like(seg)
	s  = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:,:-1]    = seg[:,1:]
	s[:-1,:]    = seg[1:,:]
	se[:-1,:-1] = seg[1:,1:]

	b        = seg^e | seg^s | seg^se
	b[-1,:]  = seg[-1,:]^e[-1,:]
	b[:,-1]  = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+np.floor((y-1)+height / h)
					i = 1+np.floor((x-1)+width  / h)
					bmap[j,i] = 1;
	return bmap

