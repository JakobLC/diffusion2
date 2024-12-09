import numpy as np
import torch
import os
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix
from source.utils.dataloading import load_raw_image_label
from scipy.optimize import linear_sum_assignment
from skimage.morphology import binary_dilation,disk
import warnings
from functools import partial
from source.utils.analog_bits import ab_likelihood

def get_all_metrics(output,ignore_zero=False,ambiguous=False,ab_kw={}):
    assert isinstance(output,dict), "output must be an output dict"
    assert "pred_int" in output.keys(), "output must have a pred_bit key"
    assert "gt_int" in output.keys(), "output must have a gt_bit key"
    mask = output.get("loss_mask",None)
    metrics = {**get_mse_metrics(output)}
    if ambiguous:
        raise NotImplementedError("Ambiguous metrics are not yet implemented")
        #metrics.update(get_ambiguous_metrics(output["pred_int"],output["gt_int"]))
    else:
        metrics.update(get_segment_metrics(output["pred_int"],output["gt_int"],mask=mask,ignore_zero=ignore_zero))
    if "pred_bit" in output.keys() and "gt_bit" in output.keys():
        metrics["likelihood"] = get_likelihood(output["pred_bit"],output["gt_bit"],mask=mask,ab_kw=ab_kw)[1]
    return metrics

def get_likelihood(pred,gt,mask,outside_mask_fill_value=0.0,clamp=True,ab_kw={}):
    assert isinstance(pred,torch.Tensor), "pred must be a torch tensor"
    assert isinstance(gt,torch.Tensor), "gt must be a torch tensor"
    assert len(pred.shape)==len(gt.shape), "pred and gt must be 3D or 4D torch tensors. got pred.shape: "+str(pred.shape)+", gt.shape: "+str(gt.shape)
    if len(pred.shape)==3:
        was_single = True
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    else:
        was_single = False
    if mask is None:
        mask = torch.ones_like(pred)
    else:
        mask = mask.to(pred.device)
    bs = pred.shape[0]
    try:
        likelihood_images = ab_likelihood(pred,gt,**ab_kw)
    except Exception as e:
        from source.utils.mixed import tensor_info
        print("Error in get_likelihood")
        print("Tensor info gt:\n",tensor_info(gt))
        print("Tensor info pred:\n",tensor_info(pred))
        raise e
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
    if ("pred_bit" in output.keys()) and ("x" in output.keys()):
        metrics["mse_x"] = mse_loss(output["gt_bit"],output["pred_bit"],output["loss_mask"]).tolist()
    if ("pred_eps" in output.keys()) and ("eps" in output.keys()):
        metrics["mse_eps"] = mse_loss(output["gt_eps"],output["pred_eps"],output["loss_mask"]).tolist()
    return metrics

def mse_loss(pred, gt, loss_mask=None, batch_dim=0):
    """mean squared error loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(gt.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(gt)*(1/torch.numel(gt[0])).to(pred.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred.device)
    return torch.sum(loss_mask*(pred-gt)**2, dim=non_batch_dims)

def ce1_loss(pred, gt, loss_mask=None, batch_dim=0):
    """crossentropy loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(gt.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(gt)*(1/torch.numel(gt[0])).to(pred.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred.device)
    likelihood = torch.prod(1-0.5*torch.abs(pred-gt),axis=1,keepdims=True)
    return -torch.sum(loss_mask*torch.log(likelihood), dim=non_batch_dims)

def ce2_loss(pred, gt, loss_mask=None, batch_dim=0):
    """crossentropy loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(gt.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(gt)*(1/torch.numel(gt[0])).to(pred.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred.device)
    likelihood = 1-0.5*torch.abs(pred-gt)
    return -torch.sum(loss_mask*torch.log(likelihood), dim=non_batch_dims)

def ce2_logits_loss(pred, gt, loss_mask=None, batch_dim=0):
    """BCEWithLogits loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(gt.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(gt)*(1/torch.numel(gt[0])).to(pred.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    return torch.mean(bce(pred, (gt.clone()>0.0).float(), reduction="none")*loss_mask, dim=non_batch_dims)

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

def get_segment_metrics(pred,gt,
                        mask=None,
                        reduce_to_mean=True,
                        acceptable_ratio_diff=0.1,
                        ignore_zero=False):
    if isinstance(gt,(dict,str)):
        #we are in pure evaluation mode, i.e compare with the same gt for any method in the native resolution
        #load raw gt and reshape pred
        assert (len(pred.shape)==4 and pred.shape[1]==1) or len(pred.shape)==3, "pure evaluation mode expects non-batched 3D or 4D tensors"
        assert mask is None, "mask must be None in pure evaluation mode. Manually apply your mask to the gt."
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred)
        if len(pred.shape)==3:
            pred = pred.unsqueeze(0)
        gt = torch.tensor(load_raw_label(gt),device=pred.device).unsqueeze(0).unsqueeze(0)
        h,w = gt.shape[-2:]
        h1,w1 = pred.shape[-2:]
        if h1!=h or w1!=w:
            ratio_diff = min(abs(h1/w1-h/w),abs(w1/h1-w/h))
            assert ratio_diff<acceptable_ratio_diff, f"pred and gt aspect ratios deviate too much. found pred.shape: {pred.shape}, gt.shape: {gt.shape}"
            pred = torch.nn.functional.interpolate(pred,(h,w),mode="nearest")

    assert len(pred.shape)==len(gt.shape)==3 or len(pred.shape)==len(gt.shape)==4, "pred and gt must be 3D or 4D torch tensors. found pred.shape: "+str(pred.shape)+", gt.shape: "+str(gt.shape)
    assert pred.shape[-1]==gt.shape[-1] and pred.shape[-2]==gt.shape[-2], "pred and gt must have the same spatial dimensions. found pred.shape: "+str(pred.shape)+", gt.shape: "+str(gt.shape)
    if isinstance(pred,torch.Tensor):
        assert isinstance(gt,torch.Tensor), "gt must be a torch tensor if pred is a torch tensor"
        assert (mask is None) or isinstance(mask,torch.Tensor), "mask must be a torch tensor"
    else:
        assert isinstance(pred,np.ndarray), "pred must be a numpy array"
        assert isinstance(gt,np.ndarray), "gt must be a numpy array"
        assert (mask is None) or isinstance(mask,np.ndarray), "mask must be a numpy array"
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        if mask is not None:
            mask = torch.tensor(mask)
    if len(pred.shape)==len(gt.shape)==3:
        was_single = True
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    else:
        was_single = False
    assert pred.shape[1]==gt.shape[1]==1, f"pred and gt must be 1-channel, found pred.shape: {pred.shape}, gt.shape: {gt.shape}"
    assert len(pred.shape)==len(gt.shape)==4, "batched_metrics expects 3D or 4D torch tensors"
    bs = pred.shape[0]
    assert gt.shape[0]==bs, f"pred and gt must have the same batch size. Found pred.shape={pred.shape}, gt.shape={gt.shape}"
    metric_dict = {"iou": partial(standard_iou,ignore_zero=ignore_zero),
                   "hiou": partial(hungarian_iou,ignore_zero=ignore_zero),
                   "ari": adjusted_rand_score_stable}
    metrics = list(metric_dict.keys())
    #has to be defined inline for ab to be implicitly passed

    #metric_dict = {k: handle_empty(v) for k,v in metric_dict.items()}
    out = {metric: [] for metric in metrics}
    for i in range(bs):
        pred_i,gt_i = metric_preprocess(pred[i],gt[i],mask=mask[i] if mask is not None else None)
        for metric in metrics:
            out[metric].append(metric_dict[metric](pred_i,gt_i))
    if was_single:
        for metric in metrics:
            out[metric] = out[metric][0]
    if reduce_to_mean:
        for metric in metrics:
            out[metric] = np.mean(out[metric])
    return out

def adjusted_rand_score_stable(pred,gt):
    tp,fp,fn,tn = get_TP_FP_FN_TN(pred,gt)
    return binary_ari(tp,fp,fn,tn)

def handle_empty(metric_func):
    def wrapped(pred,gt,*args,**kwargs):
        if len(gt)==0 and len(pred)==0:
            return 1.0
        elif len(gt)==0 or len(pred)==0:
            return 0.0
        else:
            return metric_func(pred,gt,*args,**kwargs)
    return wrapped

def metric_preprocess(pred,gt,mask=None):
    assert isinstance(gt,np.ndarray) or isinstance(gt,torch.Tensor), "gt must be a torch tensor or numpy array"
    assert isinstance(pred,np.ndarray) or isinstance(pred,torch.Tensor), "pred must be a torch tensor or numpy array"
    if isinstance(gt,torch.Tensor):
        gt = gt.cpu().detach().numpy()
    if isinstance(pred,torch.Tensor):
        pred = pred.cpu().detach().numpy()
    if mask is None:
        gt = gt.flatten()
        pred = pred.flatten()
    else:
        if isinstance(mask,torch.Tensor):
            mask = mask.cpu().detach().numpy()>0.5
        gt = gt[mask]
        pred = pred[mask]
    return pred,gt

def extend_shorter_vector(vec1,vec2,fill_value=0,min_len=None):
    new_len = max(len(vec1),len(vec2))
    if min_len is not None:
        new_len = max(new_len,min_len)
    if len(vec1)<new_len:
        vec1 = np.concatenate([vec1,(fill_value*np.ones(new_len-len(vec1))).astype(vec1.dtype)])
    if len(vec2)<new_len:
        vec2 = np.concatenate([vec2,(fill_value*np.ones(new_len-len(vec2))).astype(vec2.dtype)])
    return vec1,vec2

def lsa_no_warning(mat, maximize=True):
    if mat.shape==(1,1):
        unpad = True
        mat_new = np.zeros((2,2))
        mat_new[0,0] = mat[0,0]
        mat = mat_new
    else:
        unpad = False
    assignment = linear_sum_assignment(mat, maximize=maximize)
    if unpad:
        a1,a2 = assignment
        assignment = (a1[:1],a2[:1])
    return assignment

def hungarian_iou(pred,gt,ignore_zero=False,match_zero=False,return_assignment=False):
    uq_gt,gt,conf_rowsum = np.unique(gt,return_inverse=True,return_counts=True)
    uq_pred  ,pred  ,conf_colsum = np.unique(pred,return_inverse=True,return_counts=True)

    conf_rowsum,conf_colsum = extend_shorter_vector(conf_rowsum,conf_colsum)
    uq_gt,uq_pred = extend_shorter_vector(uq_gt,uq_pred,fill_value=-1)

    conf_rowsum,conf_colsum = conf_rowsum[:,None],conf_colsum[None,:]
    if len(uq_gt)==1 and len(uq_pred)==1:
        intersection = np.array([[1]])
    else:
        intersection = confusion_matrix(gt, pred)
    union = conf_rowsum + conf_colsum - intersection
    iou_hungarian_mat = intersection / union
    iou_hungarian_mat_for_lsa = iou_hungarian_mat.copy()
    if match_zero:
        #force optimal assignment to match zero with zero, if it is present in both gt and pred
        iou_hungarian_mat_for_lsa[uq_gt==0,:         ] = 0
        iou_hungarian_mat_for_lsa[:       ,uq_pred==0] = 0
        iou_hungarian_mat_for_lsa[uq_gt==0,uq_pred==0] = 1

    assignment = lsa_no_warning(iou_hungarian_mat_for_lsa, maximize=True)

    assign_gt = uq_gt[assignment[0]]
    assign_pred = uq_pred[assignment[1]]
    iou_per_assignment = iou_hungarian_mat[assignment[0],assignment[1]]
    
    #fix cases where 0 was matched with non-zero. 
    #Only happens when exact one of gt or pred has 0
    if match_zero and ((np.sum(uq_gt==0)+np.sum(uq_pred==0))==1):
        if 0 in uq_gt.tolist():
            z = np.flatnonzero(uq_gt==0).item()
        else:
            z = np.flatnonzero(uq_pred==0).item()
        z_p,z_t = assign_pred[z],assign_gt[z]
        if z_t>=0 and z_p>=0:
            #make each of the matches with 0 instead have a dummy match with -1. adjust each array accordingly
            assign_pred,assign_gt,iou_per_assignment = assign_pred.tolist(),assign_gt.tolist(),iou_per_assignment.tolist()
            iou_per_assignment = iou_per_assignment[:z]+iou_per_assignment[z+1:]+[0.0,0.0]
            assign_pred = assign_pred[:z]+assign_pred[z+1:]+[z_p,-1]
            assign_gt = assign_gt[:z]+assign_gt[z+1:]+[-1,z_t]
            assign_pred,assign_gt,iou_per_assignment = np.array(assign_pred),np.array(assign_gt),np.array(iou_per_assignment)
            
    #remove matches which have ignore_idx or dummy (-1) as both gt and pred
    mask = np.logical_or(assign_pred!=-1,assign_gt!=-1)
    if ignore_zero:
         mask = np.logical_and(mask,assign_gt!=0)
    if np.sum(mask)==0:
        #handle edge cases where no valid matches are found
        if match_zero:
            if (0 in uq_gt) and (0 in uq_pred):
                val = 1.0
            else:
                val = 0.0
        else:
            val = 1.0
    else:
        val = np.mean(iou_per_assignment[mask])
    if return_assignment:
        out = {"val":val, 
               "assign_gt": assign_gt, 
               "assign_pred": assign_pred,
               "iou_per_assignment": iou_per_assignment}
    else:
        out = val
    return out
    
def standard_iou(pred,gt,ignore_zero=False,reduce_classes=True):
    num_classes = max(gt.max(),pred.max())+1
    if gt.dtype==bool:
        gt = gt.astype(int)
    if pred.dtype==bool:
        pred = pred.astype(int)
    if num_classes==1:
        return 1.0
    bins = np.arange(num_classes + 1)
    intersection = np.histogram(gt[pred==gt], bins=bins)[0]
    area_pred    = np.histogram(pred,                 bins=bins)[0]
    area_gt  = np.histogram(gt,               bins=bins)[0]
    union = area_pred + area_gt - intersection
    if ignore_zero:
        union[0] = 0
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

def get_TP_FP_FN_TN(pred,gt):
    TP = np.sum(gt&pred).astype(int)
    FP = np.sum(~gt&pred).astype(int)
    FN = np.sum(gt&~pred).astype(int)
    TN = np.sum(~gt&~pred).astype(int)
    return TP,FP,FN,TN

def binary_sensitivity(TP,FP,FN,TN):
    if TP+FN==0:
        return 1.0
    else:
        return TP/(TP+FN)

def collective_insight(pred,gt):
    assert gt.max()<=1 and pred.max()<=1
    n_gt = gt.shape[-1]
    n_pred = pred.shape[-1]

    measures = {"combined_sensitivity": float("nan"),
                "maximum_dice_matching": float("nan"),
                "diversity_agreement": float("nan")}
                
    TP,FP,FN,TN = get_TP_FP_FN_TN(gt.any(-1),pred.any(-1))
    measures["combined_sensitivity"] = binary_sensitivity(TP,FP,FN,TN)

    dice_mat = np.zeros((n_gt,n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            TP,FP,FN,TN = get_TP_FP_FN_TN(gt[:,:,i],pred[:,:,j])
            dice_mat[i,j] = binary_dice(TP,FP,FN,TN)
    measures["maximum_dice_matching"] = dice_mat.max(axis=1).mean()

    variance_mat_pred = np.var(pred.reshape(-1,n_pred)[:,:,None].astype(int)-
                               pred.reshape(-1,n_pred)[:,None,:].astype(int),axis=0)
    variance_mat_gt = np.var(gt.reshape(-1,n_gt)[:,:,None].astype(int)-
                             gt.reshape(-1,n_gt)[:,None,:].astype(int),axis=0)
    V_pred_min = variance_mat_pred.min()
    V_pred_max = variance_mat_pred.max()
    V_gt_min = variance_mat_gt.min()
    V_gt_max = variance_mat_gt.max()
    delta_max = abs(V_pred_max-V_gt_max)
    delta_min = abs(V_pred_min-V_gt_min)
    measures["diversity_agreement"] = 1-(delta_max+delta_min)/2
    multiplied = np.prod([v for v in measures.values()])
    added = np.sum([v for v in measures.values()])
    measures["collective_insight"] = 3*multiplied/added
    return measures

def binary_iou(TP,FP,FN,TN):
    if TP+FP+FN==0:
        return 1.0
    else:
        return TP/(TP+FP+FN)

def binary_dice(TP,FP,FN,TN):
    if TP+FP+FN==0:
        return 1.0
    else:
        return 2*TP/(2*TP+FP+FN)

def generalized_energy_distance(pred,gt,dist=lambda *x: 1-binary_iou(*x),skip_same=False):
    n_gt = gt.shape[-1]
    n_pred = pred.shape[-1]
    dist_pred_gt = np.zeros((n_gt,n_pred))
    cross_mat = []
    for i in range(n_gt):
        cross_mat.append([])
        for j in range(n_pred):
            TP,FP,FN,TN = get_TP_FP_FN_TN(gt[:,:,i],pred[:,:,j])
            cross_mat[-1].append((TP,FP,FN,TN))
            dist_pred_gt[i,j] = dist(TP,FP,FN,TN)
    dist_gt = {}
    for i in range(n_gt):
        for j in range(i*skip_same,n_gt):
            TP,FP,FN,TN = get_TP_FP_FN_TN(gt[:,:,i],gt[:,:,j])
            
            dist_gt[(i,j)] = dist(TP,FP,FN,TN)
    dist_pred = {}
    for i in range(n_pred):
        for j in range(i*skip_same,n_pred):
            TP,FP,FN,TN = get_TP_FP_FN_TN(pred[:,:,i],pred[:,:,j])
            dist_pred[(i,j)] = dist(TP,FP,FN,TN)
    expected_gt = np.array(list(dist_gt.values())).mean()
    expected_pred = np.array(list(dist_pred.values())).mean()
    ged = 2*dist_pred_gt.mean()-expected_gt-expected_pred
    return ged, cross_mat

def binary_ari(tp,fp,fn,tn):
    if fp==0 and fn==0:
        out = 1.0
    else:
        out = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return out

shorthand_dict = {"combined_sensitivity": "S_c",
                    "maximum_dice_matching": "D_max",
                    "diversity_agreement": "D_a",
                    "collective_insight": "CI",
                    "generalized_energy_distance": "GED"}

def hiou_perm(TP,FP,FN,TN):
    """returns the new TP,FP,FN,TN after flipping the prediction in a binary setting"""
    return FP,TP,TN,FN

def get_ambiguous_metrics(pred,gt,shorthand=True,reduce_to_mean=True):
    """returns a dictionary of metrics for binary ambiguous segmentation"""
    assert isinstance(gt,(np.ndarray,list,dict)), "gt must be a numpy array, list of didx strings or dict (with field gts_didx)"
    if not isinstance(gt,np.ndarray):
        if isinstance(gt,dict):
            gt = gt["gts_didx"]
        assert len(gt)>0, "gt must be a non-empty list of didx strings or a numpy array"
        assert all([isinstance(gt_i,str) for gt_i in gt]), "If gt is a list, it must be a list of strings or dicts. Found types: "+str([type(gt_i) for gt_i in gt])
        max_hw = max(pred.shape[-2:])
        gt = np.concatenate([load_raw_image_label(gt_i,max_hw)[1] for gt_i in gt],axis=2)
    assert isinstance(pred,np.ndarray), "pred must be a numpy array"
    assert len(gt.shape)==3, "gt must be a 3D numpy array in (H,W,C_gt) format"
    assert len(pred.shape)==3, "pred must be a 3D numpy array in (H,W,C_pred) format"
    assert gt.shape[0]==pred.shape[0], "gt and pred must have the same height"
    assert gt.shape[1]==pred.shape[1], "gt and pred must have the same width"
    if gt.shape[2]>10 or pred.shape[2]>10:
        pass#raise ValueError("gt and pred must have at most 10 channels. This is a safety measure to prevent accidental misuse (e.g. permutation).")
    measures = collective_insight(pred,gt)
    ged, cross_mat = generalized_energy_distance(pred,gt)
    measures["generalized_energy_distance"] = ged
    if shorthand:
        measures = {shorthand_dict[k]:v for k,v in measures.items()}
    measures["iou"] = [binary_iou(*cm_ij) for cm_ij in sum(cross_mat,[])]
    measures["ari"] = [binary_ari(*cm_ij) for cm_ij in sum(cross_mat,[])]
    measures["hiou"] = [max(binary_iou(*cm_ij),binary_iou(*hiou_perm(*cm_ij))) for cm_ij in sum(cross_mat,[])]
    if reduce_to_mean:
        for k,v in measures.items():
            if isinstance(v,list):
                measures[k] = np.mean(v)
    return measures