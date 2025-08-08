import numpy as np
import torch
from source.utils.mixed import tensor_info,sam_resize_index
import warnings
from jlc import nc
from sklearn.cluster import KMeans

class AnalogBits(object):
    def __init__(self,
                 args=None,
                 num_bits=6,
                 encoding_type="analog_bits",
                 padding_idx=-1,
                 bit_dim=1):
        if args is not None:
            num_bits = args.diff_channels
            encoding_type = args.encoding_type
            padding_idx = args.padding_idx
            bit_dim = 1

        self.num_bits = num_bits
        self.encoding_type = encoding_type
        self.bit_dim = bit_dim
        self.padding_idx = padding_idx

        self.onehot = self.encoding_type=="onehot"
        self.RGB = self.encoding_type=="RGB"

        self.num_classes = 2**num_bits if not self.RGB else 125  # RGB encoding has 125 classes (0-124)
        assert encoding_type in ["analog_bits","onehot","RGB"], "encoding_type must be one of ['analog_bits','onehot','RGB'], got "+str(encoding_type)
    
    def bit2prob(self,x):
        if not self.RGB:
            if self.num_bits>8:
                warnings.warn("bit2prob will create a (H,W,2**num_bits) tensor which may be very large when num_bits>8. Consider using ab_bit2prob_idx instead.")
        x, was_torch, device = self.convert_and_assert(x)
        if self.onehot:
            onehot_probs = x
        elif self.RGB:
            onehot_shape = list(x.shape)
            onehot_shape[self.bit_dim] = self.num_classes
            onehot_probs = np.zeros(onehot_shape)
        else:
            onehot_shape = list(x.shape)
            onehot_shape[self.bit_dim] = self.num_classes
            onehot_probs = np.zeros(onehot_shape)
            for i in range(self.num_classes):
                pure_bits = self.int2bit(np.array([i]).reshape(*[1 for _ in range(len(x.shape))]))
                onehot_probs[:,i] = np.prod(1-0.5*np.abs(pure_bits-x),axis=self.bit_dim)
        onehot_probs = self.convert_back(onehot_probs, was_torch, device)
        return onehot_probs

    def bit2prob_idx(self,x,idx,keepdims=False):
        assert not self.RGB, "bit2prob_idx is not implemented for RGB encoding type"
        x, was_torch, device = self.convert_and_assert(x)

        if self.onehot:
            slicer = [slice(None) for _ in range(len(x.shape))]
            slicer[self.bit_dim] = slice(idx,idx+1) if keepdims else idx
            prob_idx = x[slicer]
        else:
            pure_bits = self.int2bit(np.array([idx]).reshape(*[1 for _ in range(len(x.shape))]))
            prob_idx = np.prod(1-0.5*np.abs(pure_bits-x),axis=self.bit_dim,keepdims=keepdims)

        prob_idx = self.convert_back(prob_idx, was_torch, device)
        return prob_idx

    def bit2color(self,x,pallete=None):
        """Converts a bit representation to a color representation using a palette, 
        in a memory-efficient way, by never storing the full probability array in case it is large.
        """
        assert not self.RGB, "bit2color is not implemented for RGB encoding type"
        if pallete is None:
            pallete = np.concatenate([np.array([[0,0,0]]),nc.largest_colors],axis=0)
        x, was_torch, device = self.convert_and_assert(x)
        
        color_array_shape = list(x.shape)
        color_array_shape[self.bit_dim] = 3
        color_array = np.zeros(color_array_shape,dtype=np.float32)
        color_shape = [1 for _ in range(len(x.shape))]
        color_shape[self.bit_dim] = 3
        if self.onehot:
            for i in range(self.num_classes):
                color_reshaped = pallete[i % len(pallete)].reshape(color_shape)/255
                color_array += x[i:i+1]*color_reshaped
        elif self.RGB:
            raise NotImplementedError("bit2color is not implemented for RGB encoding type")
        else:
            for i in range(self.num_classes):
                pure_bits = self.int2bit(np.array([i]).reshape(*[1 for _ in range(len(x.shape))]))
                color_reshaped = pallete[i % len(pallete)].reshape(color_shape)/255
                prob_idx = np.prod(1-0.5*np.abs(pure_bits-x),axis=self.bit_dim,keepdims=True)
                color_array += color_reshaped * prob_idx
        
        color_array = self.convert_back(color_array, was_torch, device)
        return color_array

    def likelihood(self,x,x_gt):
        """ Converts a bit representation to a likelihood of the bit being correct."""

        #assert not self.RGB, "likelihood is not implemented for RGB encoding type"
        x, was_torch, device = self.convert_and_assert(x)
        x_gt, _, _ = self.convert_and_assert(x_gt)

        if self.onehot:
            likelihood = (x*x_gt).sum(axis=self.bit_dim,keepdims=True)
        elif self.RGB:
            likelihood = np.zeros_like(x_gt.sum(axis=self.bit_dim,keepdims=True), dtype=np.float32) #placeholder for RGB likelihood
        else:
            if not np.abs(x_gt).max()<=1.0:#, f"Expected x_gt to be in bit form in the range [-1,1], got {x_gt.min()} to {x_gt.max()}"
                print("TENSOR INFO:\n",tensor_info(x_gt))
                assert 0
            likelihood = np.prod(1-0.5*np.abs(x_gt-x),axis=self.bit_dim,keepdims=True)

        likelihood = self.convert_back(likelihood, was_torch, device)
        return likelihood

    def convert_and_assert(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            was_torch = True
        else:
            device = None
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1

        if self.onehot:
            assert x.shape[self.bit_dim]==self.num_classes, "x.shape: "+str(x.shape)+", num_classes: "+str(self.num_classes)+", bit_dim: "+str(self.bit_dim)
        elif self.RGB:
            assert x.shape[self.bit_dim]==3, "x.shape: "+str(x.shape)+", RGB encoding requires bit_dim=1 and 3 channels"
        else:
            assert x.shape[self.bit_dim]==self.num_bits, "x.shape: "+str(x.shape)+", num_bits: "+str(self.num_bits)+", bit_dim: "+str(self.bit_dim)
        return x, was_torch, device
    
    def convert_back(self,x, was_torch, device):
        if was_torch:
            x = torch.from_numpy(x).to(device)
        return x

    def bit2int(self,x,info=None):
        """ Converts a bit representation to an integer representation."""
        x, was_torch, device = self.convert_and_assert(x)

        if self.onehot:
            x = np.argmax(x,axis=self.bit_dim,keepdims=True).astype(np.uint8)
        elif self.RGB:
            assert info is not None, "Expected info to be provided for RGB encoding"
            x_list = []
            for i in range(x.shape[0]):
                crop_hw = sam_resize_index(*info[i]["imshape"][:2], x[i].shape[-1])
                #x_list.append(np.zeros((1, x[i].shape[-2], x[i].shape[-1]), dtype=np.int32))
                """x_list.append(progressive_dichotomy_module(x[i],
                                                           crop_hw=crop_hw,
                                                           delta=10.0, 
                                                           max_depth=10, 
                                                           min_pixels=20))"""
                x_list.append(palette_baseline_segmentation(x[i],
                                                            min_pixels=20,
                                                            crop_hw=crop_hw,
                                                            padding_idx=self.padding_idx))
            x = np.stack(x_list, axis=0)
        else:
            if x.dtype in [np.float32,np.float64,np.float16]:
                x = (x>0).astype(np.uint8)
            x = np.packbits(x,axis=self.bit_dim,bitorder="little")
            if x.shape[self.bit_dim]>1:
                mult_shape = [1 for _ in range(len(x.shape))]
                mult_shape[self.bit_dim] = x.shape[self.bit_dim]
                multiplier = np.array([256**i for i in range(x.shape[self.bit_dim])]).reshape(mult_shape)
                x = np.sum(x*multiplier,axis=self.bit_dim,keepdims=True).astype(int)

        x = self.convert_back(x, was_torch, device)
        return x

    def int2bit(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.clone().cpu().detach().numpy()
            was_torch = True
        else:
            device = None
            was_torch = False
            x = x.copy()
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        assert x.shape[self.bit_dim]==1
        assert x[x!=self.padding_idx].max()<=self.num_classes, f"Expected x to be in the range [0, {self.num_classes-1}], got {x[x!=self.padding_idx].max()}"
        
        if self.onehot:
            x_new = np.zeros([x.size,self.num_classes])
            x_new[np.arange(x.size),x.flatten()] = 1
            x_new = x_new.reshape(list(x.shape)+[self.num_classes])
            transpose_list = list(range(len(x_new.shape)))
            transpose_list[-1],transpose_list[self.bit_dim] = self.bit_dim,-1
            y = x_new.transpose(transpose_list).squeeze(-1).astype(np.float32)
            y[x.repeat(self.num_classes,axis=self.bit_dim)==self.padding_idx] = 0
        elif self.RGB:
            assert x.max()<=124, "RGB max value allowed is 2**5-1=124, got "+str(x.max())
            y = UniGS_colormap_encoding(x,axis=self.bit_dim)
            y[x.repeat(3,axis=self.bit_dim)==self.padding_idx] = 0
        else:
            y = np.unpackbits(x.astype(np.uint8),
                              axis=self.bit_dim,
                              count=self.num_bits,
                              bitorder="little").astype(np.float32)*2-1
            y[x.repeat(self.num_bits,axis=self.bit_dim)==self.padding_idx] = 0
        y = self.convert_back(y, was_torch, device)
        return y

def fives_roots(a,axis=1):
    return np.concatenate((a//25,(a%25)//5,a%5),axis=axis)

def UniGS_colormap_encoding(a,padding_idx=-1,axis=1):
    assert a.max()<=124, "colormap_encoding max value allowed is 2**5-1=124, got "+str(a.max())
    assert a.shape[axis]==1, "colormap_encoding expects a shape with a single channel along the specified axis"
    assert is_int_like(a), "colormap_encoding expects an integer-like tensor of class indices in [0,124]."
    roots = fives_roots(a,axis=axis)
    enc = (np.clip(roots*64,0,255).astype(np.float32)*2-1)/255.0
    enc[a.repeat(3,axis=axis)==padding_idx] = padding_idx
    return enc

def is_float_like(x):
    """Check if the input is a float-like tensor."""
    if isinstance(x, torch.Tensor):
        return x.dtype.is_floating_point
    elif isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.floating)
    else:
        return False
    
def is_int_like(x):
    """Check if the input is an int-like tensor."""
    if isinstance(x, torch.Tensor):
        return x.dtype.is_integer
    elif isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.integer)
    else:
        return False
    
def palette_baseline_segmentation(image, min_pixels=20, crop_hw=None, padding_idx=-1):
    """
    Simple palette-based segmentation baseline. Assumes image values in [-1, 1] (RGB).
    
    Args:
        image (np.ndarray): array of shape (3, H, W), values in [-1,1]
        min_pixels (int): Minimum pixel count to keep a region

    Returns:
        np.ndarray: (1, H, W) array of class indices (int)
    """
    assert len(image.shape) == 3 and image.shape[0] == 3, "Expected (3, H, W) image"
    if crop_hw is not None:
        orig_h, orig_w = image.shape[1:3]
        crop_h, crop_w = crop_hw
        image = image[:, :crop_h, :crop_w]
    assert is_float_like(image), "Image must be a float-like tensor with values in [-1, 1], found dtype: "+str(image.dtype)
    assert isinstance(image, np.ndarray), "Expected image to be a numpy array, found type: "+str(type(image))
    C, H, W = image.shape

    # Step 1: scale and threshold RGB
    image_np = ((image+1)*(255.0/2)) # shape (3, H, W)
    palette = np.array([0, 64, 128, 192, 255])
    thresholds = ((palette[:-1] + palette[1:]) / 2).astype(np.uint8)  # [32, 96, 160, 224]

    def quantize(channel):
        out = np.zeros_like(channel, dtype=np.uint8)
        for i, t in enumerate(thresholds):
            out[channel >= t] = palette[i + 1]
        return out

    q_image = np.stack([quantize(image_np[c]) for c in range(3)], axis=0)  # (3, H, W)

    # Step 2: flatten and group pixels by RGB triplet
    flat_rgb = q_image.reshape(3, -1).transpose(1, 0)  # (H*W, 3)
    rgb_tuples, inverse_indices = np.unique(flat_rgb, axis=0, return_inverse=True)

    # Count pixels in each class
    class_pixel_indices = {i: [] for i in inverse_indices}
    for idx, class_id in enumerate(inverse_indices):
        class_pixel_indices[class_id].append(idx)

    # Step 3: assign labels to valid classes
    class_map = np.full((H * W,), -1, dtype=np.int32)  # -1 for invalid pixels
    valid_classes = {}
    label_counter = 0

    for class_id, indices in class_pixel_indices.items():
        if len(indices) >= min_pixels:
            class_map[indices] = label_counter
            valid_classes[class_id] = (label_counter, rgb_tuples[class_id])
            label_counter += 1

    # Step 4: reassign invalid pixels to nearest valid RGB class
    invalid_indices = np.flatnonzero(class_map == -1)
    if len(invalid_indices) > 0 and valid_classes:
        valid_colors = np.stack([v[1] for v in valid_classes.values()])  # shape (N_valid, 3)
        valid_labels = [v[0] for v in valid_classes.values()]

        invalid_colors = flat_rgb[invalid_indices]  # (N_invalid, 3)
        dists = np.linalg.norm(invalid_colors[:, None, :] - valid_colors[None, :, :], axis=2)
        best_matches = dists.argmin(axis=1)
        assigned_labels = np.array(valid_labels)[best_matches]

        class_map[invalid_indices] = assigned_labels
    if crop_hw is not None:
        class_map_uncropped = np.full((1, orig_h, orig_w), fill_value=padding_idx, dtype=np.int32)
        class_map_uncropped[:, :crop_h, :crop_w] = class_map.reshape(1, H, W)
        return class_map_uncropped
    else:
        return class_map.view(1, H, W)

def progressive_dichotomy_module(image, crop_hw=None, padding_idx=-1, 
                                 delta=10.0, max_depth=10, min_pixels=20,
                                 is_255=False):
    """
    Progressive Dichotomy Module using RGB-only features.

    Args:
        image (np.ndarray): RGB image of shape (3, H, W) with values in [-1, 1]
        delta (float): Variance threshold to stop recursive splitting
        max_depth (int): Maximum recursion depth
        min_pixels (int): Minimum region size to attempt further splitting

    Returns:
        torch.Tensor: (1, H, W) image of class indices, where each connected region is given a unique label
    """
    assert image.ndim == 3 and image.shape[0] == 3, "Expected image shape (3, H, W)"
    assert is_float_like(image), "Image must be a float-like tensor with values in [-1, 1], found dtype: "+str(image.dtype)
    assert isinstance(image, np.ndarray), "Expected image to be a numpy array, found type: "+str(type(image))
    image = image.copy()
    if crop_hw is not None:
        orig_h, orig_w = image.shape[1:3]
        crop_h, crop_w = crop_hw
        image = image[:, :crop_h, :crop_w]
    if not is_255:
        image = (image+1)*(255/2)  # Convert to [0, 255] range
    C, H, W = image.shape

    class_map = np.full((H * W,), fill_value=0, dtype=np.int32) 
    is_finished = [False]
    num_splits = {0: 0}  # Track number of splits for each class
    for _ in range(2**max_depth):
        if all(is_finished):
            break
        #smallest unfinished class
        current_label = is_finished.index(False)
        mask_indices = np.where(class_map == current_label)[0]
        if len(mask_indices) < min_pixels or num_splits[current_label] >= max_depth:
            is_finished[current_label] = True
            continue
        cluster_data = image.transpose((1,2,0)).reshape(-1, C)[mask_indices]  # Reshape to (num_pixels, C)
        center = cluster_data.mean(axis=0)
        dist = np.linalg.norm(cluster_data - center, axis=1)
        if dist.mean() < delta:
            is_finished[current_label] = True
            continue

        # Split with KMeans
        kmeans = KMeans(n_clusters=2, n_init=3, max_iter=100, algorithm="elkan")
        labels = kmeans.fit_predict(cluster_data)

        # Assign: one group keeps old label, other gets new one
        keep_old = labels == 0
        new_idx = len(is_finished)

        class_map_indices = mask_indices
        class_map[class_map_indices[keep_old]] = current_label
        class_map[class_map_indices[~keep_old]] = new_idx

        num_splits[current_label] += 1
        num_splits[new_idx] = num_splits[current_label]
        is_finished.append(False)
    class_map = class_map.reshape(1, H, W)  # Convert to (1, H, W)
    if crop_hw is not None:
        class_map_uncropped = np.full((1, orig_h, orig_w), 
                                         fill_value=padding_idx, 
                                         dtype=np.int32)
        class_map_uncropped[:, :crop_h, :crop_w] = class_map
        return class_map_uncropped
    else:
        return class_map 

def ab_kwargs_from_args(args):
    return {
        "num_bits": args.diff_channels,
        "encoding_type": args.encoding_type,
        "padding_idx": args.padding_idx,
        "bit_dim": 1
    }