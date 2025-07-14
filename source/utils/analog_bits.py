import numpy as np
import torch
from source.utils.mixed import tensor_info
import warnings
from jlc import nc
"""class AnalogBits(object):
    def __init__(self,num_bits=8,
                 shuffle_zero=False,
                 shuffle=False,
                 permanent_seed=None,
                 bit_dim=1,
                 batch_has_different_seed=False,
                 onehot=False,
                 padding_idx=255):
        self.onehot = onehot
        self.bit_dim = bit_dim
        self.num_bits = num_bits
        self.num_classes = 2**num_bits
        self.padding_idx = padding_idx
        if num_bits>8:
            raise NotImplementedError("num_bits>8 not implemented")
        self.shuffle_zero = shuffle_zero
        self.shuffle = shuffle
        self.permanent_seed = permanent_seed
        self.batch_has_different_seed = batch_has_different_seed
        if batch_has_different_seed:
            assert permanent_seed is None
        if permanent_seed is not None:
            np.random.seed(permanent_seed)
            if shuffle_zero:
                self.perm = np.random.permutation(self.num_classes)
            else:
                self.perm = np.concatenate([[0],np.random.permutation(self.num_classes-1)+1])
        else:
            self.perm = None
            
    def get_perm(self,inv=False):
        if self.shuffle:
            if self.perm is None:
                if self.shuffle_zero:
                    perm = np.random.permutation(2**self.num_bits)
                else:
                    perm = np.concatenate([[0],np.random.permutation(2**self.num_bits-1)+1])
            else:
                perm = self.perm
        else:
            perm = np.arange(2**self.num_bits)
        if inv:
            perm = np.argsort(perm)
        return perm
    
    def int2pad(self,x):
        return x==self.padding_idx
    
    def int2bit(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        assert x.shape[self.bit_dim]==1
        x[self.int2pad(x)] = 0
        assert x.max()<=self.num_classes
        if self.shuffle:
            x = self.get_perm(inv=False)[x]
        
        if self.onehot:
            x_new = np.zeros([x.size,self.num_classes])
            x_new[np.arange(x.size),x.flatten()] = 1
            x_new = x_new.reshape(list(x.shape)+[self.num_classes])
            transpose_list = list(range(len(x_new.shape)))
            transpose_list[-1],transpose_list[self.bit_dim] = self.bit_dim,-1
            x = x_new.transpose(transpose_list).squeeze(-1).astype(np.float32)
        else:
            x = np.unpackbits(x.astype(np.uint8),axis=self.bit_dim,count=self.num_bits,bitorder="little").astype(np.float32)*2-1

        if was_torch:
            x = torch.from_numpy(x).to(device)
        return x
        
    def bit2int(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        if self.onehot:
            assert x.shape[self.bit_dim]==self.num_classes, "x.shape: "+str(x.shape)+", self.num_classes: "+str(self.num_classes)+", self.bit_dim: "+str(self.bit_dim)
        else:
            assert x.shape[self.bit_dim]==self.num_bits, "x.shape: "+str(x.shape)+", self.num_bits: "+str(self.num_bits)+", self.bit_dim: "+str(self.bit_dim)
        #convert to ints if necessary
        if x.dtype in [np.float32,np.float64]:
            x = (x>0).astype(np.uint8)
        if self.onehot:
            x = np.argmax(x,axis=self.bit_dim,keepdims=True)
        else:
            x = np.packbits(x,axis=self.bit_dim,bitorder="little")
        if self.shuffle:
            x = self.get_perm(inv=True)[x]
        if was_torch:
            x = torch.from_numpy(x).to(device)
        return x

    def likelihood(self,x,x_gt):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            x_gt = x_gt.cpu().detach().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        if self.onehot:
            assert x.shape[self.bit_dim]==self.num_classes, "x.shape: "+str(x.shape)+", self.num_classes: "+str(self.num_classes)+", self.bit_dim: "+str(self.bit_dim)
        else:
            assert x.shape[self.bit_dim]==self.num_bits, "x.shape: "+str(x.shape)+", self.num_bits: "+str(self.num_bits)+", self.bit_dim: "+str(self.bit_dim)
        if self.onehot:
            raise NotImplementedError("onehot likelihood not implemented")
        else:
            assert np.abs(x_gt).max()<=1.0, f"Expected x_gt to be in bit form in the range [-1,1], got {x_gt.min()} to {x_gt.max()}"
            assert x_gt.shape[self.bit_dim]==self.num_bits, "x_gt.shape: "+str(x_gt.shape)+", self.num_bits: "+str(self.num_bits)+", self.bit_dim: "+str(self.bit_dim)
            likelihood = np.prod(1-0.5*np.abs(x_gt-x),axis=self.bit_dim,keepdims=True)
        if was_torch:
            likelihood = torch.from_numpy(likelihood).to(device)
        return likelihood

    def bit2prob(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        if self.onehot:
            assert x.shape[self.bit_dim]==self.num_classes, "x.shape: "+str(x.shape)+", self.num_classes: "+str(self.num_classes)+", self.bit_dim: "+str(self.bit_dim)
        else:
            assert x.shape[self.bit_dim]==self.num_bits, "x.shape: "+str(x.shape)+", self.num_bits: "+str(self.num_bits)+", self.bit_dim: "+str(self.bit_dim)
        if self.onehot:
            onehot = x
        else:
            onehot_shape = list(x.shape)
            onehot_shape[self.bit_dim] = self.num_classes
            onehot = np.zeros(onehot_shape)
            for i in range(self.num_classes):
                pure_bits = self.int2bit(np.array([i]).reshape(*[1 for _ in range(len(x.shape))]))
                onehot[:,i] = np.prod(1-0.5*np.abs(pure_bits-x),axis=self.bit_dim)
        if was_torch:
            onehot = torch.from_numpy(onehot).to(device)
        return onehot"""

def ab_bit2prob(x,num_bits=6,onehot=False,padding_idx=-1,bit_dim=1):
    if num_bits>8:
        warnings.warn("ab_bit2prob will create a (H,W,2**num_bits) tensor which may be very large when num_bits>8. Consider using ab_bit2prob_idx instead.")
    num_classes = 2**num_bits
    if isinstance(x,torch.Tensor):
        device = x.device
        x = x.cpu().detach().numpy()
        was_torch = True
    else:
        was_torch = False
    assert isinstance(x,np.ndarray)
    assert len(x.shape)>=bit_dim+1
    if onehot:
        assert x.shape[bit_dim]==num_classes, "x.shape: "+str(x.shape)+", num_classes: "+str(num_classes)+", bit_dim: "+str(bit_dim)
    else:
        assert x.shape[bit_dim]==num_bits, "x.shape: "+str(x.shape)+", num_bits: "+str(num_bits)+", bit_dim: "+str(bit_dim)
    if onehot:
        onehot = x
    else:
        onehot_shape = list(x.shape)
        onehot_shape[bit_dim] = num_classes
        onehot = np.zeros(onehot_shape)
        for i in range(num_classes):
            pure_bits = ab_int2bit(np.array([i]).reshape(*[1 for _ in range(len(x.shape))]),
                                      num_bits=num_bits,
                                      onehot=False,
                                      padding_idx=padding_idx,
                                      bit_dim=bit_dim)
            onehot[:,i] = np.prod(1-0.5*np.abs(pure_bits-x),axis=bit_dim)
    if was_torch:
        onehot = torch.from_numpy(onehot).to(device)
    return onehot

def ab_bit2prob_idx(x,idx,keepdims=False,num_bits=6,onehot=False,padding_idx=-1,bit_dim=1):
    num_classes = 2**num_bits
    if isinstance(x,torch.Tensor):
        device = x.device
        x = x.cpu().detach().numpy()
        was_torch = True
    else:
        was_torch = False
    assert isinstance(x,np.ndarray)
    assert len(x.shape)>=bit_dim+1
    if onehot:
        assert x.shape[bit_dim]==num_classes, "x.shape: "+str(x.shape)+", num_classes: "+str(num_classes)+", bit_dim: "+str(bit_dim)
    else:
        assert x.shape[bit_dim]==num_bits, "x.shape: "+str(x.shape)+", num_bits: "+str(num_bits)+", bit_dim: "+str(bit_dim)
    if onehot:
        slicer = [slice(None) for _ in range(len(x.shape))]
        slicer[bit_dim] = slice(idx,idx+1) if keepdims else idx
        prob_idx = x[slicer]
    else:
        pure_bits = ab_int2bit(np.array([idx]).reshape(*[1 for _ in range(len(x.shape))]),
                                    num_bits=num_bits,
                                    onehot=False,
                                    padding_idx=padding_idx,
                                    bit_dim=bit_dim)
        prob_idx = np.prod(1-0.5*np.abs(pure_bits-x),axis=bit_dim,keepdims=keepdims)
    if was_torch:
        prob_idx = torch.from_numpy(prob_idx).to(device)
    return prob_idx

def ab_bit2color(x,pallete=None,num_bits=6,onehot=False,padding_idx=-1,bit_dim=1):
    """Converts a bit representation to a color representation using a palette, 
    in a memory-efficient way, by never storing the full probability array in case it is large.
    """
    if pallete is None:
        pallete = np.concatenate([np.array([[0,0,0]]),nc.largest_colors],axis=0)
    num_classes = 2**num_bits
    if isinstance(x,torch.Tensor):
        device = x.device
        x = x.cpu().detach().numpy()
        was_torch = True
    else:
        was_torch = False
    assert isinstance(x,np.ndarray)
    assert len(x.shape)>=bit_dim+1
    if onehot:
        assert x.shape[bit_dim]==num_classes, "x.shape: "+str(x.shape)+", num_classes: "+str(num_classes)+", bit_dim: "+str(bit_dim)
    else:
        assert x.shape[bit_dim]==num_bits, "x.shape: "+str(x.shape)+", num_bits: "+str(num_bits)+", bit_dim: "+str(bit_dim)
    
    color_array_shape = list(x.shape)
    color_array_shape[bit_dim] = 3
    color_array = np.zeros(color_array_shape,dtype=np.float32)
    color_shape = [1 for _ in range(len(x.shape))]
    color_shape[bit_dim] = 3
    if onehot:
        for i in range(num_classes):
            color_reshaped = pallete[i % len(pallete)].reshape(color_shape)/255
            color_array += x[i:i+1]*color_reshaped
    else:
        for i in range(num_classes):
            pure_bits = ab_int2bit(np.array([i]).reshape(*[1 for _ in range(len(x.shape))]),
                                      num_bits=num_bits,
                                      onehot=False,
                                      padding_idx=padding_idx,
                                      bit_dim=bit_dim)
            color_reshaped = pallete[i % len(pallete)].reshape(color_shape)/255
            prob_idx = np.prod(1-0.5*np.abs(pure_bits-x),axis=bit_dim,keepdims=True)
            color_array += color_reshaped * prob_idx
    
    if was_torch:
        color_array = torch.from_numpy(color_array).to(device)
    return color_array

def ab_likelihood(x,x_gt,num_bits=6,onehot=False,padding_idx=-1,bit_dim=1):
    """ Converts a bit representation to a likelihood of the bit being correct."""
    num_classes = 2**num_bits
    if isinstance(x,torch.Tensor):
        device = x.device
        x = x.cpu().detach().numpy()
        x_gt = x_gt.cpu().detach().numpy()
        was_torch = True
    else:
        was_torch = False
    assert isinstance(x,np.ndarray)
    assert len(x.shape)>=bit_dim+1
    if onehot:
        assert x.shape[bit_dim]==num_classes, "x.shape: "+str(x.shape)+", num_classes: "+str(num_classes)+", bit_dim: "+str(bit_dim)
    else:
        assert x.shape[bit_dim]==num_bits, "x.shape: "+str(x.shape)+", num_bits: "+str(num_bits)+", bit_dim: "+str(bit_dim)
    if onehot:
        likelihood = (x*x_gt).sum(axis=bit_dim,keepdims=True)
    else:
        if not np.abs(x_gt).max()<=1.0:#, f"Expected x_gt to be in bit form in the range [-1,1], got {x_gt.min()} to {x_gt.max()}"
            print("TENSOR INFO:\n",tensor_info(x_gt))
            assert 0
        likelihood = np.prod(1-0.5*np.abs(x_gt-x),axis=bit_dim,keepdims=True)
    if was_torch:
        likelihood = torch.from_numpy(likelihood).to(device)
    return likelihood

def ab_bit2int(x,num_bits=6,onehot=False,padding_idx=-1,bit_dim=1):
    num_classes = 2**num_bits
    if isinstance(x,torch.Tensor):
        device = x.device
        x = x.cpu().detach().numpy()
        was_torch = True
    else:
        was_torch = False
    assert isinstance(x,np.ndarray)
    assert len(x.shape)>=bit_dim+1
    if onehot:
        assert x.shape[bit_dim]==num_classes, "x.shape: "+str(x.shape)+", num_classes: "+str(num_classes)+", bit_dim: "+str(bit_dim)
    else:
        assert x.shape[bit_dim]==num_bits, "x.shape: "+str(x.shape)+", num_bits: "+str(num_bits)+", bit_dim: "+str(bit_dim)
    #convert to ints if necessary
    if x.dtype in [np.float32,np.float64,np.float16]:
        x = (x>0).astype(np.uint8)
    if onehot:
        x = np.argmax(x,axis=bit_dim,keepdims=True).astype(np.uint8)
    else:
        x = np.packbits(x,axis=bit_dim,bitorder="little")
        if x.shape[bit_dim]>1:
            mult_shape = [1 for _ in range(len(x.shape))]
            mult_shape[bit_dim] = x.shape[bit_dim]
            multiplier = np.array([256**i for i in range(x.shape[bit_dim])]).reshape(mult_shape)
            x = np.sum(x*multiplier,axis=bit_dim,keepdims=True).astype(int)
    if was_torch:
        x = torch.from_numpy(x).to(device)
    return x

def ab_int2bit(x,num_bits=6,onehot=False,padding_idx=-1,bit_dim=1):
    num_classes = 2**num_bits
    if isinstance(x,torch.Tensor):
        device = x.device
        x = x.clone().cpu().detach().numpy()
        was_torch = True
    else:
        was_torch = False
        x = x.copy()
    assert isinstance(x,np.ndarray)
    assert len(x.shape)>=bit_dim+1
    assert x.shape[bit_dim]==1
    x[x==padding_idx] = 0
    assert x.max()<=num_classes
    
    if onehot:
        x_new = np.zeros([x.size,num_classes])
        x_new[np.arange(x.size),x.flatten()] = 1
        x_new = x_new.reshape(list(x.shape)+[num_classes])
        transpose_list = list(range(len(x_new.shape)))
        transpose_list[-1],transpose_list[bit_dim] = bit_dim,-1
        x = x_new.transpose(transpose_list).squeeze(-1).astype(np.float32)
    else:
        x = np.unpackbits(x.astype(np.uint8),axis=bit_dim,count=num_bits,bitorder="little").astype(np.float32)*2-1

    if was_torch:
        x = torch.from_numpy(x).to(device)
    return x

def ab_kwargs_from_args(args):
    return {
        "num_bits": args.diff_channels,
        "onehot": args.onehot,
        "padding_idx": args.padding_idx,
        "bit_dim": 1
    }