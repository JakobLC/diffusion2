import sys
import torch
import scipy.ndimage as nd
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

class AnalogBits(object):
    def __init__(self,num_bits=8,
                 shuffle_zero=False,
                 shuffle=True,
                 permanent_seed=None,
                 bit_dim=1,
                 batch_has_different_seed=False):
        self.bit_dim = bit_dim
        self.num_bits = num_bits
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
                self.perm = np.random.permutation(2**num_bits)
            else:
                self.perm = np.concatenate([[0],np.random.permutation(2**num_bits-1)+1])
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
        
    def int2bit(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        assert x.shape[self.bit_dim]==1
        if self.shuffle:
            x = self.get_perm(inv=False)[x]
        #convert to binary
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
        assert x.shape[self.bit_dim]==self.num_bits
        #convert to ints if necessary
        if x.dtype in [np.float32,np.float64]:
            x = (x>0).astype(np.uint8)
        x = np.packbits(x,axis=self.bit_dim,bitorder="little")
        if self.shuffle:
            x = self.get_perm(inv=True)[x]
        if was_torch:
            x = torch.from_numpy(x).to(device)
        return x
        

class CatBallDataset(torch.utils.data.Dataset):
    def __init__(self,
                 size : int = 64,
                 dtype : str ="uint8",
                 dataset_len : int = 1000,
                 background_is_zero : bool = True,
                 num_balls : list = list(range(1,10)),
                 max_classes : int = 8,
                 seed_translation : int = 0):
        assert dtype in ["float","double","uint8"]
        self.dtype = dtype
        self.dataset_len = dataset_len
        self.size = size
        self.background_is_zero = background_is_zero
        if isinstance(num_balls,int):
            num_balls = [num_balls]
        self.num_balls = num_balls
        self.max_classes = max_classes
        self.x,self.y = np.meshgrid(range(self.size),range(self.size))
        self.seed_translation = seed_translation
        
    def get_image(self,item,noise_coef=0.1,std=0.05):
        noise = np.random.randn(*item.shape)
        foreground = (item>0).astype(float)
        image = (1-noise_coef)*foreground+noise_coef*noise
        image = gaussian_filter(image,sigma=std*self.size)
        return image*2-1
    
    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        np.random.seed(idx+self.seed_translation)
        R = []
        CX = []
        CY = []
        item = np.zeros((self.size,self.size),dtype=self.dtype)
        nb = np.random.choice(self.num_balls)
        for i in range(nb):
            cx = self.size*np.random.rand()
            cy = self.size*np.random.rand()
            max_r = min([cx,cy,self.size-cx,self.size-cy])
            r = max_r*np.random.rand()
            dist = r-((self.x-cx)**2+(self.y-cy)**2)**0.5
            item[dist>=0] = np.random.randint(1,self.max_classes+1)
            R.append(r)
            CX.append(cx)
            CY.append(cy)
        info = {"r": R,"cx": CX,"cy": CY, "nb": nb}
        info["image"] = torch.from_numpy(self.get_image(item)).unsqueeze(0)
        item = torch.from_numpy(item).unsqueeze(0)
        return item,info

def custom_collate_with_info(original_batch):
    n = len(original_batch[0])
    normal_batch = []
    for i in range(n):
        list_of_items = [item[i] for item in original_batch]
        if i+1==n:
            info = list_of_items
        else:
            normal_batch.append(torch.stack(list_of_items,axis=0))
    return *normal_batch,info

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0: display batch")
        import jlc
        import matplotlib.pyplot as plt
        dataloader = torch.utils.data.DataLoader(CatBallDataset(size=64,dataset_len=10),batch_size=10,shuffle=False,collate_fn=custom_collate_with_info)
        x,info = next(iter(dataloader))
        im = torch.stack([info_i["image"] for info_i in info],dim=0)
        images = torch.cat((x,im),dim=0).clamp(0,1)
        jlc.montage(images)
        plt.show()
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()