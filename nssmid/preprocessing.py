import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import importlib 
import os

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

## square-wave ----------------------------------------------------------
def square_wave_loaders(config):
    fpath = "./data/square-wave.pt"
    train_size, test_size = 300, 200
    if os.path.isfile(fpath):
        data = torch.load(fpath)
    else:
        x, y = square_wave(train_size)
        xt, yt = square_wave(test_size, xrand=False)
        data = {"xt":xt, "yt":yt, "x":x, "y":y}
        torch.save(data, fpath)

    train_dataset = Curve(data['x'], data['y'])
    test_dataset  = Curve(data['xt'], data['yt'])
        
    trainLoader = DataLoader(train_dataset,batch_size=config.train_batch_size, shuffle=True, pin_memory=True)

    testLoader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, pin_memory=True)

    return trainLoader, testLoader

def square_wave(size, xrand=True):
        if xrand:
            x = 2*(2*torch.rand(size,1) - 1)
        else:
            x = torch.linspace(-2.0, 2.0, size).reshape((size,1))
        y = torch.zeros((size,1))
        for i in range(size):
            if x[i, 0] <= -1.0:
                y[i, 0] = 1.0
            if x[i, 0] > 0.0 and x[i, 0] <= 1.0:
                y[i, 0] = 1.0
        return x, y

class Curve(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = x.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def getDataLoader(config):
    loaders = {
        'square_wave': square_wave_loaders
    }[config.dataset]
    return loaders(config)

def load_obj(obj_path: str, default_obj_path: str = ''):
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)



class MultiMargin(nn.Module):

    def __init__(self, margin = 0.5):
        super().__init__()
        self.margin = margin 

    def __call__(self, outputs, labels):
        return F.multi_margin_loss(outputs, labels, margin=self.margin)
    
## from https://github.com/araujoalexandre/lipschitz-sll-networks
class Xent(nn.Module):

  def __init__(self, num_classes, offset=3.0/2):
    super().__init__()
    self.criterion = nn.CrossEntropyLoss()
    self.offset = (2 ** 0.5) * offset
    self.temperature = 0.25 
    self.num_classes = num_classes

  def __call__(self, outputs, labels):
    one_hot_labels = F.one_hot(labels, num_classes=self.num_classes)
    offset_outputs = outputs - self.offset * one_hot_labels
    offset_outputs /= self.temperature
    loss = self.criterion(offset_outputs, labels) * self.temperature
    return loss

def empirical_lipschitz(model, x, eps=0.05):

    norms = lambda X: X.view(X.shape[0], -1).norm(dim=1) ** 2
    gam = 0.0
    for r in range(10):
        dx = torch.zeros_like(x)
        dx.uniform_(-eps,eps)
        x.requires_grad = True
        dx.requires_grad = True
        optimizer = torch.optim.Adam([x, dx], lr=1E-1)
        iter, j = 0, 0
        LipMax = 0.0
        while j < 50:
            LipMax_1 = LipMax
            optimizer.zero_grad()
            dy = model(x + dx) - model(x)
            Lip = norms(dy) / (norms(dx) + 1e-6)
            Obj = -Lip.sum()
            Obj.backward()
            optimizer.step()
            LipMax = Lip.max().item()
            iter += 1
            j += 1
            if j >= 5:
                if LipMax < LipMax_1 + 1E-3:  
                    optimizer.param_groups[0]["lr"] /= 10.0
                    j = 0

                if optimizer.param_groups[0]["lr"] <= 1E-5:
                    break
        
        gam = max(gam, np.sqrt(LipMax))

    return gam 

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess_mat_file(dMat : dict, idxMax : int = 100000):
        fs = dMat['fs']
        ts = 1/fs[0][0]
        u = dMat['uTot']
        y = dMat['yTot']
        x = dMat['xTot']
        u = np.reshape(u, (max(u.shape), min(u.shape)))
        y = np.reshape(y, (max(y.shape), min(y.shape)))
        x = np.reshape(x, (max(x.shape), min(x.shape)))
        
        u_torch = torch.from_numpy(u[:idxMax,:]).to(dtype= torch.float32)
        y_torch = torch.from_numpy(y[:idxMax,:]).to(dtype= torch.float32)
        x_torch = torch.from_numpy(x[:idxMax,:]).to(dtype= torch.float32)
        y_torch_dot = (y_torch[1:,:]-y_torch[0:-1,:])/ts
        y_torch_dot = torch.cat([torch.zeros((1,2)),y_torch_dot])

        return u_torch, y_torch, x_torch, y_torch_dot, ts

def preprocess_mat_file_dist(dMat : dict, nx : int, idxMax : int = 100000):
        fs = dMat['fs']
        ts = 1/fs[0][0]
        u = dMat['uTot']
        y = dMat['yTot']
        d = dMat['pTot']
        if u.shape[0]< u.shape[1]: # number of samples is on dimension 1
            u = u.T
        if y.shape[0] < y.shape[1]:
            y = y.T
        if d.shape[0] < d.shape[1]:
            d = d.T

        u_torch = torch.from_numpy(u[:idxMax,:]).to(dtype= torch.float32)
        y_torch = torch.from_numpy(y[:idxMax,:]).to(dtype= torch.float32)
        d_torch = torch.from_numpy(d[:idxMax,:]).to(dtype= torch.float32)

        try :
            dMat['xTot']
            x = np.reshape(x, (max(x.shape), min(x.shape)))
            x_torch = torch.from_numpy(x[:idxMax,:]).to(dtype= torch.float32)
        except:
            x_torch = torch.zeros((y.shape[0], nx), dtype=torch.float32, requires_grad=True)


        u_torch = torch.cat((u_torch, d_torch), dim = 1)
        y_torch_dot = (y_torch[1:,:]-y_torch[0:-1,:])/ts
        y_torch_dot = torch.cat([torch.zeros((1,y.shape[1])),y_torch_dot])

        return u_torch, y_torch, x_torch, y_torch_dot, ts


class SeriesDataset(Dataset):
    def __init__(self, u : torch.Tensor, y: torch.Tensor, x : torch.Tensor, 
                 seq_len : int, ts : float):
        """
            u : input data N_samples x N_channels
            y : output data N_samples x N_channels
            x : hidden_state to be trained or not Nsamples x nx
            seq_len (int) : length of the sequences
            ts (float) : sample time of the time series
        """
       
        self.u = u
        self.y = y
        self.x = x
        if seq_len == 'all':
            self.seq_len = u.shape[0]
        else:
            self.seq_len = seq_len
        self.ts = ts

    def __len__(self):
        '''
            During an epoch every x in self.x is tried as an initial condition x0.
            This leads to very long epochs but is consistent with an epoch being over when all datapoints are used.
        '''
        return self.y.shape[0]-self.seq_len +1 # Learning is done over the whole time series
    
    def __getitem__(self, index):
        """
            Returns 
                u,y, x, x0
        
        """

        return (self.u[index:index+self.seq_len, :], 
                self.y[index:index+self.seq_len, :],
                self.x[index:index+self.seq_len,:],
                self.x[index,:])

    def standardize_(self, a = 0, b = 1):
        # Maximum
        self.u_max, _ = torch.max(self.u,dim=0)
        self.y_max, _ = torch.max(self.y, dim = 0)
        self.x_max, _ = torch.max(self.x, dim = 0)

        # Minimum
        self.u_min, _ = torch.min(self.u,dim=0)
        self.y_min, _ = torch.min(self.y, dim = 0)
        self.x_min, _ = torch.min(self.x, dim = 0)

        if not any(self.u_max==0.):
            self.u = a + (b-a)*(self.u - self.u_min)/(self.u_max - self.u_min).requires_grad_()
        if not any(self.y_max==0.):
            self.y = a + (b-a)*(self.y - self.y_min)/(self.y_max - self.y_min).requires_grad_()
        if not any(self.x_max==0.):
            if self.x.requires_grad:
                self.x = a + (b-a)* (self.x - self.x_min)/(self.x_max - self.x_min).requires_grad_()
            else:
                self.x = a + (b-a)*(self.x - self.x_min)/(self.x_max - self.x_min)





class SeriesDatasetv2(Dataset):
    def __init__(self, u : torch.Tensor, y: torch.Tensor, x : torch.Tensor, 
                 seq_len : int, ts : float):
        """
            u : input data N_samples x N_channels
            y : output data N_samples x N_channels
            x : hidden_state to be trained or not Nsamples x nx
            seq_len (int) : length of the sequences
            ts (float) : sample time of the time series
        """
       
        self.u = u
        self.y = y
        self.x = x
        if seq_len == 'all':
            self.seq_len = u.shape[0]
        else:
            self.seq_len = seq_len
        self.ts = ts

    def __len__(self):
        '''
            During an epoch every x in self.x is tried as an initial condition x0.
            This leads to very long epochs but is consistent with an epoch being over when all datapoints are used.
        '''
        return self.y.shape[0]-self.seq_len +1 # Learning is done over the whole time series
    
    def __getitem__(self, index):
        """
            Returns 
                u,y, x, x0
        
        """

        return (self.u[index:index+self.seq_len, :], 
                self.y[index:index+self.seq_len, :],
                self.x[index:index+self.seq_len,:],
                self.x[index,:])

    def standardize_(self, a = 0, b = 1):
        # Maximum
        self.u_max, _ = torch.max(self.u,dim=0)
        self.y_max, _ = torch.max(self.y, dim = 0)
        self.x_max, _ = torch.max(self.x, dim = 0)

        # Minimum
        self.u_min, _ = torch.min(self.u,dim=0)
        self.y_min, _ = torch.min(self.y, dim = 0)
        self.x_min, _ = torch.min(self.x, dim = 0)

        if not any(self.u_max==0.):
            self.u = a + (b-a)*(self.u - self.u_min)/(self.u_max - self.u_min).requires_grad_()
        if not any(self.y_max==0.):
            self.y = a + (b-a)*(self.y - self.y_min)/(self.y_max - self.y_min).requires_grad_()
        if not any(self.x_max==0.):
            if self.x.requires_grad:
                self.x = a + (b-a)* (self.x - self.x_min)/(self.x_max - self.x_min).requires_grad_()
            else:
                self.x = a + (b-a)*(self.x - self.x_min)/(self.x_max - self.x_min)


class dataSet():
    def __init__(self, input, output, hidden_state, batch_size, seq_len, ts, isHidStTrain = True):
        self.u = input
        self.y = output
        self.x = hidden_state
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bTrainable = isHidStTrain
        self.ts = ts

    # Batch extraction funtions
    def get_batch(self):
        # Select batch indexes
        num_train_samples = self.u.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - self.seq_len+1, dtype=np.int64), self.batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(self.seq_len) # batch samples indices
        # batch_idx = batch_idx.T  # transpose indexes to obtain batches with structure (m, q, n_x)
                
        # Extract batch data        
        batch_x0_hidden = self.x[batch_start, :]
        batch_x_hidden = self.x[[batch_idx]]
        batch_u = self.u[[batch_idx]]
        batch_y = self.y[[batch_idx]]

        return batch_x0_hidden, batch_u, batch_y, batch_x_hidden
    def setSeq_len(self, seq_len):
        self.seq_len = seq_len
    
    def standardize_(self):
        # Maximum
        self.u_max, _ = torch.max(self.u,dim=0)
        self.y_max, _ = torch.max(self.y, dim = 0)
        self.x_max, _ = torch.max(self.x, dim = 0)

        # Minimum
        self.u_min, _ = torch.min(self.u,dim=0)
        self.y_min, _ = torch.min(self.y, dim = 0)
        self.x_min, _ = torch.min(self.x, dim = 0)

        self.u = (self.u - self.u_min)/(self.u_max - self.u_min).requires_grad_()
        self.y = (self.y - self.y_min)/(self.y_max - self.y_min).requires_grad_()
        if not any(self.x_max==0.):
            self.x = (self.x - self.x_min)/(self.x_max - self.x_min).requires_grad_()



def create_dataloader(dMat, batch_size, seq_len, bStandardize = True, idxMax = 1000000):


    fs = dMat['fs']
    ts = 1/fs[0][0]
    u = dMat['uTot']
    y = dMat['yTot']
    x = dMat['xTot']
    if seq_len == 'all':
        seq_len = u.shape[0]
    u = np.reshape(u, (max(u.shape), min(u.shape)))
    y = np.reshape(y, (max(y.shape), min(y.shape)))
    x = np.reshape(x, (max(x.shape), min(x.shape)))

    
    u_torch = torch.from_numpy(u[:idxMax,:]).to(dtype= torch.float32)
    y_torch = torch.from_numpy(y[:idxMax,:]).to(dtype= torch.float32)
    x_torch = torch.from_numpy(x[:idxMax,:]).to(dtype= torch.float32)

    y_torch_dot = (y_torch[1:,:]-y_torch[0:-1,:])/ts
    y_torch_dot = torch.cat([torch.zeros((1,2)),y_torch_dot])
    u_torch = torch.cat([u_torch, y_torch, y_torch_dot], dim= 1)
    

    nx = x_torch.shape[1]
    z_hidden_fit = torch.zeros((y.size, nx), dtype=torch.float32, requires_grad=True)  # hidden state is an optimization variable
    data_set = SeriesDataset(u_torch, y_torch, z_hidden_fit, seq_len, ts=ts)

    if bStandardize:
        data_set.standardize_()
    

    return data_set


