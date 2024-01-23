import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader




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


