from alive_progress import alive_bar
import time
import torch
from torch.utils.data import DataLoader
from nssmid.preprocessing import *
from nssmid.postprocessing import *
from nssmid.losses import *
from nssmid.ss_models import *

from scipy.io import savemat
import matplotlib.pyplot as plt
import math
import optuna as opt
import os


def train_feedforward(config):
    seed_everything(config.seed)
    trainLoader, testLoader = getDataLoader(config)
    model = getModel(config)
    criterion = getLoss(config, model)

    if not os.path.exists(config.train_dir):
        os.makedirs(config.train_dir)
    # wanlog = WandbLogger(config)

    print(f"Set global seed to {config.seed:d}")
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        print(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-6*nparams:.1f}M")
    else:
        print(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-3*nparams:.1f}K")


    # Choice of the training algorithm
    if hasattr(criterion, 'lmi'):
        if config.reg_lmi == 'logdet':
            best_model, dict_res = train_logdet(model, criterion, trainLoader, testLoader, config)
        elif config.reg_lmi == 'dd' or config.reg_lmi == 'dd2':
            if config.bReqGradDD == True:
                best_model, dict_res = train_dd_no_bp(model, criterion, trainLoader, testLoader, config)
            else:
                best_model, dict_res = train_dd(model, criterion, trainLoader, testLoader, config)

    else:
        best_model, dict_res = train_ff(model, criterion, trainLoader, testLoader, config)


    # Post-processing ?

    # Saving config
    savemat(config.train_dir + '/config.mat', config.__dict__)
    print(config.train_dir + '/config.mat')
    # Saving best model
    weights, biases = best_model.extract_weights()
    savemat(config.train_dir + '/model.mat', {'weights' : weights, 'biases' : biases})

    savemat(config.train_dir + '/losses.mat', dict_res)
    return best_model, dict_res

def train_logdet(model, criterion, trainLoader, testLoader, config):

    def is_legal(v):
        legal = not torch.isnan(v).any() and not torch.isinf(v)
        return legal
    Epochs = config.epochs
    Lr = config.lr
    update_lmi_cert = config.bCertGrad
    #steps_per_epoch = len(trainLoader)
    if [criterion.lmi.parameters()]:
        params_net = list(model.parameters())
        params_LMI = list(criterion.lmi.parameters())
        params = list(set(params_net+params_LMI))# Get only unique values from all parameters
        optim = torch.optim.Adam(params, lr = 1e-3, weight_decay=0)
    else :
        optim = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    

    print_freq = config.print_freq #100
    patience = config.patience # 500
    tol_change = config.tol_change #1e-5 # Taille de la boule
    max_ls = config.max_ls # 1000
    alpha_ls = config.alpha_ls # 0.5
    bBacktrack = config.backtrack # False
    print(bBacktrack)
    mu_dec = config.mu_dec

    n_u, no_decrease_counter = 0,0
    best_obj = 1e7
    bls_reached = False
    vObj,vReg, vUpd, vTest = [], [], [], []
    mu0 = criterion.mu
    tot_backtrack = 0
    best_model = model.clone()
    for epoch in range(Epochs):
        ## train_step
        n, Loss = 0, 0.0
        model.train()
        for _, batch in enumerate(trainLoader):
            optim.zero_grad()

            x, y = batch[0], batch[1]

            yh = model(x)
            old_theta = model.flatten_params().detach()

            MSE, barr = criterion(yh, y)

            J = MSE + barr*criterion.mu
            J.backward()

            # step model

            optim.step()
            new_theta = model.flatten_params().detach()

            loss = MSE.item()

            n += y.size(0)
            Loss += loss * y.size(0)
            
            
            # Perform a backtracking linesearch to avoid inf or NaNs
            MSE, barrier = criterion(model(x), y)
            ls = 0
            if not is_legal(barrier):
                if bBacktrack:
                    while not is_legal(barrier) and bBacktrack:
                        tot_backtrack = tot_backtrack+1

                        # step back by factor of alpha
                        new_theta = alpha_ls * old_theta + (1 - alpha_ls) * new_theta
                        model.write_flat_params(new_theta)

                        #ls_eigval.append(getEigenvalues(NNODE.ss_model.P.detach()))
                        ls += 1
                        #print(ls)
                        if ls == max_ls:
                            bls_reached = True
                            print("maximum ls reached")
                            break

                        MSE, barrier = criterion(model(x), y) 
                else:
                    break
        if not is_legal(barrier):
            break
        train_loss = Loss/n 

        if bls_reached:
            break
        obj = train_loss
        #r = torch.max(dQ).detach().numpy()
        reg = (barr*criterion.mu).detach().numpy()
        vObj.append(obj)
        vReg.append(reg)
        if  (best_obj - obj)/best_obj > tol_change:
            best_obj = obj
            no_decrease_counter = 0
            best_model = model.clone()
            
        else:
            no_decrease_counter = no_decrease_counter +1
        if no_decrease_counter> patience:
            with torch.no_grad():
                #print("Updating U")
                criterion.mu = criterion.mu*mu_dec
                n_u = n_u +1
                vUpd.append(epoch)
            no_decrease_counter = 0
        if criterion.mu < 1e-8:
            break
        

        n, Loss = 0, 0.0
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(testLoader):
                x, y = batch[0], batch[1]
                yh = model(x)
                MSE, barr = criterion(yh, y)

                J = MSE + barr*criterion.mu
                Loss += MSE.item() * y.size(0)
                n += y.size(0)

        test_loss = Loss/n
        vTest.append(test_loss)
        if epoch%print_freq ==0:
            lr = optim.param_groups[0]['lr']
            print(f"Iter {epoch} : Loss  = {float(train_loss):.5f} -- Test loss {float(test_loss):5f} | Barr term : {reg:.5f} | No dec count {no_decrease_counter} | lr: {lr:.3f}")
            
        if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")  

    print(f"Number of barrier term updates : {n_u}")
    print(f"Number of backtracking steps : {tot_backtrack}")

    fig, ax = plt.subplots(2,1, sharex=True)

    ax[0].plot(range(len(np.array(vObj))),np.log10(np.array(vObj)), label = 'Loss')
    ax[0].plot((range(len(np.array(vTest)))), np.log10(np.array(vTest)), label = 'Test loss') 
    # Tracer des lignes verticales pour chaque indice
    for epch in vUpd:
        ax[0].axvline(x=epch, color='r', linestyle='--')

    # Ajouter une légende
    ax[0].legend()

    ax[1].plot(range(len(np.array(vReg))), vReg, label = 'Barrier term x mu')
    for epch in vUpd:
        ax[1].axvline(x=epch, color='r', linestyle='--')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("MSE (log10)")
    plt.title(f"{config.model} log det backtrack {config.backtrack}")
    plt.show()

    strSaveName = str(config.model) + '_logdet' + str(mu0) + f'lr_{Lr}' + f'_{Epochs}epch'
    if bBacktrack:
        strSaveName = strSaveName + '_bt'
    if update_lmi_cert:
        strSaveName = strSaveName + '_updtCert'
        

    fig.savefig(f"{config.train_dir}/{strSaveName}.png")

    dict_res = {'train_loss' : vObj,
                'test_loss' : vTest,
                'reg_term' : vReg}


    return best_model, dict_res  


def train_dd(model, criterion, trainLoader, testLoader, config):
    Epochs = config.epochs
    Lr = config.lr
    update_lmi_cert = config.bCertGrad
    #steps_per_epoch = len(trainLoader)
    if [criterion.lmi.parameters()]:
        params_net = list(model.parameters())
        params_LMI = list(criterion.lmi.parameters())
        params = list(set(params_net+params_LMI))# Get only unique values from all parameters
        optim = torch.optim.Adam(params, lr = Lr, weight_decay=0)
    else :
        optim = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    

    print_freq = config.print_freq #100
    patience = config.patience # 500
    tol_change = config.tol_change #0.01 = 1% relative tolerance

    n_u, no_decrease_counter = 0,0
    best_obj = 1e10
    vObj,vReg, vUpd, vTest = [], [], [], []
    best_model = model.clone()
    for epoch in range(Epochs):
        ## train_step
        n, Loss, r = 0, 0.0, 0
        model.train()
        for batch_idx, batch in enumerate(trainLoader):
            optim.zero_grad()

            x, y = batch[0], batch[1]

            yh = model(x)
            MSE, dQ, lmis = criterion(yh, y)
            
            reg = 0
            for delta in dQ:
                r = r + torch.max(delta) # total distance to DD+
                reg = reg + torch.max(delta)*criterion.mu
                
            J = MSE + reg
            #print(reg)
            J.backward()
            optim.step()

            loss = MSE.item()

            n += y.size(0)
            Loss += loss * y.size(0)
        train_loss = Loss/n 
        r = r/(batch_idx+1) # Average distance to DD+ on the whole batch.

        obj = train_loss
        deltas = [torch.max(delt).detach().numpy() for delt in dQ]
        #r = 10
        vObj.append(obj)
        vReg.append(deltas)
        if r<=0: # All Q are DD+
            # Start counting
            if  (best_obj - obj)/best_obj > tol_change:
                best_obj = obj
                no_decrease_counter = 0
                best_model = model.clone()
                
            else:
                no_decrease_counter = no_decrease_counter +1
            if no_decrease_counter> patience:
                with torch.no_grad():
                    #print("Updating U")
                    for i,lmi in enumerate(lmis):
                        criterion.ddLayers[i].updateU_(lmi)
                    n_u = n_u +1
                    vUpd.append(epoch)
                no_decrease_counter = 0


        

        ## dummy call to flush the new model parameter in the last batch
        #model(torch.rand((1,x.shape[1])).to(x.device)) 

        n, Loss= 0, 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testLoader):
                x, y = batch[0], batch[1]
                yh = model(x)
                MSE, dQ, M = criterion(yh, y)

                reg = 0
                
                for delta in dQ:
                    reg = reg + torch.max(delta)*criterion.mu
                    
                J = MSE + reg
                Loss += MSE.item() * y.size(0)
                n += y.size(0)

        test_loss = Loss/n
        vTest.append(test_loss)
        if epoch%print_freq ==0:
            lr = optim.param_groups[0]['lr']
            print(f"Iter {epoch} : Loss  = {float(train_loss):.5f} -- Test loss {float(test_loss):5f} | Dist to DD+ : {r:.5f} | No dec count {no_decrease_counter} | lr: {lr:.3f}")
            
        if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")  

    print(f"Number of bases updates : {n_u}")
    print(criterion.ddLayers[0].Ui)


    fig, ax = plt.subplots(2,1, sharex=True)

    ax[0].plot(range(len(np.array(vObj))),np.log10(np.array(vObj)), label = 'Loss')
    ax[0].plot((range(len(np.array(vTest)))), np.log10(np.array(vTest)), label = 'Test loss') 
    # Tracer des lignes verticales pour chaque indice
    for epoch in vUpd:
        ax[0].axvline(x=epoch, color='r', linestyle='--')

    # Ajouter une légende
    ax[0].legend()
    vRegs = np.array(vReg)
    for reg in vRegs.T:
        ax[1].plot(range(len(np.array(reg))), reg, label = 'Distance to DD+')
    for epoch in vUpd:
        ax[1].axvline(x=epoch, color='r', linestyle='--')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("MSE (log10)")
    plt.title(f"Dummy model mu = {criterion.mu}")
    plt.show()

    strSaveName = str(config.model) + '_' +config.reg_lmi + str(criterion.mu) + f'lr_{Lr}' + f'_{Epochs}epch'
    if update_lmi_cert:
        strSaveName = strSaveName + '_updtCert'
        

    fig.savefig(f"{config.train_dir}/{strSaveName}.png")

    dict_res = {'train_loss' : vObj,
                'test_loss' : vTest,
                'reg_term' : vReg}

    return best_model, dict_res  


def train_dd_no_bp(model, criterion, trainLoader, testLoader, config):
    Epochs = config.epochs
    Lr = config.lr
    update_lmi_cert = config.bCertGrad
    #steps_per_epoch = len(trainLoader)
    if [criterion.lmi.parameters()]:
        params_net = list(model.parameters())
        params_LMI = list(criterion.parameters())
        params = list(set(params_net+params_LMI))# Get only unique values from all parameters
        optim = torch.optim.Adam(params, lr = Lr, weight_decay=0)
    else :
        optim = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    

    print_freq = config.print_freq #100
    patience = config.patience # 500
    tol_change = config.tol_change #0.01 = 1% relative tolerance

    n_u, no_decrease_counter = 0,0
    best_obj = 1e10
    vObj,vReg, vUpd, vTest = [], [], [], []
    best_model = model.clone()
    for epoch in range(Epochs):
        ## train_step
        n, Loss, r = 0, 0.0, 0
        model.train()
        for batch_idx, batch in enumerate(trainLoader):
            optim.zero_grad()

            x, y = batch[0], batch[1]

            yh = model(x)
            MSE, dQ, lmis = criterion(yh, y)

            reg = 0
            for delta in dQ:
                r = r + torch.max(delta) # total distance to DD+
                reg = reg + torch.max(delta)*criterion.mu
                
            J = MSE + reg
            #print(reg)
            J.backward()
            optim.step()

            loss = MSE.item()

            n += y.size(0)
            Loss += loss * y.size(0)
        train_loss = Loss/n 
        r = r/(batch_idx+1) # Average distance to DD+ on the whole batch.

        obj = train_loss
        deltas = [torch.max(delt).detach().numpy() for delt in dQ]
        #r = 10
        vObj.append(obj)
        vReg.append(deltas)
        if r==0: # All Q are DD+
            # Start counting
            if  (best_obj - obj)/best_obj > tol_change:
                best_obj = obj
                no_decrease_counter = 0
                best_model = model.clone()
                
            else:
                no_decrease_counter = no_decrease_counter +1
            if no_decrease_counter> patience:
                break
        else: # We are not yet in DD+
            '''
            if obj < best_obj -tol_change:
                best_obj = obj
                no_decrease_counter = 0
                best_model = model.clone()
            else:
                no_decrease_counter = no_decrease_counter +1
            if no_decrease_counter> patience:
                criterion.mu = criterion.mu / config.mu_dec
                no_decrease_counter = 0
                vUpd.append(epoch)
        if criterion.mu >= 1e8:
            break '''
        

        ## dummy call to flush the new model parameter in the last batch
        #model(torch.rand((1,x.shape[1])).to(x.device)) 

        n, Loss= 0, 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testLoader):
                x, y = batch[0], batch[1]
                yh = model(x)
                MSE, dQ, M = criterion(yh, y)

                reg = 0
                
                for delta in dQ:
                    reg = reg + torch.max(delta)*criterion.mu
                    
                J = MSE + reg
                Loss += MSE.item() * y.size(0)
                n += y.size(0)

        test_loss = Loss/n
        vTest.append(test_loss)
        if epoch%print_freq ==0:
            lr = optim.param_groups[0]['lr']
            print(f"Iter {epoch} : Loss  = {float(train_loss):.5f} -- Test loss {float(test_loss):5f} | Dist to DD+ : {r:.5f} | No dec count {no_decrease_counter} | lr: {lr:.3f}")
            
        if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")  

    print(f"Number of bases updates : {n_u}")

    print(criterion.ddLayers[0].Ui)
    fig, ax = plt.subplots(2,1, sharex=True)

    ax[0].plot(range(len(np.array(vObj))),np.log10(np.array(vObj)), label = 'Loss')
    ax[0].plot((range(len(np.array(vTest)))), np.log10(np.array(vTest)), label = 'Test loss') 
    # Tracer des lignes verticales pour chaque indice
    for epoch in vUpd:
        ax[0].axvline(x=epoch, color='r', linestyle='--')

    # Ajouter une légende
    ax[0].legend()
    vRegs = np.array(vReg)
    for reg in vRegs.T:
        ax[1].plot(range(len(np.array(reg))), reg, label = 'Distance to DD+')
    for epoch in vUpd:
        ax[1].axvline(x=epoch, color='r', linestyle='--')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("MSE (log10)")
    plt.title(f"Dummy model mu = {criterion.mu}")
    plt.show()

    strSaveName = str(config.model) + '_dd_bis' + str(criterion.mu) + f'lr_{Lr}' + f'_{Epochs}epch'
    if update_lmi_cert:
        strSaveName = strSaveName + '_updtCert'
        

    fig.savefig(f"{config.train_dir}/{strSaveName}.png")

    dict_res = {'train_loss' : vObj,
                'test_loss' : vTest,
                'reg_term' : vReg}

    return best_model, dict_res  



def train_ff(model, criterion, trainLoader, testLoader, config):

    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        print(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-6*nparams:.1f}M")
    else:
        print(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-3*nparams:.1f}K")
    
    Epochs = config.epochs
    Lr = config.lr
    steps_per_epoch = len(trainLoader)

    optim = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    lr_schedule = lambda t: np.interp([t], [0, Epochs*2//5, Epochs*4//5, Epochs], [0, Lr, Lr/20.0, 0])[0]
    vObj, vTest = [], []
    print_freq = 100
    no_decrease_counter = 0
    patience = config.patience #500
    tol_change = config.tol_change #1e-4 # Taille de la boule
    best_loss = math.inf
    for epoch in range(Epochs):
        ## train_step
        n, Loss = 0, 0.0
        model.train()
        for batch_idx, batch in enumerate(trainLoader):
            optim.zero_grad()

            x, y = batch[0], batch[1]
            lr = lr_schedule(epoch + (batch_idx+1)/steps_per_epoch)
            optim.param_groups[0].update(lr=lr)

            yh = model(x)
            J = criterion(yh, y)
            J.backward()
            optim.step()

            loss = J.item()

            n += y.size(0)
            Loss += loss * y.size(0)
        train_loss = Loss/n 
        
        vObj.append(train_loss)
        n, Loss = 0, 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testLoader):
                x, y = batch[0], batch[1]
                yh = model(x)
                J = criterion(yh,y)
                Loss += J.item() * y.size(0)
                n += y.size(0)

        test_loss = Loss/n
        vTest.append(test_loss)

        if  test_loss <  best_loss- tol_change:
            best_loss = test_loss
            no_decrease_counter = 0
            best_model = model.clone()
        else:
            no_decrease_counter = no_decrease_counter +1
        if epoch%print_freq ==0:
            lr = optim.param_groups[0]['lr']
            print(f"Iter {epoch} : Loss  = {float(train_loss):.5f} -- Test loss {float(test_loss):5f} | No dec count {no_decrease_counter} | lr: {lr:.3f}")
            
        if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")  

        if no_decrease_counter> patience: # Early stopping
            break


    fig = plt.figure()

    plt.plot(range(len(vObj)),np.log10(np.array(vObj)), label = 'Loss')
    plt.plot(range(len(vTest)), np.log10(np.array(vTest)), label = 'Test loss') 

    # Ajouter une légende
    plt.legend()
    plt.title(f"Test")
    plt.show()

    strSaveName = str(config.model) + f'lr_{Lr}' + f'_{Epochs}epch'

    fig.savefig(f"{config.train_dir}/{strSaveName}.png")

    dict_res = {'train_loss' : vObj,
                'test_loss' : vTest}
    return best_model, dict_res  


def train_recurrent(config):
    seed_everything(config.seed)
    trainLoader, testLoader = getDataLoader(config)
    model = getModel(config)
    criterion = getLoss(config, model)

    if not os.path.exists(config.train_dir):
        os.makedirs(config.train_dir)
    # wanlog = WandbLogger(config)

    print(f"Set global seed to {config.seed:d}")
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        print(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-6*nparams:.1f}M")
    else:
        print(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-3*nparams:.1f}K")
    best_model, dict_res = train_rnn(model, criterion, trainLoader, testLoader, config)

    # Post-processing ?

    # Saving config
    savemat(config.train_dir + '/config.mat', config.__dict__)
    print(config.train_dir + '/config.mat')
    # Saving best model
    weights, biases = best_model.extract_weights()
    savemat(config.train_dir + '/model.mat', {'weights' : weights, 'biases' : biases})

    savemat(config.train_dir + '/losses.mat', dict_res)
    return best_model, dict_res


def train_rnn(model, criterion, trainSet, testSet, config):

    u_train = trainSet.u
    y_train = trainSet.y

    u_test = testSet.u
    y_test = testSet.y
    nx = config.nx

    x_est = np.zeros((y_train.shape[0], nx), dtype=np.float32)
    # Hidden state variable
    x_hidden_fit = torch.tensor(x_est, dtype=torch.float32, requires_grad=True)  # hidden state is an optimization variable

    # Batch extraction funtions
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = u_train.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch samples indices

        # Extract batch data
        
        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_x_hidden = x_hidden_fit[[batch_idx]]
        batch_u = u_train[[batch_idx]]
        batch_y = y_train[[batch_idx]]

        return batch_x0_hidden, batch_u, batch_y, batch_x_hidden


    lr = config.lr

    # Setup optimizer
    params_net = list(model.parameters())
    params_hidden = [x_hidden_fit]
    optimizer = torch.optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': lr},
    ], lr=lr)



    x0_val = torch.zeros((nx), dtype=torch.float32)
    u_torch_val = u_test.to(dtype= torch.float32)
    y_true_torch_val = y_test.to(dtype= torch.float32)
    
    _, y_sim_init = model.simulate(u_torch_val, x0_val)
    val_mse =  torch.mean((y_true_torch_val-y_sim_init)**2)
    print("Initial val_MSE = {:.7f} \n".format(float(val_mse)))
    vLoss, vVal_mse, vInfo = [], [], []


    start_time = time.time()
    # Training loop
    best_loss = val_mse
    best_model = model.clone()
    no_decrease_counter = 0
    Epochs = config.epochs

    batch_size = config.batch_size
    seq_len = config.seq_len
    test_freq = config.test_freq
    tol_change = config.tol_change # 0.2
    patience = config.patience
    
    with alive_bar(Epochs) as bar:
        epoch_loss = 0.0
        for itr in range(0, Epochs):

            optimizer.zero_grad()

            # Simulate
            #x0_torch = torch.zeros((nx))
            batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
            x_sim_torch_fit, y_sim_torch_fit = model(batch_u, batch_x0_hidden)


            # Compute fit loss
            loss = criterion(batch_y, y_sim_torch_fit, batch_x_hidden, x_sim_torch_fit)

            epoch_loss += float(loss.item())

            if itr % test_freq == 0:
                # Statistics
                epoch_loss = epoch_loss/test_freq
                vLoss.append(epoch_loss)

                with torch.no_grad():
                    # Simulation perf on test data
                    _, y_sim_val = model.simulate(u_torch_val, x0_val)
                    val_mse =  torch.mean((y_true_torch_val-y_sim_val)**2)
                    vVal_mse.append(val_mse)

                if (best_loss - val_mse)/best_loss > tol_change:
                        no_decrease_counter = 0
                        best_loss = val_mse
                        best_model = model.clone()

                        # Check if the lmi referenced attributes are indeed the one from the current best model
                        '''
                        A_lmi= criterion.lmi.A
                        A_model = best_model.ss_model.linmod.A.weight
                        B_lmi= criterion.lmi.B
                        B_model = best_model.ss_model.linmod.B.weight[:,1:]
                        C_lmi = criterion.lmi.C
                        assert torch.all(A_lmi == A_model)
                        assert torch.all(B_lmi == B_model)
                        assert torch.all(C_lmi ==  best_model.ss_model.linmod.C.weight)
                        print(f'Gamma = {float(criterion.lmi.gamma):.4e}')
                        lmi = criterion.lmi()
                        P_opt, gamma_opt = criterion.lmi.solve_lmi(best_model.ss_model.linmod.A.weight.detach().numpy(), 
                            best_model.ss_model.linmod.B.weight[:,1:].detach().numpy(),
                            best_model.ss_model.linmod.C.weight.detach().numpy(), abs_tol = 1e-4, solver = "MOSEK")
                        '''
                        
                else:
                    no_decrease_counter += 1
                print(" Epoch loss = {:.7f} || Val_MSE = {:.7f} || Best loss = {:.7f} \n".format(float(epoch_loss),
                        float(val_mse), float(best_loss)))    
                epoch_loss = 0.0
                if no_decrease_counter> patience/5 and hasattr(criterion, 'mu'):
                    criterion.update_mu_(0.1)
                    print(f"Updating barrier term weight : mu = {criterion.mu}")
                    no_decrease_counter = 0
                if hasattr(criterion, 'mu') and criterion.mu<1e-8:
                    break
                if no_decrease_counter > patience: # early stopping
                    break
                
            if (math.isnan(loss)): 
                break
            # Optimize
            loss.backward()
            optimizer.step()

            
            bar()

    train_time = time.time() - start_time

    print("Total dentification runtime : {} \n Best loss : {} \n".format(train_time, best_loss))

    # Final simulation perf on test data
    _, y_sim_val = best_model.simulate(u_torch_val, x0_val)
    val_mse =  torch.mean((y_true_torch_val-y_sim_val)**2)

    # Final simulation on train data
    _, y_sim_train = best_model.simulate(u_train, torch.zeros(nx))
    train_mse = torch.mean((y_train - y_sim_train)**2)
    print(" Final MSE = {:.7f} || Val_MSE = {:.7f} \n".format(float(train_mse.detach()),float(val_mse)))

    # Saving Mat-file training results
    weights, biases = best_model.extract_weights()
    dictRes = {'weights' : weights, 'biases' : biases, 'best_loss' : best_loss,
                'info': vInfo, 'Loss_hist' : vLoss, 'Val_loss_hist' : vVal_mse,
                 'y_sim' : y_sim_val.squeeze(0)}

    return best_model, dictRes
    
def train_network_opt(trial, model, u_train : torch.Tensor, y_train : torch.Tensor, nx : int, 
            u_test : torch.Tensor, y_test : torch.Tensor, batch_size, seq_len,
            lr, num_iter, test_freq = 100, patience = 70, tol_change = 0.05):

    # Build initial state estimate
    x_est = np.zeros((y_train.shape[0], nx), dtype=np.float32)
    x_est_test = np.zeros((y_test.shape[0], nx), dtype=np.float32)
    # Hidden state variable
    x_hidden_fit = torch.tensor(x_est, dtype=torch.float32, requires_grad=True)  # hidden state is an optimization variable

    # Batch extraction funtions
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = u_train.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch samples indices

        # Extract batch data
        
        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_x_hidden = x_hidden_fit[[batch_idx]]
        batch_u = u_train[[batch_idx]]
        batch_y = y_train[[batch_idx]]

        return batch_x0_hidden, batch_u, batch_y, batch_x_hidden

    nu = u_train.shape[1]
    ny = y_train.shape[1]

    # Setup optimizer
    params_net = list(model.parameters())
    params_hidden = [x_hidden_fit]
    optimizer = torch.optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': lr},
    ], lr=lr)



    x0_val = torch.zeros((nx), dtype=torch.float32)
    u_torch_val = u_test.to(dtype= torch.float32)
    y_true_torch_val = y_test.to(dtype= torch.float32)
    
    x_sim_val, y_sim_init = model.simulate(u_torch_val, x0_val)
    val_mse =  torch.mean((y_true_torch_val-y_sim_init)**2)
    print("Initial val_MSE = {:.7f} \n".format(float(val_mse)))
    vLoss = []
    vVal_mse = []
    vInfo = []

    start_time = time.time()
    # Training loop
    best_loss = val_mse
    best_model = model.clone()
    no_decrease_counter = 0
    with alive_bar(num_iter) as bar:
        for itr in range(0, num_iter):

            optimizer.zero_grad()

            # Simulate
            #x0_torch = torch.zeros((nx))
            batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
            x_sim_torch_fit, y_sim_torch_fit = model(batch_u, batch_x0_hidden)
            x_sim_torch_fit = x_sim_torch_fit.squeeze()
            y_sim_torch_fit = y_sim_torch_fit.squeeze()

            # Compute fit loss
            err_fit = y_sim_torch_fit - batch_y.squeeze()
            err_fit_scaled = err_fit 
            loss = torch.mean(err_fit_scaled**2)

            
            # Statistics
            vLoss.append(loss.item())
            if itr % test_freq == 0:
                with torch.no_grad():
                    # Simulation perf on test data
                    x_sim_val, y_sim_val = model.simulate(u_torch_val, x0_val)
                    val_mse =  torch.mean((y_true_torch_val-y_sim_val)**2)
                    vVal_mse.append(val_mse)
                    
                if (best_loss - val_mse)/best_loss > tol_change:
                        no_decrease_counter = 0
                        best_loss = val_mse
                        best_model = model.clone()
                else:
                    no_decrease_counter += 1
                print(" MSE = {:.7f} || Val_MSE = {:.7f} || Best loss = {:.7f} \n".format(float(loss.detach()),
                        float(val_mse), float(best_loss)))
                if no_decrease_counter > patience: # early stopping
                    break
                trial.report(loss, itr)
                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    raise opt.exceptions.TrialPruned()
            if (math.isnan(loss) or loss<1e-7): 
                break
            # Optimize
            loss.backward()
            optimizer.step()

            
            bar()

    train_time = time.time() - start_time

    print("Total dentification runtime : {} \n Best loss : {} \n".format(train_time, best_loss))

    # Final simulation perf on training data
    x_sim_val, y_sim_val = best_model.simulate(u_torch_val, x0_val)
    val_mse =  torch.mean((y_true_torch_val-y_sim_val)**2)
    print(" MSE = {:.7f} || Val_MSE = {:.7f} \n".format(float(loss.detach()),float(val_mse)))

    # Saving Mat-file training results
    weights, biases = best_model.extract_weights()
    dictRes = {'weights' : weights, 'biases' : biases, 'best_loss' : best_loss,
                'info': vInfo, 'Loss_hist' : vLoss, 'Val_loss_hist' : vVal_mse,
                 'y_sim' : y_sim_val.squeeze(0)}

    return best_model, dictRes

'''
How to save a model

# Saving best model 
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model.state_dict(),
            'loss': vLoss,
            'val_loss': vVal_Loss,
            'trainingInfo' : info}, strSavePath)

'''