from torch.optim import *
from alive_progress import alive_bar
import time
import torch
from torch.utils.data import DataLoader
from nssmid.postprocessing import *
import matplotlib.pyplot as plt
import math
import optuna as opt
def run_id(model, train_loader : DataLoader, num_epochs, criterion,
          lr= 1e-3, optimizer :str = 'adam', test_loader : DataLoader = None, 
          test_criterion = None, test_freq = 100,patience = 100, 
          tol_change = 1e-8, val_loader= None):

    # ADAM optimizer
    params_net = list(model.parameters())
    param_groups = [{'params': params_net,    'lr': lr, "betas": (0.95, 0.99)}]

    if train_loader.dataset.x.requires_grad:
        param_groups.append({'params': train_loader.dataset.x, 'lr' : lr})

    if optimizer == 'adam':
        optimizer = Adam(param_groups)
    elif optimizer == 'SGD':
        optimizer = SGD(param_groups)
    else:
        raise NotImplementedError(" Please select another optimizer")

    vLoss = []
    vVal_Loss = []
    vInfo = []
    no_decrease_counter = 0


    # 1st loss (linear model)
    with torch.no_grad():
        val_loss = 0.0
        for i, batch_test in enumerate(test_loader):
            u,y_true,x_true,x0 = batch_test
            x_sim, y_sim = model(u, x0)
            val_loss += test_criterion(y_true,y_sim, x_true, x_sim)

            # Simulation
            fig, ax = plt.subplots(2)
            ax[0].plot(x_sim.squeeze())
            #ax[0].plot(x_true.squeeze())
            plt.ylim([-4,4])
          
            ax[1].plot(y_true.squeeze())
            ax[1].plot(y_sim.squeeze())
            plt.ylim([-4,4])
            plt.title('Response of initial linear estimate')

        best_loss = val_loss/(i+1)
        print(" Initial Val_MSE (BLA) = {:.7f} \n".format(float(best_loss)))
        vVal_Loss.append(best_loss)
        best_model = model.clone()

    with alive_bar(num_epochs) as bar:
        #  Main Loop of Optimizer
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i_batch, batch in enumerate(train_loader):
                #  --------------- Training Step ---------------
                def closure():
                        optimizer.zero_grad()
                        u, y_true, x_true, x0 = batch
                        x_sim, y_sim = model(u, x0)
                        L = criterion(y_true, y_sim, x_true, x_sim)
                        L.backward()
                        return L

                # step model
                L = optimizer.step(closure)
                epoch_loss += float(L.detach())
                #vInfo.append(info)
                # Printing
            epoch_loss = epoch_loss/(i_batch+1)
            vLoss.append(float(epoch_loss))
            if epoch % test_freq == 0:
                with torch.no_grad():
                    test_loss = 0.0
                    for i, batch_test in enumerate(test_loader):
                        u_test,y_true_test,x_true_test,x0_test = batch_test
                        x_sim_test, y_sim_test = model(u_test, x0_test)
                        test_loss += test_criterion(y_true_test,y_sim_test, x_true_test, x_sim_test)
                    test_mse = test_loss/(i+1)
                    print(" MSE = {:.7f} || Test_MSE = {:.7f} \n".format(epoch_loss,float(test_mse)))
                    vVal_Loss.append(test_mse)

                    if test_mse < best_loss - tol_change:
                        no_decrease_counter = 0
                        best_loss = test_mse
                        best_model = model.clone()
                    else:
                        no_decrease_counter += 1
                    if no_decrease_counter > patience: # early stopping
                        break
            bar()
        end_time = time.time()
        print("Total dentification runtime : {} \n Best loss : {} \n".format(end_time - start_time, best_loss))

        # Final simulation perf on training data
        with torch.no_grad():
            val_loss = 0.0
            for i, batch_val in enumerate(val_loader):
                u_val,y_true_val,x_true_val,x0_val = batch_val
                x_sim_val, y_sim_val = model(u_val, x0_val)
                val_loss += criterion(y_true_val,y_sim_val, x_true_val, x_sim_val)
            val_mse = val_loss/(i+1)
            print(" MSE = {:.7f} || Val_MSE = {:.7f} \n".format(float(L.detach()),float(val_mse)))
        # Saving Mat-file training results
        weights, biases = best_model.extract_weights()
        dictRes = {'weights' : weights, 'biases' : biases,
                   'info': vInfo, 'Loss_hist' : vLoss, 'best_loss' : best_loss,
                   'Val_loss_hist' : vVal_Loss, 'y_sim' : y_sim_val.squeeze(0)}

        
    return best_model, dictRes
    

def run_id2(model, train_set , num_epochs, criterion,
          lr= 1e-3, optimizer :str = 'adam', test_set  = None, 
          test_criterion = None, test_freq = 100,patience = 100, 
          tol_change = 1e-8, val_set= None):

    # ADAM optimizer
    params_net = list(model.parameters())
    param_groups = [{'params': params_net,    'lr': lr, "betas": (0.95, 0.99)}]

    if train_set.x.requires_grad:
        param_groups.append({'params': train_set.x, 'lr' : lr})

    if optimizer == 'adam':
        optimizer = Adam(param_groups)
    elif optimizer == 'SGD':
        optimizer = SGD(param_groups)
    else:
        raise NotImplementedError(" Please select another optimizer")

    vLoss = []
    vVal_Loss = []
    vInfo = []
    no_decrease_counter = 0


    # 1st loss (linear model)
    with torch.no_grad():
        val_loss = 0.0
        
        x0, u, y_true, x_true = test_set.get_batch()
        x_sim, y_sim = model(u, x0)
        val_loss += test_criterion(y_true,y_sim, x_true, x_sim)


        best_loss = val_loss
        print(" Initial Val_MSE (BLA) = {:.7f} \n".format(float(best_loss)))
        vVal_Loss.append(best_loss)
        best_model = model.clone()
    
    with alive_bar(num_epochs) as bar:
        #  Main Loop of Optimizer
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            #  --------------- Training Step ---------------
            def closure():
                    optimizer.zero_grad()
                    x0, u, y_true, x_true = train_set.get_batch()
                    x_sim, y_sim = model(u, x0)
                    L = criterion(y_true, y_sim, x_true, x_sim)
                    L.backward(retain_graph=True)
                    return L

            # step model
            L = optimizer.step(closure)
            epoch_loss += float(L.detach())
            #vInfo.append(info)
                # Printing
            epoch_loss = epoch_loss
            vLoss.append(float(epoch_loss))
            if epoch % test_freq == 0:
                with torch.no_grad():
                    test_loss = 0.0
                    
                    x0_test,u_test,y_true_test,x_true_test = test_set.get_batch()
                    x_sim_test, y_sim_test = model(u_test, x0_test)
                    test_loss += test_criterion(y_true_test,y_sim_test, x_true_test, x_sim_test)
                    test_mse = test_loss
                    print(" MSE = {:.7f} || Test_MSE = {:.7f} \n".format(epoch_loss,float(test_mse)))
                    vVal_Loss.append(test_mse)

                    if test_mse < best_loss - tol_change:
                        no_decrease_counter = 0
                        best_loss = test_mse
                        best_model = model.clone()
                    else:
                        no_decrease_counter += 1
                    if no_decrease_counter > patience: # early stopping
                        break
            bar()
        end_time = time.time()
        print("Total dentification runtime : {} \n Best loss : {} \n".format(end_time - start_time, best_loss))

        # Final simulation perf on training data
        with torch.no_grad():
            val_loss = 0.0
            
            x0_val,u_val,y_true_val,x_true_val = val_set.get_batch()
            x_sim_val, y_sim_val = model(u_val, x0_val)
            val_loss += criterion(y_true_val,y_sim_val, x_true_val, x_sim_val)
            val_mse = val_loss
            print(" MSE = {:.7f} || Val_MSE = {:.7f} \n".format(float(L.detach()),float(val_mse)))
        # Saving Mat-file training results
        weights, biases = best_model.extract_weights()
        dictRes = {'weights' : weights, 'biases' : biases,
                   'info': vInfo, 'Loss_hist' : vLoss, 
                   'Val_loss_hist' : vVal_Loss, 'y_sim' : y_sim_val.squeeze(0)}

        
    return best_model, dictRes
    



def train_network(model, u_train : torch.Tensor, y_train : torch.Tensor, nx : int, 
            u_test : torch.Tensor, y_test : torch.Tensor, batch_size, seq_len,
            lr, num_iter, criterion, test_freq = 100, patience = 50, tol_change = 0.02):

    # Build initial state estimate
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
        epoch_loss = 0.0
        for itr in range(0, num_iter):

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
    x_sim_val, y_sim_val = best_model.simulate(u_torch_val, x0_val)
    val_mse =  torch.mean((y_true_torch_val-y_sim_val)**2)

    # Final simulation on train data
    x_sim_train, y_sim_train = best_model.simulate(u_train, torch.zeros(nx))
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