from alive_progress import alive_bar
import time
import torch
import math
# import optuna as opt
from ..integrators.integrators import Simulator
from abc import ABC, abstractmethod
from ..utils.data import ExperimentsDataset
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from typing import Union, Optional
from ..losses.losses import BaseLoss
from ..utils.misc import is_legal, flatten_params, write_flat_params


class Trainer(ABC):
    """
        This class is an abstract class Trainer for which the fit method and eval method must be implemented
        rnn_trainer = Trainer(sim_model, training_loss, optimizer, **kwargs)
        net = rnn_trainer.fit_(train_data, **kwargs)
        metrics = rnn_trainer.eval_(test_data, **kwargs)
    """

    def __init__(self, sim_model: Simulator, loss: BaseLoss, val_loss: Union[BaseLoss, None], **kwargs) -> None:
        super(Trainer, self).__init__()
        self.sim_model = sim_model
        self.criterion = loss
        self.val_criterion = val_loss

    @abstractmethod
    def fit_(self):
        pass

    @abstractmethod
    def eval_(self):
        pass


class SSTrainer(Trainer):
    """
        This class is a used for training state-space models.
    """

    def __init__(self, sim_model: Simulator, loss: BaseLoss, val_loss: BaseLoss) -> None:
        super(SSTrainer, self).__init__(sim_model=sim_model, loss=loss, val_loss=val_loss)

    def fit_(self, train_set: ExperimentsDataset, test_set: ExperimentsDataset,
             seq_len: int = 30, batch_size: int = 256, opt: str = 'adam', lr: float = 1e-2, min_lr: float = 1e-5, test_freq: int = 10,
             patience: int = 10, tol_change: float = 0.01, epochs: int = 1000, save_path: Optional[str] = None,
             keep_best: bool = True, max_val_samples: int = 3000, scheduled: bool = False, patience_soft: int = 5,
             backtrack: bool = False):

        train_set.set_seq_len(seq_len=seq_len)

        self.sim_model.set_save_path(save_path)

        # Consider making x a training parameter or not
        trainable_xs = [exp.x for exp in train_set.experiments if exp.x_trainable]
        params_model = self.sim_model.parameters()

        list_params = [{'params': params_model, 'lr': lr},
                       {'params': trainable_xs, 'lr': lr}]
        # Choice of the optimizer
        if opt == 'adam':
            optimizer = Adam(list_params, lr=lr)
        elif opt == 'adamw':
            optimizer = AdamW(list_params, lr=lr)
        elif opt == 'sgd':
            optimizer = SGD(list_params, lr=lr)
        else:
            raise NotImplementedError("Please specify an optimizer among 'adam', 'sgd' , 'adamw' \n")

        # Learning rate and soft constraints scheduleur

        if scheduled:
            step_sched = 0.1
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', step_sched,
                                                                   patience_soft, verbose=True, min_lr=min_lr)
        else:
            scheduled = False
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Logging
        vLoss, vVal_loss = [], []
        # Compute first loss
        init_val_loss, _, _ = self.eval_(val_set=test_set)
        vVal_loss.append(float(init_val_loss))
        print("Initial val_MSE = {:.7f} \n".format(float(init_val_loss)))
        best_loss, no_decrease_counter = init_val_loss, 0
        best_model = self.sim_model.clone()
        start_time = time.perf_counter()

        with alive_bar(total=epochs) as bar:
            for itr in range(0, epochs):

                epoch_loss = 0.0
                for _, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    batch_u, batch_y, batch_x, batch_x0 = batch
                    x_sim_fit, y_sim_fit = self.sim_model(batch_u, batch_x0)

                    # Compute fit loss
                    loss = self.criterion(batch_y, y_sim_fit, x_pred=x_sim_fit, x_true=batch_x)

                    # Optimize
                    loss.backward()
                    optimizer.step()

                    if backtrack and not is_legal(loss):
                        loss = self.backtrack(batch_y, y_sim_fit, batch_x, x_sim_fit)
                    epoch_loss += float(loss.item())
                epoch_loss = epoch_loss / (len(train_loader))
                # Statistics
                vLoss.append(epoch_loss)
                if itr % test_freq == 0:
                    with torch.no_grad():
                        # Simulation perf on test data
                        val_crit, bcheck, _ = self.eval_(val_set=test_set, max_idx=max_val_samples)
                        vVal_loss.append(float(val_crit))
                    if (best_loss - val_crit) / best_loss > tol_change:
                        no_decrease_counter = 0
                        best_loss = val_crit
                        best_model = self.clone()
                    else:
                        no_decrease_counter += 1
                    if scheduled:
                        scheduler.step(val_crit)
                    if no_decrease_counter >= patience_soft:
                        print("Updating criterion weights")
                        self.criterion.update()
                        patience_soft = 0
                    print("Epoch loss = {:.7f} || val loss = {:.7f} || Best val loss = {:.7f} \n".format(float(epoch_loss),
                          float(val_crit), float(best_loss)))
                if no_decrease_counter > patience:  # early stopping
                    break

                if (math.isnan(epoch_loss)):
                    print("Loss became nan -- training stopped")
                    break
                bar()
        train_time = time.perf_counter() - start_time

        print("Total dentification runtime : {} \n Best loss : {} \n".format(train_time, best_loss))

        if keep_best:
            self.sim_model = best_model

        # Final simulation perf on train data
        final_train_mse, bcheck, y_sim_list_train = self.eval_(train_set)

        # Final simulation on test data
        final_val_mse, bcheck_val, y_sim_list_val = self.eval_(test_set)
        print(" Final MSE = {:.10f} || Val_MSE = {:.10f} \n".format(float(final_train_mse), float(final_val_mse)))

        # Saving Mat-file training results
        weights, biases = self.sim_model.extract_weights()
        dictRes = {'weights': weights, 'biases': biases, 'best_loss': best_loss,
                   'Loss_hist': vLoss, 'Val_loss_hist': vVal_loss,
                   'y_sim_val': y_sim_list_val, 'y_sim_train': y_sim_list_train}
        return best_model, dictRes

    def eval_(self, val_set: ExperimentsDataset, max_idx: Optional[int] = None):
        '''
            This method evaluates the validation metric if one is given.
            It also checks all the potential constraints that a model has.
            It is assumed each constraint that has to be checked at a submodule
            level is checked at the parent module level.
        '''
        if self.val_criterion is not None:
            val_loss = 0.0
            y_sim_list = []
            if val_set.experiments:
                for exp in val_set.experiments:
                    u, y_true, x_true = exp.get_data(idx=max_idx)
                    x_sim, y_sim = self.sim_model.simulate(u, x_true[0, :])
                    y_sim_list.append(y_sim.detach().numpy())
                    val_loss += self.val_criterion(y_true, y_sim)
                val_crit = val_loss / (len(val_set.experiments))
        else:
            val_crit = math.inf
            y_sim_list = None

        # Now check if all constraints in the model are checked if there is any
        const, _ = self.sim_model.check_() if hasattr(self.sim_model, 'check_') else True, None
        return val_crit, const, y_sim_list

    def clone(self) -> Simulator:
        return self.sim_model.clone()

    def save(self):
        self.sim_model.save()

    def __str__(self) -> str:
        return f"{str(self.sim_model)}"

    def backtrack(self, *args, step_ratio=0.5, max_iter=100):
        with torch.no_grad():
            print(" Statrting backtracking")
            theta0 = flatten_params(self.sim_model)
            i = 0
            while i <= max_iter:
                crit = self.criterion(*args)
                if not is_legal(crit):
                    theta = theta0 * step_ratio + theta0 * (1 - step_ratio)
                    write_flat_params(self.sim_model, theta)
                    i += 1
                else:
                    break
            if i > max_iter:
                print("Maximum iterations reached")
            return crit


r'''
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


How to save a model

# Saving best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model.state_dict(),
            'loss': vLoss,
            'val_loss': vVal_Loss,
            'trainingInfo' : info}, strSavePath)

'''
