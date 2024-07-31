from alive_progress import alive_bar
import time
import torch
import math
from ..integrators.integrators import Simulator
from abc import ABC, abstractmethod
from ..utils.data import ExperimentsDataset
from torch.utils.data import DataLoader
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

    def __init__(self, sim_model: Simulator, loss: BaseLoss, val_loss: Union[BaseLoss, None],
                 optimizer: torch.optim.Optimizer, **kwargs) -> None:
        super(Trainer, self).__init__()
        self.sim_model = sim_model
        self.criterion = loss
        self.val_criterion = val_loss
        self.optimizer = optimizer



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

    def __init__(self, sim_model: Simulator, loss: BaseLoss, val_loss: BaseLoss, optimizer: torch.optim.Optimizer) -> None:
        super(SSTrainer, self).__init__(sim_model=sim_model, loss=loss, val_loss=val_loss, optimizer=optimizer)

    def fit_(self, train_set: ExperimentsDataset, test_set: ExperimentsDataset,
             seq_len: int = 30, batch_size: int = 256, lr: float = 1e-2, min_lr: float = 1e-5, test_freq: int = 10,
             patience: int = 10, threshold: float = 0.001, epochs: int = 1000, save_path: Optional[str] = None,
             keep_best: bool = True, max_val_samples: int = 3000, scheduled: bool = False, patience_soft: int = 5,
             backtrack: bool = False, device = torch.device('cpu')):

        # Setting saving and data attributes
        train_set.set_seq_len(seq_len=seq_len)
        self.sim_model.set_save_path(save_path)
        self.sim_model.to(device)
        
        # Initialize trainable hidden states with simulated values
        with torch.no_grad():
            for exp in train_set.experiments:
                if exp.x_trainable:
                    u, _, _ = exp.get_data()
                    x0 = torch.zeros(1, exp.nx, device=device)
                    x_sim, _ = self.sim_model(u.unsqueeze(0).to(device), x0)
                    exp.x = x_sim.squeeze(0).detach().clone()
                    exp.x.requires_grad = True

        # Consider making x a training parameter or not
        trainable_xs = [exp.x for exp in train_set.experiments if exp.x_trainable]
        x_param_group = {'params': trainable_xs, 'lr': lr}
        self.optimizer.add_param_group(x_param_group)

        # Learning rate and soft constraints scheduleur
        if scheduled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, threshold=threshold,
                                                               mode='min', factor=0.1, patience=patience_soft,
                                                               threshold_mode='rel')
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        no_improvement_count = 0
        train_losses, test_losses, check_constraints = [], [], []

        # Compute first loss
        best_val_loss, _, _ = self.eval_(val_set=test_set)
        test_losses.append(float(best_val_loss))
        print("Initial val_MSE = {:.7f} \n".format(float(best_val_loss)))
        best_model = self.sim_model.clone()
        
        current_loss = None  # Define current_loss in the outer scope
        states = None
        outputs = None

        def closure():
            nonlocal current_loss, outputs, states
            self.optimizer.zero_grad()
            states, outputs = self.sim_model(batch_u, batch_x0)
            loss = self.criterion(outputs, batch_y, x_pred=states, x_true=batch_x)
            loss.backward()
            current_loss = loss.item()
            return loss
        
        start_time = time.perf_counter()

        with alive_bar(total=epochs) as bar:
            for epoch in range(epochs):
                epoch_loss = 0.0
                for (batch_u, batch_y, batch_x, batch_x0) in train_loader:
                    batch_u, batch_y, batch_x0 = batch_u.to(device), batch_y.to(device), batch_x0.to(device)
                    
                    self.optimizer.step(closure)
                    if current_loss is not None:
                        epoch_loss += current_loss
                        current_loss = None

                    if backtrack and not is_legal(loss):
                        loss = self.backtrack(batch_y, outputs, batch_x, states)
                        epoch_loss += float(loss.item())
                
                average_train_loss = epoch_loss / len(train_loader)
                train_losses.append(average_train_loss)
                
                if epoch % test_freq == 0:
                    with torch.no_grad():
                        val_loss, check_result, y_sim = self.eval_(val_set=test_set, max_idx=max_val_samples)
                        test_losses.append(float(val_loss))
                        check_constraints.append(check_result)
                    if val_loss < best_val_loss * (1 - threshold):
                        no_improvement_count = 0
                        best_val_loss = val_loss
                        best_model = self.sim_model.clone()
                    else:
                        no_improvement_count += 1
                    
                    scheduler.step(val_loss)  # Use the scheduler
                    
                    if no_improvement_count >= patience_soft:
                        print(f"No improvement for {patience_soft} epochs. Updating regularizations.")
                        self.criterion.update()
                        patience_soft = 0
                        no_improvement_count = 0
                    
                    print(f"Epoch loss = {average_train_loss:.7f} || val loss = {val_loss:.7f} || "
                          f"Best val loss = {best_val_loss:.7f} || current lr : {scheduler.get_last_lr()[0]:.7f}")
                
                if no_improvement_count > patience:  # early stopping
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

                if math.isnan(epoch_loss):
                    print("Loss became nan -- training stopped")
                    break
                bar()
    
        train_time = time.perf_counter() - start_time

        print("Total dentification runtime : {} \n Best loss : {} \n".format(train_time, best_val_loss))

        if keep_best:
            self.sim_model = best_model

        # Final simulation perf on train data
        final_train_mse, bcheck, y_sim_list_train = self.eval_(train_set)

        # Final simulation on test data
        final_val_mse, bcheck_val, y_sim_list_val = self.eval_(test_set)
        print(" Final MSE = {:.10f} || Val_MSE = {:.10f} \n".format(float(final_train_mse), float(final_val_mse)))

        # Saving Mat-file training results
        weights, biases = self.sim_model.extract_weights()
        dictRes = {'weights': weights, 'biases': biases, 'best_loss': best_val_loss,
                   'train_loss_hist': train_losses, 'test_loss_hist': test_losses,
                   'y_sim_val': y_sim_list_val, 'y_sim_train': y_sim_list_train,
                   'final_train_sim_loss' : final_train_mse, 'final_val_sim_loss': final_val_mse}
        return best_model, dictRes

    def eval_(self, val_set: ExperimentsDataset, max_idx: Optional[int] = None):
        '''
        This method evaluates the validation metric on the test set and checks model constraints.
        It simulates all experiments in the test set up to a specified index and computes the average loss.
        '''
        val_loss = 0.0
        y_sim_list = []

        if val_set.experiments:
            # Determine the simulation length
            if max_idx is None:
                max_idx = min(exp.n_samples for exp in val_set.experiments)

            # Prepare batched inputs
            batch_u = torch.stack([exp.u[:max_idx] for exp in val_set.experiments])
            batch_y_true = torch.stack([exp.y[:max_idx] for exp in val_set.experiments])
            batch_x0 = torch.stack([exp.x[0] for exp in val_set.experiments])

            # Simulate all experiments at once
            with torch.no_grad():
                batch_x_sim, batch_y_sim = self.sim_model(batch_u, batch_x0)

            # Compute validation loss
            val_loss = self.val_criterion(batch_y_true, batch_y_sim)
            val_loss = val_loss / len(val_set.experiments)  # Average loss per experiment

            # Store simulation results if needed
            y_sim_list = batch_y_sim.detach().cpu().numpy()

        # Check model constraints
        check_result, check_info = self.sim_model.check_() if hasattr(self.sim_model, 'check_') else (True, None)

        return val_loss, check_result, y_sim_list

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
