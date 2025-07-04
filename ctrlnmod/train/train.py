import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, Timer, Callback, ModelCheckpoint
from ctrlnmod.optim import ProjectedOptimizer, BackTrackOptimizer, project_to_pos_def
import os
from typing import Callable, Optional

'''
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

class StopTrainingCallback(Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.stop_training_flag:
            print("\n Stopping training because the boolean condition is set to True.")
            trainer.should_stop = True

class LRSchedulerLogger(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Assuming the LR is scheduled per epoch
        lrs = [group['lr'] for group in trainer.optimizers[0].param_groups]
        print(f"\n Epoch {trainer.current_epoch}: Learning rate(s): {lrs}")

class LitNode(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Callable,
        val_criterion: Callable,
        lr: float,
        patience_soft: int = 30,
        use_backtracking: bool = False,
        use_projection: bool = False,
        condition_fn: Optional[Callable] = None,
        custom_logging_fn: Optional[Callable] = None,
        log_gradient_norms: bool = False,
        val_idx_max = None
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.val_criterion = val_criterion
        self.lr = lr
        self.best_val_loss = float('inf')
        self.patience_soft = patience_soft
        self.no_decrease_counter = 0
        self.use_backtracking = use_backtracking
        self.use_projection = use_projection
        self.condition_fn = condition_fn
        self.custom_logging_fn = custom_logging_fn
        self.log_gradient_norms = log_gradient_norms
        self.timer = Timer()
        self.stop_training_flag = False
        self.val_idx_max = val_idx_max
        self.best_model = model.clone()
        if self.use_backtracking and self.use_projection:
            raise ValueError("Cannot use both backtracking and projection. Choose one or neither.")
        # Cannot save if there is logdetregularization since it is potentially linked to a parameterize model
        self.save_hyperparameters(ignore=['model', 'criterion'])  

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        if self.use_projection:
            optimizer = ProjectedOptimizer(optimizer, project_to_pos_def, self.model)
        elif self.use_backtracking:
            optimizer = BackTrackOptimizer(optimizer, self.model, self.condition_fn or (lambda x: True))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=self.patience_soft,min_lr=1e-5),
                "monitor": "val_loss",
                "strict": True,
                "frequency": 1}
                }

    def forward(self, u, x0, d=None):
        return self.model(u, x0, d)

    def training_step(self, train_batch, batch_idx):
        
        batch_u, batch_y, batch_x, batch_x0, batch_d = train_batch
        states, outputs = self(batch_u, batch_x0, batch_d)
        loss = self.criterion(outputs, batch_y, x_pred=states, x_true=batch_x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_u, batch_y_true, batch_x0, batch_d = batch
        if self.val_idx_max is not None:
            batch_u = batch_u[:,:self.val_idx_max, :]
            batch_y_true = batch_y_true[:,:self.val_idx_max, :]
            batch_d = batch_d[:,:self.val_idx_max, :] if batch_d is not None else None
        with torch.no_grad():
            batch_x_sim, batch_y_sim = self(batch_u, batch_x0, batch_d)
            val_loss = self.val_criterion(batch_y_true, batch_y_sim)

        self.log('val_loss', val_loss, prog_bar=True)

        # Custom logging
        if self.custom_logging_fn:
            try:
                custom_logs = self.custom_logging_fn(self.model, batch_u, batch_y_true, batch_y_sim, self.criterion)
                for log_name, log_value in custom_logs.items():
                    self.log(log_name, log_value)
            except Exception as e:
                print(f"Error in custom logging function: {e}")
                
            if 'is_solve_ok' in custom_logs and  (not custom_logs['is_solve_ok']):
                self.stop_training_flag = True  # Stop training if lmi is false
        return {'val_loss': val_loss, 'y_sim': batch_y_sim.detach()}

    def on_train_start(self):
        self.version = self.logger.version if isinstance(self.logger, TensorBoardLogger) else 0

    def on_train_epoch_end(self):
        self.timer.time_elapsed('train')
        self.timer.time_elapsed('validate')
        
    def on_after_backward(self):
        if self.log_gradient_norms:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param_norm = param.norm()
                    self.log(f'gradient_norm_normed/{name}', param.grad.norm()/param_norm, on_epoch=True)

    def get_res_dir(self):
        return os.path.join(self.logger.log_dir)

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.callback_metrics.get('val_loss')
        if avg_val_loss is not None:
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.no_decrease_counter = 0

                # Store current best_model
                self.best_model = self.model.clone()
            else:
                self.no_decrease_counter += 1

            if self.no_decrease_counter >= self.patience_soft:
                if self.criterion.regularizers and hasattr(self.criterion, 'update') and callable(self.criterion.update):
                    print("Updating criterion weights")
                    self.criterion.update()
                    self.no_decrease_counter = 0

            self.log('no_decrease_counter', self.no_decrease_counter, on_epoch=True)
        epoch_time = self.timer.time_elapsed('validate')
        self.log('val_epoch_time', epoch_time)

        
    @classmethod
    def load_model(cls, checkpoint_path, model):
        """Méthode helper pour charger le modèle plus facilement"""
        dict = torch.load(checkpoint_path, weights_only=False)
        
        simulator_dict = dict['hyper_parameters']['simulator_dict']
        ss_model_dict = dict['hyper_parameters']['ss_model_dict']
        return cls.load_from_checkpoint(
            checkpoint_path,
            model=model
        )
    

def train_model(lit_model, data_module, logger, epochs, patience=100):

    # Callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=patience, verbose=True, mode='min')
    timer_callback = Timer()
    break_callback = StopTrainingCallback()
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',                # Métrique à surveiller
    filename='best-model-{epoch:03d}-v{logger.version:03d}',  # Format du nom
    save_top_k=1,                      # Garde uniquement le meilleur
    mode='min',                        # Car on minimise la loss
    save_weights_only=False,           # Sauvegarde le modèle complet
    save_last=False                    # Ne sauvegarde pas le dernier modèle
)
    store_lr_callback = LRSchedulerLogger()

    trainer = pl.Trainer(logger=logger, num_sanity_val_steps=0, max_epochs=epochs, log_every_n_steps=1,
                         callbacks=[early_stop_callback, timer_callback, break_callback, 
                                    checkpoint_callback, store_lr_callback])
    trainer.fit(lit_model, datamodule=data_module)

    return trainer

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
