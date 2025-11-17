import torch
from torch.utils.data import DataLoader
from typing import Optional
import logging, tqdm, os
from model.model import PresetGenModel, TransformerModel
from model.loss import ParamsLoss, AudioEmbedLoss
from dataset.dataset import Synth1Dataset
from config import DEVICE
from utils.types import *
from utils.param import *
from utils.plot import plot_loss_curve

class Trainer():
    def __init__(
        self,
        model,
        dataset,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[torch.nn.Module] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: str = 'checkpoints',
        log_interval=10,
        early_stopping_patience=10,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.logger = logger or logging.getLogger(__name__)

        dataset = dataset
        self.train_dataloader = DataLoader(dataset.dataset['train'], batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.dataset['validation'], batch_size=32, shuffle=False, collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.dataset['test'], batch_size=32, shuffle=False, collate_fn=dataset.collate_fn)
    
    def train(self, num_epochs: int, resume_from_checkpoint: Optional[str] = None) -> None:
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self._load_checkpoint(resume_from_checkpoint)
            self.logger.info(f"Resumed training from checkpoint: {resume_from_checkpoint} at epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_loss = 0.0

            train_bar = tqdm.tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", postfix="total_loss=0.0, categ_loss=0.0, cont_loss=0.0")
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                texts_batch, tensor_batch, params_batch = batch
                for key in tensor_batch:
                    tensor_batch[key] = tensor_batch[key].to(DEVICE)
                for key in params_batch['categ']:
                    params_batch['categ'][key] = params_batch['categ'][key].to(DEVICE)
                for key in params_batch['cont']:
                    params_batch['cont'][key] = params_batch['cont'][key].to(DEVICE)

                outputs = self.model(texts_batch, tgt=tensor_batch)
                total_loss, categ_loss, cont_loss = self.criterion(
                    categ_pred=outputs['categorical'],
                    categ_target=params_batch['categ'],
                    cont_pred=outputs['continuous'],
                    cont_target=params_batch['cont']
                )
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()
                if (batch_idx + 1) % self.log_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    train_bar.set_postfix_str(f"total_loss={total_loss:.4f}, categ_loss={categ_loss.item():.4f}, cont_loss={cont_loss.item():.4f}")
                train_bar.update(1)
            train_bar.close()

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            self.train_losses.append(avg_epoch_loss)

            if self.val_dataloader is not None:
                val_loss = self.evaluate(self.val_dataloader)
                self.val_losses.append(val_loss)

                self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.early_stopping_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        self.logger.info("Early stopping triggered.")
                        break

            plot_loss_curve(self.train_losses, self.val_losses, save_path=f"{self.checkpoint_path}/loss_curve.png")
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

    def evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                texts, tensor_batch, params_batch = batch
                for key in tensor_batch:
                    tensor_batch[key] = tensor_batch[key].to(DEVICE)
                for key in params_batch['categ']:
                    params_batch['categ'][key] = params_batch['categ'][key].to(DEVICE)
                for key in params_batch['cont']:
                    params_batch['cont'][key] = params_batch['cont'][key].to(DEVICE)
                outputs = self.model(texts, tgt=tensor_batch)
                loss, _, _ = self.criterion(
                    categ_pred=outputs['categorical'],
                    categ_target=params_batch['categ'],
                    cont_pred=outputs['continuous'],
                    cont_target=params_batch['cont']
                )
                total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def evaluate_detailed(self, data_loader: Optional[DataLoader] = None) -> None:
        self.model.eval()
        data_loader = data_loader if data_loader is not None else self.test_dataloader

        total_loss = 0.0
        cont_mae = 0.0
        cont_count = 0
        categ_accuracies = {}
        categ_totals = {}
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc="Detailed Evaluation"):
                # 全体のLoss計算
                texts, tensor_batch, params_batch = batch
                for key in tensor_batch:
                    tensor_batch[key] = tensor_batch[key].to(DEVICE)
                for key in params_batch['categ']:
                    params_batch['categ'][key] = params_batch['categ'][key].to(DEVICE)
                for key in params_batch['cont']:
                    params_batch['cont'][key] = params_batch['cont'][key].to(DEVICE)
                outputs = self.model(texts, tgt=None)
                loss, _, _ = self.criterion(
                    categ_pred=outputs['categorical'],
                    categ_target=params_batch['categ'],
                    cont_pred=outputs['continuous'],
                    cont_target=params_batch['cont']
                )
                total_loss += loss.item()

                # ContinuousパラメータのMAE計算
                cont_pred = outputs['continuous']
                cont_target = params_batch['cont']
                for param_name in cont_pred.keys():
                    if param_name in cont_target:
                        pred_values = cont_pred[param_name]
                        true_values = cont_target[param_name]
                        cont_mae += torch.abs(pred_values - true_values).sum().item()
                        cont_count += true_values.numel()

                # CategoricalパラメータのAccuracy計算
                categ_pred = outputs['categorical']
                categ_target = params_batch['categ']
                for param_name in categ_target.keys():
                    pred_labels = torch.argmax(categ_pred[param_name], dim=1)
                    true_labels = categ_target[param_name]
                    correct = (pred_labels == true_labels).sum().item()
                    total = true_labels.size(0)

                    if param_name not in categ_accuracies:
                        categ_accuracies[param_name] = 0
                    if param_name not in categ_totals:
                        categ_totals[param_name] = 0

                    categ_accuracies[param_name] += correct
                    categ_totals[param_name] += total

        categ_correct = sum(categ_accuracies.values())
        categ_total = sum(categ_totals.values())
        cont_mae = cont_mae / cont_count if cont_count > 0 else 0.0
        avg_loss = total_loss / len(data_loader)

        self.logger.info(f"Detailed Evaluation - Total Avg Loss: {avg_loss:.4f}")
        self.logger.info(f"Continuous Params MAE: {cont_mae:.4f}")
        self.logger.info(f"Categorical Params Overall Accuracy: {categ_correct / categ_total if categ_total > 0 else 0.0:.4f}")

        for param_name in categ_accuracies.keys():
            accuracy = categ_accuracies[param_name] / categ_totals[param_name] if categ_totals[param_name] > 0 else 0.0
            self.logger.info(f"Categorical Param: {param_name}, Accuracy: {accuracy:.4f}")

        # save detailed results to a text file
        with open(f"{self.checkpoint_path}/detailed_evaluation.txt", "w") as f:
            f.write(f"Detailed Evaluation Results\n")
            f.write(f"Total Avg Loss: {avg_loss:.4f}\n")
            f.write(f"Continuous Params MAE: {cont_mae:.4f}\n")
            f.write(f"Categorical Params Overall Accuracy: {categ_correct / categ_total if categ_total > 0 else 0.0:.4f}\n")
            for param_name in categ_accuracies.keys():
                accuracy = categ_accuracies[param_name] / categ_totals[param_name] if categ_totals[param_name] > 0 else 0.0
                f.write(f"Categorical Param: {param_name}, Accuracy: {accuracy:.4f}\n")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        path: str = f"{self.checkpoint_path}/checkpoint_epoch_{epoch+1}.pth" if not is_best else f"{self.checkpoint_path}/best_model.pth"

        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        if is_best:
            self.logger.info(f"Saved best model checkpoint to {path}")

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        return checkpoint['epoch']
    