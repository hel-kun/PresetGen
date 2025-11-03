import torch
import torch.nn as nn
from config import DEVICE
from utils.param import CATEGORICAL_PARAM_NAMES, CONTINUOUS_PARAM_NAMES

class ParamsLoss(nn.Module):
    def __init__(
        self, 
        cont_weight=1.0,
        categ_weight=1.0,
        categ_param_weight=None,
        categ_class_weights=None
    ):
        super(ParamsLoss, self).__init__()
        self.cont_weight = cont_weight
        self.categ_weight = categ_weight
        self.categ_param_weight = categ_param_weight
        self.categ_class_weights = categ_class_weights

        self.cont_loss_fn = nn.MSELoss()
        if categ_class_weights is not None:
            self.categ_loss_fns = {}
            for param_name, class_weights in categ_class_weights.items():
                class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
                self.categ_loss_fns[param_name] = nn.CrossEntropyLoss(weight=class_weights_tensor, reduction='none')
        else:
            self.categ_loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, categ_pred, categ_target, cont_pred, cont_target):
        # 連続値パラメータの損失計算
        cont_losses = []
        for param_name in CONTINUOUS_PARAM_NAMES:
            if param_name in cont_pred and param_name in cont_target:
                param_loss = self.cont_loss_fn(cont_pred[param_name], cont_target[param_name])
                cont_losses.append(param_loss)

        if cont_losses:
            cont_loss = torch.stack(cont_losses).mean()
        else:
            cont_loss = torch.tensor(0.0, device=DEVICE)

        # カテゴリカルパラメータの損失計算
        categ_losses = []
        for param_name in CATEGORICAL_PARAM_NAMES:
            if param_name in categ_pred and param_name in categ_target:
                pred = cont_pred[param_name].squeeze(1) if cont_pred[param_name].dim() > 1 else cont_pred[param_name]
                if self.categ_class_weights is not None and param_name in self.categ_loss_fns:
                    param_loss = self.categ_loss_fns[param_name](pred, categ_target[param_name])
                else:
                    param_loss = self.categ_loss_fn(pred, categ_target[param_name])
                categ_losses.append(param_loss.mean())

        if categ_losses:
            categ_loss = torch.stack(categ_losses).mean()
        else:
            categ_loss = torch.tensor(0.0, device=DEVICE)

        total_loss = self.cont_weight * cont_loss + self.categ_weight * categ_loss
        return total_loss, self.categ_weight * categ_loss, self.cont_weight * cont_loss


# TODO: AudioEmbedLossの実装
# Synth1をCLI(Python)でうこかせるようにしないと実装できないかも
# Synth1以外のシンセサイザはdawdreamerで動かせる
class AudioEmbedLoss(nn.Module):
    pass