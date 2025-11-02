import torch
import torch.nn as nn

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

    def forward(
        self,
        categ_pred,
        categ_target,
        cont_pred,
        cont_target
    ):
        cont_loss = 0.0

        for name in cont_target.keys():
            pred = cont_pred[name]
            target = cont_target[name]
            cont_loss += self.cont_loss_fn(pred, target)
        cont_loss /= len(cont_target)

        # Categorical Lossの計算
        categ_loss = 0.0
        for i, param_name in enumerate(categ_target.keys()):
            loss_fn = self.categ_loss_fns[param_name] if self.categ_class_weights is not None else self.categ_loss_fn
            pred = categ_pred[param_name]
            target = categ_target[param_name]
            loss = loss_fn(pred, target)
            if self.categ_param_weight is not None and param_name in self.categ_param_weight:
                loss = loss * self.categ_param_weight[param_name]
            categ_loss += loss.mean()
        categ_loss /= len(categ_target)

        total_loss = self.cont_weight * cont_loss + self.categ_weight * categ_loss
        return total_loss, self.cont_weight * cont_loss, self.categ_weight * categ_loss

# TODO: AudioEmbedLossの実装
# Synth1をCLI(Python)でうこかせるようにしないと実装できないかも
# Synth1以外のシンセサイザはdawdreamerで動かせる
class AudioEmbedLoss(nn.Module):
    pass