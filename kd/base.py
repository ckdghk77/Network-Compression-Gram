import torch
from torch import nn
import numpy as np

class KD_Base(nn.Module):
    def __init__(self, s_model: nn.Module, t_model: nn.Module,):
        super().__init__()
        self.feature_dict = {};
        self.n_f_dict = {}
        self.f_n_dict = {}

        self._hooks = []
        self._hook_all_layers(t_model, prefix="teacher_");
        self._hook_all_layers(s_model, prefix="student_");

        self.t_model = t_model
        self.s_model = s_model

    def _hook_all_layers(self, net, prefix=""):
        def _hook_fn(m, i, o):
            self.feature_dict[self.f_n_dict[m]] = o

        for name, layer in net._modules.items():
            # If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if len(layer._modules):
                self._hook_all_layers(layer, '{}_'.format(prefix + name))
            else:
                # it's a non sequential. Register a hook
                self.n_f_dict[prefix+name] = layer
                self.f_n_dict[layer] = prefix+name
                self._hooks.append(layer.register_forward_hook(_hook_fn))

    def remove_all_hooks(self,):
        [hook.remove() for hook in self._hooks]

    def forward(self, input) :
        return self.s_model(input);

    def iterate_teaching(self, dloader, optimizer, scheduler, gpu=False):
        losses = []

        for b_idx, (data, label) in enumerate(dloader) :
            if gpu :
                data = data.cuda();
            with torch.no_grad() :
                self.t_model(data);
            self.s_model(data);

            optimizer.zero_grad()
            loss = self.estm_loss();
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(self.s_model.parameters(),
                                           0.1)
            optimizer.step()
        scheduler.step()
        return np.mean(losses)


