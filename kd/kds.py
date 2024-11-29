from kd.base import KD_Base
from torch import nn
import torch
import torch.nn.functional as F

class KD_pred(KD_Base):
    def __init__(self, s_model: nn.Module, t_model:nn.Module):
        super().__init__(s_model = s_model, t_model=t_model)
        self.features = ['fc']

    def estm_loss(self):
        loss = 0.0
        for feature in self.features :
            pred = self.feature_dict['student_{}'.format(feature)]
            target = self.feature_dict['teacher_{}'.format(feature)]
            closs = -(F.softmax(target / 2, dim=1) * torch.log(F.softmax(pred, dim=1) + 1e-6));
            loss += torch.clamp(closs - 0.1, min=0.0).sum(1).mean();

        return loss

class KD_fit_BN(KD_Base) :
    def __init__(self, s_model: nn.Module, t_model:nn.Module):
        super().__init__(s_model = s_model, t_model=t_model)
        self.features = ['bn1', 'layer1_2_bn3', 'layer2_3_bn3', 'layer3_5_bn3', 'layer4_2_bn3']

    def estm_loss(self):
        loss = 0.0
        for feature in self.features:
            pred = self.feature_dict['student_{}'.format(feature)]
            target = self.feature_dict['teacher_{}'.format(feature)]
            floss = F.mse_loss(pred, target, reduction='none')#/len(self.features)
            floss = torch.clamp(floss-0.1, min=0.0).mean();
            loss += floss
        return loss


class KD_fit_GRAM(KD_Base) :
    def __init__(self, s_model: nn.Module, t_model:nn.Module):
        super().__init__(s_model = s_model, t_model=t_model)
        self.features_C = [('layer1_1_conv1', 'layer1_2_bn3'),
                           ('layer2_1_conv1', 'layer2_3_bn3'),
                           ('layer3_1_conv1', 'layer3_5_bn3'),
                           ('layer4_1_conv1', 'layer4_2_bn3')]

    def estm_loss(self):
        loss = 0.0
        for (f1,f2) in self.features_C:
            sf1,sf2 = self.feature_dict['student_{}'.format(f1)], self.feature_dict['student_{}'.format(f2)]
            B, C, W, H = sf1.shape;
            B, C2, W, H = sf2.shape;
            sf1 = sf1.view(B,C,W*H);
            sf2 = sf2.view(B,C2,W*H).permute(0,2,1);
            Gs = torch.bmm(sf1, sf2)/(W*H);

            tf1,tf2 = self.feature_dict['teacher_{}'.format(f1)], self.feature_dict['teacher_{}'.format(f2)]
            B, C, W, H = tf1.shape;
            B, C2, W, H = tf2.shape;
            tf1 = tf1.view(B, C, W * H);
            tf2 = tf2.view(B, C2, W * H).permute(0, 2, 1);
            Gt = torch.bmm(tf1, tf2) / (W * H);

            gloss = F.mse_loss(Gs,Gt, reduction='none');
            gloss = torch.clamp(gloss-1.0, min=0.0).mean()
            loss += gloss

        return loss*0.1


class KD_fit_RL(KD_Base) :
    def __init__(self, s_model: nn.Module, t_model:nn.Module):
        super().__init__(s_model = s_model, t_model=t_model)
        self.features = ['relu', 'layer1_2_relu',
                         'layer2_3_relu',
                         'layer3_5_relu',
                         'layer4_2_relu']

    def estm_loss(self):
        loss = 0.0
        for feature in self.features:
            pred = self.feature_dict['student_{}'.format(feature)]
            target = self.feature_dict['teacher_{}'.format(feature)]
            loss += F.mse_loss(pred, target)#/len(self.features)
        return loss


class KD_fit_APOOL(KD_Base) :
    def __init__(self, s_model: nn.Module, t_model:nn.Module):
        super().__init__(s_model = s_model, t_model=t_model)
        self.features = ['avgpool']

    def estm_loss(self):
        loss = 0.0
        for feature in self.features:
            pred = self.feature_dict['student_{}'.format(feature)]
            target = self.feature_dict['teacher_{}'.format(feature)]
            mloss = F.mse_loss(pred, target, reduction='none')#/len(self.features)
            loss += torch.clamp(mloss-0.001, min=0.0).mean()
        return loss

