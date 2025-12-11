import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt)
        print(f"using {self.model.__class__.__name__}")

        if opt.head_type == "fc":
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        elif opt.head_type == "attention":
            for name, params in self.model.attention_head.named_parameters():
                torch.nn.init.normal_(params, 0.0, opt.init_gain)
        
        if opt.resume_path is not None:
            state_dict = torch.load(opt.resume_path)
            if self.opt.fix_backbone:
                if self.opt.head_type == "attention" or opt.head_type == "crossattention":
                    self.model.attention_head.load_state_dict(state_dict)
                else:
                    self.model.fc.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

        if opt.fix_backbone:
            params = []
            if opt.head_type == "fc":
                for name, p in self.model.named_parameters():
                    if  name=="fc.weight" or name=="fc.bias": 
                        params.append(p) 
                    else:
                        p.requires_grad = False
            elif opt.head_type == "mlp":
                for p in self.model.mlp.parameters():
                    params.append(p)
            elif opt.head_type == "attention" or opt.head_type == "crossattention":
                for p in self.model.attention_head.parameters():
                    params.append(p)

            elif opt.head_type == "transformer":
                params = [{'params': self.model.transformer_block.parameters()},
                {'params': self.model.fc.parameters()}]
                # params = self.model.transformer.parameters()
                # params["fc"] = self.model.fc.parameters()


        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.model = nn.parallel.DistributedDataParallel(self.model)
        self.model.to(opt.gpu_ids[0])



    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)
        # self.output = self.output.view(-1).unsqueeze(1)
        self.output = self.output




    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


