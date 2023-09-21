import os
import time
import torch
import torch.nn as nn

class VisionBase(nn.Module):
    def __init__(self, params, device):
        super(VisionBase, self).__init__()
        self.train_params = params['train_params']
        self.device = device

    def get_feature_map(self, x):
        raise NotImplementedError()

    def forward(self, x, return_loss=False):
        raise NotImplementedError()

    def load_parameters(self, load_path):
        self.load_state_dict(torch.load(load_path, map_location=self.device)['state_dict'])

    def train_epochs(self, train_dataloader, validation_dataloader, load_path=None):
        train_params = self.train_params
        optimizer = train_params['optimizer']
        lr_scheduler = train_params['lr_scheduler']
        max_epoch_number = train_params['max_epoch_number']
        save_interval = train_params['save_interval']
        save_path = train_params['save_path'] 
        log_interval = train_params['log_interval']
        perturbation = train_params['perturbation']
        device = self.device
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if load_path is not None:
            checkpoint = torch.load(load_path, map_location=device)
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)
            current_epoch_numbers = checkpoint['current_epoch_numbers']
            loss_plot = checkpoint['loss_plot']
            optimizer = optimizer["optim_class"](self.parameters(), **(optimizer["args"]))
            lr_scheduler = lr_scheduler["lr_scheduler_class"](optimizer, last_epoch=current_epoch_numbers, **(lr_scheduler["args"]))
        else:
            optimizer = optimizer["optim_class"](self.parameters(), **(optimizer["args"]))
            lr_scheduler = lr_scheduler["lr_scheduler_class"](optimizer, **(lr_scheduler["args"]))
            current_epoch_numbers = 0
            loss_plot = []

        for e in range(current_epoch_numbers, max_epoch_number):
            self.train()
            running_loss = 0
            start = time.time()
            for i, (image, gt, kinematics) in enumerate(train_dataloader):
                self.zero_grad()
                data = {}
                if perturbation is not None:
                    image = perturbation(image/255) * 255
                data['image'] = image.to(device=device)
                data['gt'] = gt.to(device=device)
                data['kinematics'] = kinematics.to(device=device)
                pred, loss = self.forward(data, return_loss=True)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                elapsed = time.time() - start
                if (i+1) % log_interval == 0:
                    loss_plot.append(running_loss / (i+1))
                    print("Epoch_step : %d Loss: %f iteration per Sec: %f" %
                            (i+1, running_loss / (i+1), (i+1)*pred.size(0) / elapsed))
            print("Epoch : %d Loss: %f iteration per Sec: %f" %
                            (e, running_loss / (i+1), (i+1)*pred.size(0) / elapsed))
            lr_scheduler.step()
            if (e+1) % save_interval == 0:
                save_dict = {}
                save_dict['state_dict'] = self.state_dict()
                save_dict['current_epoch_numbers'] = e
                save_dict['loss_plot'] = loss_plot
                torch.save(save_dict, os.path.join(save_path,"model_"+str(e)+".pth"))
                self.eval()
                validation_loss = 0
                start = time.time()
                for i, (image, gt, kinematics) in enumerate(validation_dataloader):
                   data['image'] = image.to(device=device)
                   data['gt'] = gt.to(device=device)
                   data['kinematics'] = kinematics.to(device=device)
                   pred, loss = self.forward(data, return_loss=True)
                   validation_loss += loss.item()
                elapsed = time.time() - start
                print("Validation at epch : %d Validation Loss: %f iteration per Sec: %f" %
                            (e, validation_loss / (i+1), (i+1) / elapsed))
        return loss_plot