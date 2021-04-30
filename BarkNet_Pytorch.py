import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F

import Image_Preprocessing_PyTorch
import DatasetMaker


class BarkNet(nn.Module):

    def __init__(self):
        super(BarkNet, self).__init__()

        resnet = models.resnet34(pretrained=True)

        # freeze first layer of resnet to finetune rest of network
        for i, layer in enumerate(resnet.children()):
            if i == 5:
                break
            for param in layer.parameters():
                param.requires_grad = False

        modules = list(resnet.children())[:-2]

        # need to separate layers for GradCAM to register hooks
        self.conv1 = nn.Sequential(modules[0])
        self.bn1 = nn.Sequential(modules[1])
        self.relu = nn.Sequential(modules[2])
        self.maxpool = nn.Sequential(modules[3])
        self.layer1 = nn.Sequential(modules[4])
        self.layer2 = nn.Sequential(modules[5])
        self.layer3 = nn.Sequential(modules[6])
        self.layer4 = nn.Sequential(modules[7])

        # self.features = nn.Sequential(*modules)

        num_ftrs = resnet.fc.in_features
        self.Linear = nn.Linear(num_ftrs, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)

        x = x.view(x.shape[0], -1)

        x = self.Linear(x)

        return x


def train_model(model, criterion, optimizer, scheduler, dataloader,
                dataset_size, model_save_path, num_epochs=15):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # multiply by batch size since CrossEntropy is reduced using
                # the mean
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            with open('pytorch_tuned_model_metrics', 'a+') as f:
                f.write(f'{phase},{epoch_loss:.4f},{epoch_acc:.4f}')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}' + '\n')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":

    BATCH = 16
    DATA_PATH = '/home/aparker/PycharmProjects/local_objects_PersonalBarkNet/data/train_1_2/'

    dg = DatasetMaker.DatasetGenerator(path=DATA_PATH)

    k_folds = dg.stratified_kfold_cv(folds=2)

    train_sets, val_sets = Image_Preprocessing_PyTorch.make_train_validation_set(ds=k_folds)

    model = BarkNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00023)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.047)

    for i in range(6):
        train_dataset = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=DATA_PATH, data=train_sets[i])
        val_dataset = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=DATA_PATH, data=val_sets[i], val=True)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True)

        dataloaders = {'train': train_dataloader,
                       'val': val_dataloader}

        dataset_sizes = {'train': len(train_dataset),
                         'val': len(val_dataset)}

        model = train_model(model=model, criterion=criterion, optimizer=optimizer,
                                      scheduler=exp_lr_scheduler, dataloader=dataloaders,
                                      dataset_size=dataset_sizes,
                                      model_save_path=f'./models/pytorch_tuned_fold_{i}_model')


