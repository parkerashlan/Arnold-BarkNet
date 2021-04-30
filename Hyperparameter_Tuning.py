from ray import tune
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.schedulers import ASHAScheduler
import nevergrad as ng
from ray.tune import CLIReporter

import BarkNet_Pytorch
import DatasetMaker
import Image_Preprocessing_PyTorch

import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import os


def train_resnet(config, checkpoint_dir=os.path.abspath('./tune_chckpoints')):

    net = BarkNet_Pytorch.BarkNet()
    net.load_state_dict(torch.load('/home/aparker/PycharmProjects/PersonalBarkNet/models/pytorch_redux_fold_0_model'))

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'],
                                           gamma=config['gamma'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint')
        )

        net.load_state_dict(model_state)

        optimizer.load_state_dict(optimizer_state)

    DATA_PATH = '/home/aparker/PycharmProjects/local_objects_PersonalBarkNet/data/train_1_2/'

    dg = DatasetMaker.DatasetGenerator(path=DATA_PATH)

    k_folds = dg.stratified_kfold_cv()

    train_sets, val_sets = Image_Preprocessing_PyTorch.make_train_validation_set(ds=k_folds)

    train_dataset = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=DATA_PATH, data=train_sets[0])
    val_dataset = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=DATA_PATH, data=val_sets[0], val=True)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=6)

    dataloader = {'train': train_dataloader,
                  'val': val_dataloader}

    dataset_size = {'train': len(train_dataset),
                    'val': len(val_dataset)}

    # need to have training function within to report epoch acc and loss to RayTune
    num_epochs = 15
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
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
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save((net.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=epoch_loss, accuracy=epoch_acc)

if __name__ == "__main__":

    config = {
        'lr': tune.loguniform(1e-4, 1e-1),
        'gamma': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([8, 16, 32, 40]),
        'step_size': tune.choice([4, 5, 6, 7, 8, 9, 10])
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=25,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=['loss', 'accuracy', 'training_iteration']
    )

    current_best_params = [{
        'lr': 0.001,
        'gamma': 0.1,
        'batch_size': 32,
        'step_size': 7
    }]

    ng_search = NevergradSearch(
        optimizer=ng.optimizers.OnePlusOne,
        metric='loss',
        mode='min',
        points_to_evaluate=current_best_params
    )

    result = tune.run(
        train_resnet,
        search_alg=ng_search,
        resources_per_trial={'cpu':6},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))