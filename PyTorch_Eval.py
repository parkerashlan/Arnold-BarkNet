import BarkNet_Pytorch
import Image_Preprocessing_PyTorch
import DatasetMaker

import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = BarkNet_Pytorch.BarkNet().to(device)
model.load_state_dict(torch.load('./models/pytorch_tuned_fold_0_model'))

transformations = transforms.Compose([
    transforms.Resize((2016, 896)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

BATCH = 32
PATH = '/home/aparker/PycharmProjects/local_objects_PersonalBarkNet/data/test_data_1_2/'

test_dg = DatasetMaker.DatasetGenerator(path=PATH)
test_fold = test_dg.stratified_kfold_cv(folds=1)

test_dataset = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=PATH, data=test_fold[0],
                                                       transform=transformations, test=True)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=True)

model.eval()

preds = []
label = []
names = []
for inputs, labels, name in test_dataloader:

    with torch.no_grad():
        out = model(inputs.to(device))
        _, prediction = torch.max(out, 1)
        preds.append(prediction.cpu().numpy())
        label.append(labels.cpu().numpy())
        names.append(name)

pred = np.hstack(preds)
true = np.hstack(label)
names = np.hstack(names)

print(f'Accuracy is: {np.sum(pred == true)/len(test_dataset)}')

# Get array of mislabelled images
pred_col = pred.reshape(-1, 1)
true_col = true.reshape(-1, 1)
names_col = names.reshape(-1, 1)

images = np.hstack((pred_col, true_col, names_col))

mislabelled_mask = images[:, 0].astype(np.float) != images[:, 1].astype(np.float)
mislabelled = images[mislabelled_mask]
print('Mislabeled Images:',mislabelled)

