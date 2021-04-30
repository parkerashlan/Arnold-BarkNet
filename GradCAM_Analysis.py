import torch
from GradCAM import GradCAM
import cv2
import Image_Preprocessing_PyTorch
import DatasetMaker
from torchvision import transforms
import numpy as np
import matplotlib.cm as cm
import BarkNet_Pytorch
from torchvision import models




def gradcam(model, dataset, grad_dataset, target_layers, target_class):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    fnames = [name for _, _, name in dataset]

    sampled_images = [img for img, _, _ in dataset]

    images = torch.stack(tuple(sampled_images)).to(device)

    gcam = GradCAM(model=model, candidate_layers=target_layers)

    probs, ids = gcam.forward(images)

    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)

    gcam.backward(ids=ids_)

    raw_images = [np.array(img) for img, _, _ in grad_dataset]

    for layer in target_layers:

        regions = gcam.generate(target_layer=layer)

        for j in range(len(images)):
            heatmap = regions[j, ...].cpu().numpy().squeeze()

            # ellipses is saying to select all values from first two dimension then only the first 3 dimension
            cmap = cm.jet_r(heatmap)[..., :3] * 255.0

            superimposed_img = (cmap.astype(np.float) + raw_images[j].astype(np.float)) / 2
            cv2.imwrite(f'./heatmaps/resnet34_{layer}_{target_class}_{fnames[j]}', np.uint8(superimposed_img))


if __name__ == "__main__":
    PATH = '/home/aparker/PycharmProjects/local_objects_PersonalBarkNet/data/test_data_1_2/'

    dg = DatasetMaker.DatasetGenerator(path=PATH)

    test_df = dg.stratified_kfold_cv(folds=1)[0]

    sample1 = test_df[test_df['fold 1'] == '1_0_PON_OnePlus7_0161_0013.jpg']  # misclassified as 2
    sample2 = test_df[test_df['fold 1'] == '1_0_PON_OnePlus7_0159_0005.jpg']  # misclassified as 2
    sample3 = test_df[test_df['fold 1'] == '1_0_PON_OnePlus7_0162_0000.jpg']  # classified correctly
    sample4 = test_df[test_df['fold 1'] == '2_0_CAL_OnePlus7_0045_0002.jpg']  # classified correctly

    samples1 = sample1.append(sample3)
    samples2 = sample2.append(sample4)

    transformations = transforms.Compose([
        transforms.Resize((2016, 896)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    grad_transform = transforms.Compose([
        transforms.Resize((2016, 896)),
        transforms.CenterCrop(224)
    ])

    test_dataset1 = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=PATH, data=samples1,
                                                           transform=transformations, test=True)

    grad_dataset1 = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=PATH, data=samples1,
                                                           transform=grad_transform, test=True)

    test_dataset2 = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=PATH, data=samples2,
                                                            transform=transformations, test=True)

    grad_dataset2 = Image_Preprocessing_PyTorch.BarkDataset(path_to_data=PATH, data=samples2,
                                                           transform=grad_transform, test=True)

    model = BarkNet_Pytorch.BarkNet()
    model.load_state_dict(torch.load('./models/pytorch_redux_fold_0_model'))

    print(model)

    gradcam(model, test_dataset1, grad_dataset1, target_layers=['layer4.0'], target_class=1)
    gradcam(model, test_dataset2, grad_dataset2, target_layers=['layer4.0'], target_class=2)


