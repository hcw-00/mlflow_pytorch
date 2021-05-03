
import argparse
import mlflow
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import shutil
from sklearn.metrics import confusion_matrix, classification_report #, accuracy_score
import time

from network import vgg16
from mlflow.tracking import MlflowClient
import mlflow.pytorch
import yaml

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def data_transforms(load_size=256, input_size=224):
    data_transforms = {
        'train' : transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(load_size),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean_train,
                                std=std_train)]),
        'val' : transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(load_size),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])}
    return data_transforms


def get_args():

    parser = argparse.ArgumentParser(description='CLASSIFICATION')
    parser.add_argument('--dataset_path', default=r'/home/changwoo/hdd/datasets/REVIEW_BOE_HKC_WHTM/BOE_HKC_WHTM_210219') #D:\Dataset\BOE_B11\BOE_B11_20191028_Evaluation_image_original_crop_2\dataset_1_512_jpg')
    parser.add_argument('--num_epoch', default=1)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--use_batchnorm', default=False)
    args = parser.parse_args()
    return args

def fit():
    for epoch in range(num_epochs):
        print('-'*20)
        print('Time consumed : {}s'.format(time.time()-start_time))
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('Dataset size : Train set - {}, Validation set - {}'.format(dataset_sizes['train'],dataset_sizes['val']))
        print('-'*20)
        
        for phase in ['train', 'val']: # train -> validation
            if phase == 'train':
                print("Train Phase")
                model.train()
            else:
                print("Validation Phase")
                model.eval()
            
            pred_list = []
            label_list = []
            train_loss_list = []
            val_loss_list = []
            for idx, (batch, labels) in enumerate(dataloaders[phase]): # batch loop
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(batch)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) # outputs : one-hot shape (ex. [0.1, -0.013, 0.345]) , labels : class number (ex. 3)
                    if phase == 'train':
                        train_loss_list.append(float(loss.data))
                        loss.backward()
                        optimizer.step()
                    if phase == 'val':
                        pred_list.extend(preds.tolist())
                        label_list.extend(labels.tolist())
                        val_loss_list.append(float(loss.data))

                if idx%10 == 0:
                    print('Epoch : {} | Loss : {:.4f}'.format(epoch, float(loss.data)))

            if phase == 'val':
                cr = classification_report(label_list, pred_list, output_dict=True)
                print(cr)

    return cr


if __name__ == '__main__':
    # torch.cuda.set_device(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))
    
    args = get_args()
    num_epochs = args.num_epoch
    lr = args.lr
    batch_size = args.batch_size
    load_size = args.load_size
    input_size = args.input_size
    use_batchnorm = args.use_batchnorm
    arch = 'vgg16_bn' if use_batchnorm else 'vgg16'

    # Prepare dataset
    data_transforms = data_transforms()
    image_datasets = {x: datasets.ImageFolder(root=os.path.join(args.dataset_path, x), \
        transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, \
        shuffle=True, num_workers=0, pin_memory=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    class_dict = {v: k for k, v in image_datasets['train'].class_to_idx.items()} # invert "class_to_idx" => {label:'classname',...}
    print(class_names)
    print(class_dict)

    # Define Network and Fit
    model = vgg16(arch=arch, pretrained=True, batch_norm=use_batchnorm, num_classes=len(class_names)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    
    cr = fit()
    
    print('Time : {}'.format(time.time() - start_time))
        
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(model, "model")
        # convert to scripted model and log the model
        scripted_pytorch_model = torch.jit.script(model)
        mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_metric("precision",cr['accuracy'])
        mlflow.log_metric("precision",cr['weighted avg']['precision'])
        mlflow.log_metric("recall",cr['weighted avg']['recall'])
        mlflow.log_metric("f1-score",cr['weighted avg']['f1-score'])

    # Fetch the logged model artifacts
    print("run_id: {}".format(run.info.run_id))
    for artifact_path in ["model/data", "scripted_model/data"]:
        artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)]
        print("artifacts: {}".format(artifacts))    
    
    # Generate conda env yaml file
    env = mlflow.pytorch.get_default_conda_env()
    print("conda env: {}".format(env))
    with open('conda.yaml', 'w') as f:
        yaml.dump(env, f)