import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as toptim
from torch.utils.data import Dataset, DataLoader
import time
from transform import get_train_transforms,get_test_transforms
from model import get_model
from apex import amp
from torch.optim import AdamW
import numpy as np
import sys
import imageio
import torchvision.transforms as transforms


# ------------------------------------config------------------------------------
class config:
    # all the seed
    seed = 26
    use_seed = True

    # input image size
    img_size = 512

    # use which model
    model_name = "Unet"

    # continue train from old model, if not, load pretrained data
    from_old_model = True

    # whether use apex or not
    use_apex = True

    # learning rate
    learning_rate = 1e-4
    # max epoch
    epochs = 100
    # batch size
    batchSize = 4

    # if acc is more than this value, start save model
    lowest_save_F1 = 0.9137

    # loss function
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()

    # create optimizer
    # optimizer_name = "SGD"
    # optimizer_name = "Adam"
    optimizer_name = "AdamW"

    # model output
    output_channel = 1

    # read data from where
    read_data_from = "Memory"
    #read_data_from = "Disk"

    # ------------------------------------path set------------------------------------
    train_img_path = "Training/original_retinal_images/"
    train_img_names = [str(i)+"_training.tif" for i in range(21,41)]
    train_label_path = "Training/blood_vessel_segmentation_masks/"
    train_label_names = [str(i) + "_manual1.gif" for i in range(21, 41)]

    test_img_path = "Test/original_retinal_images/"
    test_img_names = [str(i) + "_test.tif" for i in range(1, 21)]
    test_label_path = "Test/blood_vessel_segmentation_masks/"
    test_label_names = [str(i) + "_manual1.gif" for i in range(1, 21)]

    log_name = "log.txt"
    model_path = "save_model_" + model_name + ".pth"


# record best model with (epoch_num, last_best_F1)
best_val_F1 = (-1, config.lowest_save_F1)


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if config.use_seed:
    seed_torch(seed=config.seed)

# ------------------------------------dataset------------------------------------
if config.read_data_from == "Memory":
    # create dataset
    class blood_vessels(Dataset):
        def __init__(self, img_path, img_names, label_path, label_names, transform):
            # get lists
            self.transform = transform
            self.data = []

            for i in range(len(img_names)):
                img = cv2.imread(img_path + img_names[i], 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(image=img)['image']
                label = np.array(imageio.mimread(label_path + label_names[i]))[0]
                label = cv2.resize(label, (config.img_size, config.img_size))
                label = np.array([label])
                label = label / 255
                self.data.append((img, label))

        def __getitem__(self, index):
            img, label = self.data[index]
            return img, label

        def __len__(self):
            return len(self.data)

elif config.read_data_from == "Disk":
    pass

# ------------------------------------train------------------------------------
def train(net, train_loader, criterion, optimizer, epoch, device, log):
    # start train
    runningLoss = 0
    loss_count = 0

    batch_num = len(train_loader)
    for index, (imgs, labels) in enumerate(train_loader):
        # send data to device
        imgs, labels = imgs.to(device), labels.to(device)

        # zero grad
        optimizer.zero_grad()

        # forward
        output = net(imgs)

        # calculate loss
        loss = criterion(output, labels.float())

        runningLoss += loss.item()
        loss_count += 1

        # calculate gradients.
        if config.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # reduce loss
        optimizer.step()

        # print loss
        # print(index)
        if (index + 1) % 1 == 0:
            print("Epoch: %2d, Batch: %4d / %4d, Loss: %.3f" % (epoch + 1, index + 1, batch_num, loss.item()))

    avg_loss = runningLoss / loss_count
    print("For Epoch: %2d, Average Loss: %.3f" % (epoch + 1, avg_loss))
    log.write("For Epoch: %2d, Average Loss: %.3f" % (epoch + 1, avg_loss) + "\n")

# ------------------------------------test------------------------------------
def test(net, test_loader, criterion, optimizer, epoch, device, log, train_start):
    # test after each epoch
    net.eval()
    print("Start test:")
    with torch.no_grad():
        total_len = 0
        correct_len = 0
        global best_val_acc
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for index, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = net(imgs)
            pred = (output >= 0.5).float()

            # TP predict 和 label 同时为1
            TP += int(((pred == 1) & (labels == 1)).cpu().sum())
            # TN predict 和 label 同时为0
            TN += int(((pred == 0) & (labels == 0)).cpu().sum())
            # FN predict 0 label 1
            FN += int(((pred == 0) & (labels == 1)).cpu().sum())
            # FP predict 1 label 0
            FP += int(((pred == 1) & (labels == 0)).cpu().sum())
        #print(TP,TN,FN,FP)

        if TP + FP != 0:
            p = TP / (TP + FP)
            print("percision:", p)
            log.write("percision: " + str(p) + "\n")
        if TP + FN != 0:
            r = TP / (TP + FN)
            print("recall:", r)
            log.write("recall: " + str(r) + "\n")
        if TP + FP != 0 and TP + FN != 0 and r + p != 0:
            F1 = 2 * r * p / (r + p)

            global best_val_F1
            if F1 > best_val_F1[1]:
                # save model
                best_val_F1 = (epoch + 1, F1)
                torch.save(net.state_dict(), config.model_path)
                print("Model saved in epoch " + str(epoch + 1) + ", acc: " + str(F1) + ".")
                log.write("Model saved in epoch " + str(epoch + 1) + ", acc: " + str(F1) + ".\n")

                i = 0
                for index, (imgs, labels) in enumerate(test_loader):
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = net(imgs)
                    pred = (output >= 0.5).float()
                    for picture in pred:
                        picture = picture.cpu()
                        picture = np.array(picture[0] * 255)
                        cv2.imwrite("output/"+str(i+1)+".jpg", picture)
                        i += 1


            print("F1:", F1)
            log.write("F1: " + str(F1) + "\n")


        acc = (TP + TN) / (TP + TN + FP + FN)

        print("acc:", acc)
        log.write("acc: " + str(acc) + "\n")

        # torch.save(net.state_dict(), "normal_"+config.model_path)

    # print time pass after each epoch
    current_time = time.time()
    pass_time = int(current_time - train_start)
    time_string = str(pass_time // 3600) + " hours, " + str((pass_time % 3600) // 60) + " minutes, " + str(
        pass_time % 60) + " seconds."
    print("Time pass:", time_string)
    print()
    log.write("Time pass: " + time_string + "\n\n")

# ------------------------------------main------------------------------------
# main
def main():
    # if GPU is availale, use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    print("Start load train dataset:")
    train_dataset = blood_vessels(config.train_img_path, config.train_img_names, config.train_label_path, config.train_label_names, get_train_transforms(config.img_size))
    print("length of train dataset is", len(train_dataset))
    log.write("length of train dataset is " + str(len(train_dataset)) + "\n")

    print("Start load test dataset:")
    test_dataset = blood_vessels(config.test_img_path, config.test_img_names, config.test_label_path, config.test_label_names, get_test_transforms(config.img_size))
    print("length of val dataset is", len(test_dataset))
    log.write("length of val dataset is " + str(len(test_dataset)) + "\n\n")

    print()
    print("Start training:")

    # create dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batchSize, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batchSize, shuffle=False)

    # net model
    net = get_model(config.model_name, config.from_old_model, device, config.model_path, config.output_channel)

    params = net.parameters()

        # create optimizer
    if config.optimizer_name == "SGD":
        optimizer = toptim.SGD(params, lr=config.learning_rate)
    elif config.optimizer_name == "Adam":
        optimizer = toptim.Adam(params, lr=config.learning_rate)
    elif config.optimizer_name == "AdamW":
        optimizer = AdamW(params, lr=config.learning_rate, weight_decay=1e-6)

    # 混合精度加速
    if config.use_apex:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    train_start = time.time()

    for epoch in range(config.epochs):
        '''
        # change lr by epoch
        adjust_learning_rate(optimizer, epoch)
        '''

        # start train
        train(net, train_loader, config.criterion, optimizer, epoch, device, log)

        # start test
        test(net, test_loader, config.criterion, optimizer, epoch, device, log, train_start)

    print("Final saved model is epoch "+str(best_val_F1[0])+", acc: "+str(best_val_F1[1])+".")
    log.write("Final saved model is epoch "+str(best_val_F1[0])+", acc: "+str(best_val_F1[1])+"\n")

    print("Done.")
    log.write("Done.\n")


if __name__ == '__main__':
    with open(config.log_name, 'w') as log:
        main()




