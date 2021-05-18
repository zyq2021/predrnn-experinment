from predrnn import RNN
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


channel = 1
stride = 1
layer_norm = True
img_width = 128
filter_size = 5
total_length = 6
input_length = 5
pred_length = 1
device = 'cuda'

class Config():
    channel = 1
    stride = 1
    layer_norm = True
    img_width = 128
    filter_size = 5
    total_length = 6
    input_length = 5
    device = 'cuda'


configs = Config()


def get_data(path_root):

    files = os.listdir(path_root)
    files = sorted(files)
    sample_data = []
    for file in tqdm(files):
        real_path = os.path.join(path_root, file)
        image = Image.open(real_path)
        image = image.convert('L')
        image = image.resize((img_width, img_width))
        data = np.expand_dims(np.array(image) / 255, -1).astype(np.float32)
        sample_data.append(data)
    sample_data = np.stack(sample_data, 0)
    return sample_data




class SplitDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __getitem__(self, item):
        # (b, 64, 64, 1)
        inputs = self.data[item: item+total_length]
        mask_true = torch.zeros((pred_length-1, img_width, img_width, 1), dtype=torch.float32)
        return inputs, mask_true

    def __len__(self):
        return len(self.data) - total_length


class Predrnn(torch.nn.Module):
    def __init__(self):
        super(Predrnn, self).__init__()
        self.net = RNN(1, [64], configs)

    def forward(self, x, mask):
        """
        :param x: (b, 5, img_size, img_size, 1)
        :param mask: (b, 4, img_size, img_size, 1)
        :return: (b, 19, img_size, img_size, 1)
        """
        return self.net(x, mask)

    def configure_optimizers(self,):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_ix):
        x, mask = batch
        pred = self.forward(x, mask)
        loss = F.mse_loss(pred, x[:, 1:])
        self.log('train_loss', loss)
        return loss


path_root = 'data/photo1_8'


# 生成训练数据的数据加载器
data = get_data(path_root)
dataset = SplitDataset(data)
tol_nums = len(dataset)
train_nums = int(tol_nums*0.8)
trainds, testds = random_split(dataset, [train_nums, tol_nums-train_nums])
traindl = DataLoader(trainds, batch_size=2, shuffle=True)
testdl = DataLoader(testds, batch_size=2)



# 实例化模型
model = Predrnn()
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):

    model.train()
    batch_losses = []
    for batch_ix, batch in enumerate(traindl):
        optimizer.zero_grad()
        x, mask = batch
        x = x.cuda()
        mask = mask.cuda()
        pred = model.forward(x, mask)
        loss = F.mse_loss(pred, x[:, 1:])
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        if (batch_ix + 1) % 20 == 0:
            print(f'epoch{epoch} batch ix:{batch_ix} train loss:{np.sum(batch_losses):.4f}')
            batch_losses = []

    with torch.no_grad():
        model.eval()
        test_losses = []
        for batch in testdl:
            optimizer.zero_grad()
            x, mask = batch
            x = x.cuda()
            mask = mask.cuda()
            pred = model.forward(x, mask)
            loss = F.mse_loss(pred, x[:, 1:])
            test_losses.append(loss.item())
        print(f'epoch{epoch} test loss:{np.mean(test_losses):.4f}')











# -------------------------------------------------------------------------------------------
# 显示图像序列
# -------------------------------------------------------------------------------------------
def show_image(dataset, index=1):
    with torch.no_grad():
        sample, mask = dataset[index]
        sample = sample.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        model.to(device)
        pred = model(sample, mask)
        his_seq = sample[0, :10].detach().cpu().numpy().squeeze()
        ground_truth_seq = sample[0, -10:].detach().cpu().numpy().squeeze()
        res = pred[0:, -10:].detach().cpu().numpy().squeeze()


        plt.figure(figsize=(80, 30))
        for i in range(input_length):
            # plt.subplot(3, input_length, i + 1)
            # plt.imshow(his_seq[i], cmap='gray')  # 历史数据
            # plt.axis = ('off')
            # plt.xticks([])
            # plt.yticks([])
            plt.subplot(3, input_length, i + 1 + input_length)
            plt.imshow(ground_truth_seq[i], cmap='gray')  # 真实数据
            plt.axis = ('off')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, input_length, i + 1 + 2 * input_length)
            plt.imshow(res[i], cmap='gray')  # 预测数据
            plt.axis = ('off')
            plt.xticks([])
            plt.yticks([])
        plt.show()
        plt.savefig('result.png')
        return his_seq, ground_truth_seq, res
show_image(testds, 2)



