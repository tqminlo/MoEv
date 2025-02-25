import torch.nn.functional as F
import torch
import numpy as np
import time


class CusTom(torch.nn.Module):

    def __init__(self):
        super(CusTom, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 2400)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(2400, 2400)
        self.fc3 = torch.nn.Linear(2400, 2400)
        self.fc4 = torch.nn.Linear(2400, 2400)
        self.fc5 = torch.nn.Linear(2400, 840)
        self.fc6 = torch.nn.Linear(840, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x

    def quick_infer(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x


if __name__ == "__main":
    model = CusTom()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    torch.save(model.state_dict(), "custom_test.pth")
    model = CusTom()
    model.load_state_dict(torch.load("custom_test.pth", weights_only=True))
    model.eval()

    inp = np.random.random(size=(1, 1, 32, 32)).astype(np.float32)
    inp = torch.from_numpy(inp).to("cpu")
    for i in range(20):
        if i % 2 == 0:
            start = time.time()
            out = model(inp)
            end = time.time()
            print(i, end - start)
        else:
            start = time.time()
            out = model.quick_infer(inp)
            end = time.time()
            print(i, end - start)