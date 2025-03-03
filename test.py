from train import validation_loader
from model import CustomModel
import torch
import time

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot')

net = CustomModel()
net.load_state_dict(torch.load("saved/model_20250302_030146", weights_only=True))
net.eval()

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs


start = time.time()
with torch.no_grad():
    for data in validation_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end = time.time()
print("Num pass : ", correct)
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
print("Time : ", end - start)


correct = 0
total = 0
start = time.time()
with torch.no_grad():
    for data in validation_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net.quick_infer(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end = time.time()
print("Num pass : ", correct)
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
print("Time : ", end - start)





