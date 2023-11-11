import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import lenet5
from torch import nn
from torch import optim

batch_size = 32
epochs = 30
learning_rate = 0.001

def main():

    cifar_train = datasets.CIFAR10('cifar', train=True, download=True, 
                             transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]))
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', train=False, download=True, 
                             transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]))
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    
    x, label = next(iter(cifar_train))
    print(x.shape, label.shape)

    model = lenet5.Letnet5()
    print(model)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            logits = model(x)
            loss = criteon(logits, label)

            #backpropgate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: {} / {} Loss: {}".format(epoch+1, epochs, loss.item()))

        model.eval()
        with torch.no_grad(): 
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                logits = model(x)
                pre = logits.argmax(dim=1)
                total_correct += torch.eq(pre, label).float().sum().item()
                total_num += x.size(0)
            
            acc = total_correct / total_num
            print(epoch, acc)



if __name__ == "__main__":
    main()
