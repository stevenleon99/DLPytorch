import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# L2 loss
optimizer = optim.SGD(net.parameters(), lr=leanring_rate, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, 'min')

for epoch in xrange(args.start_epoch, args.epochs):
    train(train_loader, model, criterion, optimizer, epoch)
    result_avg, loss_val = validate(val_loader, model, criterion, epoch)
    scheduler.step(loss_val) # monitor the loss val change or not


# or use StepLR
scheduler = StepLR(optimizer, 30, 0.1) # every 30 epochs, the lr * 0.1