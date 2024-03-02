import torch

train_db, val_db = torch.utilis.data.random_split(train_db, [50000, 10000])
train_loader = torch.utilis.data.Dataloader(train_db,
                             batch_size = batch_size,
                             shuffle = True)
val_loader = torch.utilis.data.Dataloader(val_db,
                             batch_size = batch_size,
                             shuffle = True)

