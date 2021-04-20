import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from dataset.deschaintre import SvbrdfDataset
from models import MultiViewModel
from losses import MixedLoss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", "-d", required=False, type=str, default="./data/train", help="")
parser.add_argument("--log_dir", "-ld", required=False, type=str, default="./runs/", help="")
parser.add_argument("--batch_size", "-b", required=False, type=int, default=1, help="")
parser.add_argument("--epochs", "-e", required=False, type=int, default=1, help="")
parser.add_argument("--learning_rate", "-lr", required=False, type=float, default=1e-5, help="")
parser.add_argument("--mix_materials", "--mix", required=False, type=bool, default=False, help="")

args = parser.parse_args()

if __name__ == "__main__":
    N_VIEWS = 3
    IS_COLLOCATED = True
    UPDATE_METHOD = 0
    OFFSET_DIST = torch.Tensor([0.1, 0.1, 0])
    epoch_start = 0

    data_dir = args.data_dir

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size

    writer = SummaryWriter(log_dir=args.log_dir)

    accelerator = Accelerator()
    device = accelerator.device

    dt = SvbrdfDataset(data_dir, 256, "crop", 0, 10, True, mix_materials=args.mix_materials, random_crop=True)
    trainset = torch.utils.data.DataLoader(dt, batch_size=batch_size, pin_memory=True, shuffle=True)

    # Check training set 
    # for batch_num, batch in enumerate(trainset):
    #     print(batch["inputs"].shape)
    #     # print(batch["svbrdf"].shape)
    #     svbrdf_maps = torch.cat(batch["svbrdf"].split(3, dim=-3), dim=0)
    #     print(svbrdf_maps.shape)
    #     plot_imgs(str(batch_num), batch["inputs"], batch["inputs"].shape[0], batch["inputs"].shape[1], permute=True)
    #     break
        # plot_imgs(str(batch_num), svbrdf_maps.unsqueeze(0), 1, 4, permute=True)


    model = MultiViewModel().to(device)
    criterion = MixedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, optim, data = accelerator.prepare(model, optimizer, trainset)

    model.train()
    for epoch in range(epoch_start, epoch_start + epochs):
        for i, batch in enumerate(trainset):
            # Unique index of this batch
            batch_index = epoch * 1 + i
            # batch_index = epoch * batch_count + i
            batch_inputs = batch["inputs"].to(device)
            batch_svbrdf = batch["svbrdf"].to(device)

            optimizer.zero_grad() 
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_svbrdf)
            writer.add_scalar("Loss/train", loss.item(), epoch)
            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()

            print("Epoch {:d}, Batch {:d}, loss: {:f}".format(epoch, i + 1, loss.item()))

            # Statistics
            # writer.add_scalar("loss", loss.item(), batch_index)
            # last_batch_inputs = batch_inputs

        # if epoch % args.save_frequency == 0:
        #     Checkpoint.save(checkpoint_dir, args, model, optimizer, epoch)

        # if epoch % args.validation_frequency == 0 and len(validation_data) > 0:
        #     model.eval()
            
        #     val_loss = 0.0
        #     batch_count_val = 0
        #     for batch in validation_dataloader:
        #         # Construct inputs
        #         batch_inputs = batch["inputs"].to(device)
        #         batch_svbrdf = batch["svbrdf"].to(device)

        #         outputs  = model(batch_inputs)
        #         val_loss += loss_function(outputs, batch_svbrdf).item()
        #         batch_count_val += 1
        #     val_loss /= batch_count_val

        #     print("Epoch {:d}, validation loss: {:f}".format(epoch, val_loss))
        #     writer.add_scalar("val_loss", val_loss, epoch * batch_count)
        
        #     model.train()
    writer.flush()
    writer.close()