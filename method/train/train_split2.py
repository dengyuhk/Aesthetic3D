import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from netIMG import ShapeImageDataset, ModelImg
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default="Logs")
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--save_rate', type=int, default=5)
    parser.add_argument('--architecture', type=str, default="ModelImg")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pairs_path', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--margin', type=float, default=0.1)
    return parser.parse_args()


def weighted_loss(s1, s2, margin_w, margin):
    """Pairwise weighted margin loss"""
    temp = margin_w * margin - (s1 - s2)
    loss = torch.clamp(temp, min=0)
    return torch.mean(loss)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    dataset = ShapeImageDataset(
        data_path=args.data_path,
        pairs_path=args.pairs_path
    )
    loader = DataLoader(dataset, batch_size=args.batchSize,
                        shuffle=True, num_workers=8, drop_last=True)

    print("Using GPUs:", torch.cuda.device_count())

    # Model
    if args.architecture == "ModelImg":
        net = ModelImg().to(device)
    else:
        raise ValueError("Unknown architecture")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')

    # TensorBoard writer
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    print("Start training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        net.train()
        train_loss_bt = []
        train_acc_bt = []

        for i, (shape1, shape2, margin_w) in enumerate(loader):
            optimizer.zero_grad()
            shape1, shape2, margin_w = shape1.to(device), shape2.to(device), margin_w.to(device)

            with torch.amp.autocast('cuda'):
                s1 = net(shape1)
                s2 = net(shape2)
                # target = +1 if shape1 better than shape2
                target = torch.sign(margin_w)
                target[target == 0] = 1
                loss = weighted_loss(s1, s2, margin_w, args.margin)
                acc = torch.mean((s1 - s2).gt(0).float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_bt.append(loss.item())
            train_acc_bt.append(acc.item())

            if (i + 1) % 20 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Iter {i+1}/{len(loader)}, "
                      f"Loss: {np.mean(train_loss_bt):.4f}, Acc: {np.mean(train_acc_bt):.3f}")

        # Logging to TensorBoard
        train_loss = np.mean(train_loss_bt)
        train_acc = np.mean(train_acc_bt)
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', train_acc, epoch + 1)

        # Save checkpoint
        if (epoch + 1) % args.save_rate == 0:
            save_path = os.path.join(args.log_path, f"model_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc
            }, save_path)
            print("Saved checkpoint:", save_path)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished. Time: {epoch_time:.2f}s, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.3f}")
        start_time = time.time()

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
