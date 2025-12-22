import os

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.transformer_dataset.my_dataset import (
    dataset,
    en_model,
    zh_model,
)
from models.transformer.config import settings
from models.transformer.main import Transformer


def show_loss_curve(losses: list[float], save_dir: str):
    """
    Plot and save the loss curve.

    Args:
        losses: List of loss values for each epoch
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # 保存图表
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    print("Loss curve has been saved to loss_curve.png")


def train_model(
    model: Transformer,
    train_loader: DataLoader,
    optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epochs=10,
):
    """
    Train the model for a specified number of epochs.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for the training data.
        optimizer: Optimizer for the model.
        criterion: Loss function.
        num_epochs: Number of epochs to train the model.
        device: Device to run the training on (e.g., "cuda" or "cpu").
    """
    model = model.to(device)
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 用于记录每个epoch的损失值
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            src: Tensor = batch["src"].to(device)
            tgt: Tensor = batch["tgt"].to(device)
            optimizer.zero_grad()
            # (batch_size, seq_len-1, vocab_size)
            output: Tensor = model(src, tgt[:, :-1])
            loss = criterion(
                # (batch_size*(seq_len-1), vocab_size)
                output.reshape(-1, output.shape[2]),
                # 去掉了第一个 token（通常是起始符号 BOS），形状为 (batch_size, seq_len-1)。
                tgt[:, 1:].reshape(-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)  # 记录每个epoch的平均loss
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

    # 训练结束后绘制损失曲线
    show_loss_curve(epoch_losses, save_dir)


def val_model(
    model: Transformer,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
):
    """
    Validate the model on the validation set.

    Args:
        model: The model to be validated.
        val_loader: DataLoader for the validation data.
        criterion: Loss function.
        device: Device to run the validation on (e.g., "cuda" or "cpu").
    """
    model = model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(val_loader):
            src, tgt = src.to(device), tgt.to(device)
            output: Tensor = model(src, tgt[:, :-1])
            loss = criterion(
                output.reshape(-1, output.shape[2]),
                tgt[:, 1:].reshape(-1),
            )
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                # 打印前3个样本的翻译结果
                pred_tokens = output.argmax(dim=-1)  # (batch_size, seq_len-1)
                for i in range(min(3, src.size(0))):
                    pred_sentence = en_model.decode(pred_tokens[i].tolist())
                    tgt_sentence = en_model.decode(tgt[i, 1:].tolist())
                    print(f"[Batch {batch_idx}] Pred: {pred_sentence}")
                    print(f"[Batch {batch_idx}] True: {tgt_sentence}")
                print(f"Validation Batch {batch_idx}, Loss: {loss.item()}")
    print(f"Validation Loss: {total_loss / len(val_loader)}")


if __name__ == "__main__":
    # Initialize model, optimizer, and loss function
    model = Transformer(
        src_pad_idx=zh_model.pad_id,
        tgt_pad_idx=en_model.pad_id,
        tgt_sos_idx=en_model.bos_id,
        src_vocab_size=zh_model.vocab_size,
        tgt_vocab_size=en_model.vocab_size,
        d_model=settings.d_model,
        n_head=settings.n_heads,
        n_layer=settings.n_layers,
        d_ff=settings.ffn_hidden,
        max_seq_len=settings.max_seq_len,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=en_model.pad_id)

    # Get training data loader
    train_loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
    )

    # Train the model
    train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        device=settings.device,
        epochs=10,
    )
