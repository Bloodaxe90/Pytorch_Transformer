import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from src.utils.io import save_results, save_model
import time


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    eval_interval: int,
    save_interval: int,
    epochs: int,
    exp_name: str,
    device: str,
    model_name: str = "",
) -> pd.DataFrame:

    results = pd.DataFrame(columns=["Epoch", "Batch", "Loss", "Accuracy", "Time"])

    model.train()
    print("TRAINING BEGUN")
    start_time = time.time()
    for epoch in range(epochs):

        total_loss = 0
        total_accuracy = 0

        for batch_num, batch in enumerate(train_dataloader, start=1):
            x, y_target = batch

            x = x.to(device)
            y_target = y_target.to(device)

            logits: torch.Tensor = model(x)

            batch_size, seq_length, vocab_size = logits.shape
            loss = loss_fn(
                logits.view(batch_size * seq_length, vocab_size),
                y_target.view(batch_size * seq_length),
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            accuracy = (
                y_target == torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            ).sum().item() / (batch_size * seq_length)

            total_loss += loss.item()
            total_accuracy += accuracy
            if batch_num % eval_interval == 0:
                avg_loss = total_loss / eval_interval
                avg_accuracy = total_accuracy / eval_interval
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch: {epoch} | "
                    f"Batch: {batch_num} | "
                    f"Loss: {avg_loss} | "
                    f"Accuracy: {avg_accuracy} | "
                    f"Time: {elapsed_time:.3f}"
                )

                results.loc[len(results)] = [
                    epoch,
                    batch_num,
                    avg_loss,
                    avg_accuracy,
                    elapsed_time,
                ]

                total_loss = 0
                total_accuracy = 0

            if (batch_num + (epoch * len(train_dataloader))) % save_interval == 0:
                save_results(results, exp_name)
                if model_name != "":
                    save_model(model, model_name)

    save_results(results, exp_name)
    save_model(model, model_name)
    return results
