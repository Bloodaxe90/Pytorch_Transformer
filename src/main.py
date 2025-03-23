import torch.optim
from torch import nn

from src.engine.train import train
from src.models.character_generator import CharacterGenerator
from src.utils.character_dataset import CharacterDataset
from src.utils.create_dataloaders import create_dataloaders
from src.utils.io import get_text


def main():
    TXT_FILE_NAME = "bee_movie"
    text = get_text(TXT_FILE_NAME)

    BATCH_SIZE = 64
    BLOCK_SIZE = 256
    SHUFFLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WORKERS = (
        min(max(1, BATCH_SIZE // 16), torch.cuda.device_count())
        if DEVICE == "cuda"
        else torch.cpu.device_count()
    )

    train_dataloader, test_dataloader = create_dataloaders(
        text=text,
        train_percent=0.9,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        shuffle=SHUFFLE,
    )

    train_dataset: CharacterDataset = train_dataloader.dataset

    VOCAB_SIZE = len(train_dataset.characters)
    EMBEDDING_DIM = 256
    NUM_TRANSFORMERS = 8
    NUM_HEADS = 8
    POSITIONAL_FFNN_HIDDEN_NEURONS = EMBEDDING_DIM * 4
    DROPOUT_PROB = 0.2

    model = CharacterGenerator(
        embedding_dim=EMBEDDING_DIM,
        vocab_size=VOCAB_SIZE,
        num_transformers=NUM_TRANSFORMERS,
        num_heads=NUM_HEADS,
        ffnn_hidden_neurons=POSITIONAL_FFNN_HIDDEN_NEURONS,
        dropout_prob=DROPOUT_PROB,
    )

    if DEVICE == "cuda":
        model = nn.DataParallel(model, device_ids=list(range(WORKERS)))
    model.to(DEVICE)

    print(f"Device: {DEVICE}")
    print(f"Workers: {WORKERS}")

    ALPHA = 3e-4
    EPOCHS = 4
    EXP_NAME = (
        f"{TXT_FILE_NAME}_"
        f"LR{ALPHA}_"
        f"E{EPOCHS}_"
        f"BK{BLOCK_SIZE}_"
        f"V{VOCAB_SIZE}_"
        f"D{EMBEDDING_DIM}_"
        f"B{BATCH_SIZE}_"
        f"T{NUM_TRANSFORMERS}_"
        f"H{NUM_HEADS}_"
        f"PN{POSITIONAL_FFNN_HIDDEN_NEURONS}_"
        f"DP{DROPOUT_PROB}"
    )
    MODEL_NAME = f"" f"{EXP_NAME}"
    EVAL_INTERVAL = 100  # Every x batches
    SAVE_INTERVAL = 1000  # Every x batches

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(params=model.parameters(), lr=ALPHA),
        epochs=EPOCHS,
        eval_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        device=DEVICE,
        exp_name=EXP_NAME,
        model_name=MODEL_NAME,
    )

    print(results)


if __name__ == "__main__":
    main()
