from src.utils.character_dataset import CharacterDataset
import torch.nn.functional as F
import torch


def generate(
    seed_text: str,
    model,
    dataset,
    generation_length: int,
    block_size: int,
    device: str,
    stochastic: bool = True,
):
    try:
        seed_tokens = torch.LongTensor([dataset.encode[char] for char in seed_text]).to(
            device
        )
    except KeyError:
        raise KeyError(
            f"A token in the input sequence is invalid, all tokens must appear in the trained text"
        )

    output: torch.Tensor = seed_tokens
    for _ in range(generation_length):
        logits = model.generate(seed_tokens)

        token_probs = F.softmax(logits[..., -1, :], dim=-1)
        pred_token = (
            torch.multinomial(token_probs, num_samples=1).squeeze(0)
            if stochastic
            else torch.argmax(token_probs, dim=-1)
        )
        output = torch.cat((output, pred_token), dim=-1)
        seed_tokens = torch.cat((seed_tokens, pred_token), dim=-1)
        seed_tokens = (
            seed_tokens[..., -block_size:]
            if len(seed_tokens) >= block_size
            else seed_tokens
        )

    return "".join([dataset.decode[token.item()] for token in output])
