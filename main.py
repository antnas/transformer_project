import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from modelling import *
from modelling.transformer import Transformer
from modelling.lr_scheduler import TransformerLR
from utils import get_adamw_optimizer
from dataset import load_clean_dataset, translation_collate
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing


if __name__ == '__main__':
    # Set hyperparameters
    lr = 0.001
    weight_decay = 0.0
    num_epochs = 1
    batch_size = 128
    max_length = 50

    src_lan = 'de'
    tgt_lan = 'en'

    model_params = {
        "vocab_size": 30000,
        "d_model": 256,
        "n_heads": 4,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 64,
        "dropout": 0.1,
        "max_len": max_length}

    # Initialize the models
    model = Transformer(**model_params)

    # Load Tokenizers and add a post processor for the BOS and EOS tokens
    de_tokenizer = AutoTokenizer.from_pretrained("models/tokenizer_de")
    de_tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=de_tokenizer.bos_token + " $A " + de_tokenizer.eos_token,
        special_tokens=[(de_tokenizer.eos_token, de_tokenizer.eos_token_id), (de_tokenizer.bos_token, de_tokenizer.bos_token_id)],)

    en_tokenizer = AutoTokenizer.from_pretrained("models/tokenizer_en")
    en_tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=en_tokenizer.bos_token + " $A " + en_tokenizer.eos_token,
        special_tokens=[(en_tokenizer.eos_token, en_tokenizer.eos_token_id), (en_tokenizer.bos_token, en_tokenizer.bos_token_id)],)

    # Load dataset
    train_ds, val_ds, test_ds = load_clean_dataset(de_tokenizer, en_tokenizer, max_length=max_length)
    
    # Create dataloader
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=translation_collate)

    # Initialize optimizer, scheduler and criterion
    optimizer = get_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    lr_scheduler = TransformerLR(optimizer, d_model=256, warmup_steps=1000)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(train_dataloader):
            # Step 1: Load the data for the batch
            src, src_mask = data[src_lan]
            tgt, tgt_mask = data[tgt_lan]
            # Step 2: Perform a forward pass through the model
            outputs = model(src, tgt, src_mask, tgt_mask)

            # Step 3: Calculate the loss
            loss = criterion(outputs, tgt)

            # Step 4: Perform a backward pass through the model
            optimizer.zero_grad()
            loss.backward()

            # Step 5: Update the model parameters
            optimizer.step()

            # Step 6: Update the learning rate
            lr_scheduler.step()

            # Print some information if needed
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_dataloader)} | Loss: {loss.item()}")

        # Print some information at the end of each epoch
        print(f"Epoch {epoch + 1}/{num_epochs}" +  f"Learning rate: {lr_scheduler.get_lr()[0]}")
