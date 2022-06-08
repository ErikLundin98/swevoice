from model_2 import SweVoice
from dataset import CommonVoiceDataset, Collate

import torch
import math
from tqdm.auto import tqdm
import wandb
from dotenv import load_dotenv
import os
from torch.utils.data import DataLoader, random_split
import numpy as np

def main():
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    # print(torch.backends.cudnn.is_available())
    load_dotenv()
    wandb.login(key=os.getenv('WANDB'))
    wandb.init()
    

    load = True

    # Training params
    epoch = 16
    n_mels = 80
    N_EPOCHS = 300
    batch_size = 32
    validation_batch_size = 1
    train_frac, test_frac, valid_frac = 0.9, 0.05, 0.05
    lr = 0.0005

    # Model params
    num_layers = 3
    hidden_size = 256
    bidirectional = True
    num_train_workers = 4
    num_validation_workers = 2
    reduce_factor = 2 # stride 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device', device)

    dataset = CommonVoiceDataset('data', 'validated.csv', n_mels=n_mels, reduce_factor=reduce_factor)
    model = SweVoice(hidden_size=hidden_size, n_chars=len(dataset.chars), num_layers=num_layers, input_channels=n_mels, bidirectional=bidirectional)
    if load:
        model.load_state_dict(torch.load(f'checkpoints/version_2/epoch_{epoch}.pt'))
        model.train()
    
    model.to(device)
   
    n_train = math.floor(train_frac*len(dataset))
    n_test = math.floor(test_frac*len(dataset))
    n_valid = len(dataset) - n_train - n_test

    train, test, valid = random_split(dataset, [n_train, n_test, n_valid])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=num_train_workers, collate_fn=Collate(), pin_memory=True)
    val_loader = DataLoader(valid, batch_size=validation_batch_size, shuffle=False, num_workers=num_validation_workers, collate_fn=Collate(), pin_memory=True)


    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    X_demo, Y_demo, _, _ = next(iter(val_loader))
    X_demo = X_demo.to(device)
    Y_demo = Y_demo.to(device)
    
    label_text = dataset.translate_from_list_of_int(Y_demo[0].detach().cpu().numpy())
    print("label:", label_text)

    best_val_loss = np.inf

    for epoch_idx in range(N_EPOCHS):
        epoch += 1
        train_loss = 0
        model.train(True)
        progress = tqdm(train_loader)

        for i, (X, Y, X_lens, Y_lens) in enumerate(progress):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()

            h, c = model.init_hidden(X.size(0), device)
            preds, (_, _) = model(X, (h, c))
            preds = torch.log_softmax(preds, dim=2).permute(1, 0, 2)
            loss = loss_fn(preds, Y, X_lens, Y_lens)
            
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.detach().item()
            progress.set_description("Batch loss: " + str(np.round(loss.detach().cpu().numpy(), 5)))

        avg_train_loss = train_loss/len(train_loader)
        val_loss = 0
        model.train(False)
        with torch.no_grad():
            for i, (X, Y, X_lens, Y_lens) in enumerate(tqdm(val_loader)):
                h, c = model.init_hidden(X.size(0), device)
                X = X.to(device)
                Y = Y.to(device)
                preds, _ = model(X, (h, c))
                preds = torch.log_softmax(preds, dim=2).permute(1, 0, 2)
                val_loss += loss_fn(preds, Y, X_lens, Y_lens).item()
            
            avg_val_loss = val_loss/len(val_loader)
            print("Validation Loss", avg_val_loss)
            
       
            h, c = model.init_hidden(X_demo.size(0), device)
            preds, (_, _) = model(X_demo, (h, c))
            sample = dataset.translate_from_logits(preds.detach().cpu())[0]
            wandb.log({
                "Train Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Sample" : sample
            })
            print("Sample:", sample.replace('*', ''))
        if val_loss < best_val_loss:
            print("new best model created")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "checkpoints", "version_2", f"epoch_{epoch}.pt"))

if __name__=='__main__':
    main()