import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
params = Parameters(
    device=device,
    use_cache=False,
    num_heads=16,
    thresh=None,
    emb_dim=256,
    max_seq_len=256,
    ffn_hidden_dim=512,
    batch_size=8,
    div_batch=8,
    tokenizer=tokenizer,
    vocab_size=tokenizer.vocab_size + 1,
    max_im_width=480,
    max_im_height=480,
    num_layers=1,
    patch_size=16,
    dataset_name="ct_scan_data",
    token_thresh=0.2,
    imp_layer_hidden=512
)


medical_data = Medical_Data(params)
train_data, val_data = train_test_split(medical_data, test_size=0.1, random_state=42)


train_loader = DataLoader(train_data, batch_size=params.batch_size, collate_fn=train_data.collate_fn)
val_loader = DataLoader(val_data, batch_size=params.batch_size, collate_fn=val_data.collate_fn)

model = TokenFusion(params).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)

num_epochs = 10
best_val_loss = float("inf")

for epoch in range(num_epochs):
  
    model.train()
    train_loss = 0.0
    train_steps = 0
    
    for text_batch, image_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10: Training"):
        text_batch, image_batch = text_batch.to(device), image_batch.to(device)
        
        optimizer.zero_grad()
       
        outputs = model(text_batch, image_batch, cur_pos=None, im_inc=True)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_steps += 1
    

    model.eval()
    val_loss = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for text_batch, image_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/10: Validation"):
            text_batch, image_batch = text_batch.to(device), image_batch.to(device)
  
            outputs = model(text_batch, image_batch, cur_pos=None, im_inc=True)        
            loss = criterion(outputs, target)
            
            val_loss += loss.item()
            val_steps += 1

    train_loss /= train_steps
    val_loss /= val_steps
    
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "tokenfusion_model.pth")
        best_val_loss = val_loss
    
    print(f"Epoch {epoch+1}/10, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("Training completed!")
