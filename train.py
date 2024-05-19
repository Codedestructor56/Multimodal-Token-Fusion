import torch
from utils import *
from data import *
from impl import *
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from torch.optim import AdamW

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, params):
    setup(rank, world_size)
    torch.manual_seed(42)

    dataset = Medical_Data(params)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, sampler=sampler, collate_fn=dataset.collate_fn)
    model = TokenFusion(params).to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(3): 
        sampler.set_epoch(epoch)
        for batch in dataloader:
            texts, images = batch
            pt = PatchEmbeddings(params)
            patched_images = pt.patchify(images)
            optimizer.zero_grad()
            outputs = model(texts, patched_images, cur_pos=None, im_inc=True)
            probs = [F.softmax(outputs[0], dim=-1), F.softmax(outputs[1], dim=-1)]
            embeds = nn.Embedding(texts.shape[-1], params.emb_dim).to(params.device)
            texts = embeds(texts)
    
            if probs[0].shape[1] < texts.shape[1]:
                probs[0] = torch.nn.functional.pad(probs[0], (0, 0, 0, texts.shape[1]-probs[0].shape[1])) 

            if probs[1].shape[1] < texts.shape[1]:
                probs[1] = torch.nn.functional.pad(probs[1], (0, 0, 0, texts.shape[1]-probs[1].shape[1])) 

            print(texts.shape, probs[0].shape, probs[1].shape)
            loss_text1 = F.cross_entropy(probs[0], texts)
            loss_text2 = F.cross_entropy(probs[1], texts)
            loss = (loss_text1 + loss_text2) / 2.0 
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    cleanup()

def get_num_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0
    return num_gpus

def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    print(tokenizer.vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = Parameters(device = device, use_cache = False, num_heads = 16, thresh = None, emb_dim = 256, max_seq_len = 256
                    ,ffn_hidden_dim = 512, batch_size = 8, div_batch = 8, 
                    tokenizer = tokenizer, vocab_size = tokenizer.vocab_size+1,
                    max_im_width = 240, max_im_height = 240, num_layers = 1, patch_size = 16, dataset_name = "ct_scan_data",
                    token_thresh = 0.3, imp_layer_hidden = 512)

    world_size = get_num_gpus() 
    mp.spawn(train, args=(world_size, params), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
