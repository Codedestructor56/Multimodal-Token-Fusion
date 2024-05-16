import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from data_prep import *
from dataclasses import dataclass
from typing import Optional,List
from transformers import BertTokenizer, AutoTokenizer

@dataclass
class Parameters:
    device: str
    num_heads: int
    emb_dim: int
    max_seq_len: int
    tokenizer: str
    max_im_height: int
    max_im_width: int
    batch_size: int
    dataset_name: str
    use_cache: str
    ffn_hidden_dim: int
    thresh: Optional[int]
    num_layers: int
    vocab_size: int
    patch_size: int
    token_thresh: int
    imp_layer_hidden: int
    div_batch: int

class Medical_Data(Dataset):
    def __init__(self, params:Parameters):
        super().__init__()
        self.device = params.device
        self.max_seq_len = params.max_seq_len
        self.emb_dim = params.emb_dim
        self.max_height = params.max_im_height
        self.max_width = params.max_im_width
        self.batch_size = params.batch_size
        self.tokenizer = params.tokenizer
        self.dataset = MedicalDatasetCreator(params.dataset_name)
        self.dataset.load_dataset() 
     
    def __len__(self):
        return self.dataset.dataset_length
    
    def __getitem__(self, idx):
        self.dataset.load_row(idx)
        case_text = self.dataset.case_text
        encoded_case_text = torch.tensor(self.tokenizer.encode(case_text),dtype = torch.int32).to(self.device)
        image = torch.tensor(self.dataset.image, dtype = torch.int8)

        return torch.cat((encoded_case_text[:self.max_seq_len-1], encoded_case_text[-1].unsqueeze(0)), dim=0), image
    
    def pad_im(self, imgs):
        img_batch = []
        for img in imgs:  
            img_gray = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
            height, width = img_gray.shape[:2] 
 
            pad_height = max(0, self.max_height - height)
            pad_width = max(0, self.max_width - width)
            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            img_padded = cv2.copyMakeBorder(img_gray, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
            img_cropped = img_padded[:self.max_height, :self.max_width]
            img_batch.append(img_cropped)

        imgs_tensor = torch.tensor(np.array(img_batch), dtype=torch.uint8).to(self.device)
        del img_batch
        return imgs_tensor
        
    def collate_fn(self, batch):
        batch = list(zip(*batch))
        padded_text = pad_sequence(batch[0], batch_first = True, padding_value = 0) 
        padded_img = self.pad_im(batch[1])

        return padded_text.to(self.device), padded_img


