import torch.nn as nn
from data import *
from data_prep import *
import math
from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

class RotaryEmbeddings(nn.Module):
    def __init__(self, device:str, theta: int =10000):
        super().__init__()
        self.theta = theta
        self.device = device

    def forward(self, x: torch.Tensor, seq_len:Optional[int]=None, emb_dim:Optional[int]=None)->torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        assert emb_dim%2==0, "Embeddings dimension must be even"
        #Q_i=10000^(-2(i-1)/emb_dim)
        thetas = (1.0/self.theta**((2*torch.arange(0,emb_dim,2))//emb_dim)).to(self.device)
        thetas_repeated = thetas.unsqueeze(0).repeat(seq_len, 1)
        thetas_true = thetas_repeated * (torch.arange(seq_len, device = self.device)+1).unsqueeze(1)
        #calculate the rotation matrices using these thetas, apply them on the embeddings in  2D or complex space
        matrix_rot = torch.stack((torch.sin(thetas_true),torch.cos(thetas_true)),dim=-1).to(self.device)
        comp_matrix = torch.view_as_complex(matrix_rot).unsqueeze(0)
        x_reshaped = torch.view_as_complex(x.reshape(batch_size, seq_len, emb_dim//2, 2))
        rotated_x = torch.view_as_real(x_reshaped * comp_matrix).squeeze(-1).reshape(batch_size, seq_len, emb_dim).to(self.device)
        del x_reshaped, comp_matrix, matrix_rot, thetas_true, thetas_repeated, thetas
        torch.cuda.empty_cache()
        return rotated_x


class Attention(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()
        self.use_cache = params.use_cache
        self.device = params.device
        self.pos_rotor = RotaryEmbeddings(self.device)

        self.num_heads = params.num_heads
        assert params.emb_dim % self.num_heads==0, "Make the embedding dim divisible by num_heads"
        self.head_dim = params.emb_dim//self.num_heads
        self.wq = nn.Linear(params.emb_dim, self.num_heads*self.head_dim).to(self.device)
        self.wk = nn.Linear(params.emb_dim, self.num_heads*self.head_dim).to(self.device)
        self.wv = nn.Linear(params.emb_dim, self.num_heads*self.head_dim).to(self.device)
        self.wo = nn.Linear(params.emb_dim, self.num_heads*self.head_dim).to(self.device)
        if self.use_cache:
            self.c_v = torch.zeros((params.max_batch_size, params.max_seq_len, self.num_heads, self.head_dim))
            self.c_k = torch.zeros((params.max_batch_size, params.max_seq_len, self.num_heads, self.head_dim))

    def forward(self, x:torch.Tensor, cur_pos: Optional[int]=None)->torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        output = self.wo(x)
        
        xq = self.pos_rotor(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = self.pos_rotor(value).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if self.use_cache:
            self.c_v[:batch_size, cur_pos:cur_pos+seq_len]=xv
            self.c_k[:batch_size, cur_pos:cur_pos+seq_len]=xk
            
            keys = self.c_k[:batch_size, :cur_pos+seq_len]
            values = self.c_v[:batch_size, :cur_pos+seq_len]

            keys = keys[:,:,:,None,:].expand(keys.shape[0], keys.shape[1],
                                           self.num_heads, 1, self.head_dim).reshape(keys.shape[0],
                                            keys.shape[1], self.num_heads, self.head_dim)

            values = values[:,:,:,None,:].expand(values.shape[0], values.shape[1],
                                                 self.num_heads, 1, self.head_dim).reshape(values.shape[0],
                                                 values.shape[1], self.num_heads, self.head_dim)

        else:
            keys = xq
            values = xv
        
        xq = xq.permute(0, 2, 1, 3).contiguous().to(self.device)
        keys = keys.permute(0, 2, 3, 1).contiguous().to(self.device)
        values = values.permute(0, 2, 1, 3).contiguous().to(self.device)
        
        query_key_score = torch.matmul(xq, keys)/math.sqrt(self.head_dim)
        attention_score = torch.matmul(query_key_score, values).transpose(1,2).contiguous().reshape(batch_size, seq_len, -1)
        output = self.wo(attention_score)

        del query_key_score, attention_score, xq, keys, values 
        torch.cuda.empty_cache()
        #make sure that the dimensions are correct and that the training and inferencing parts are compatible
        return output

class RMSnorm(nn.Module):
    def __init__(self, dim:int, device:str, thresh: float = 1e-4):
        super().__init__()
        self.params = nn.Parameter(torch.ones(dim))
        self.thresh = thresh
        self.device = device

    def forward(self, x:torch.Tensor)->torch.Tensor:
        denom = torch.sqrt(x.pow(2).mean(-1,keepdims=True)).to(self.device)
        res = ((x.to(self.device))*self.params.to(self.device))/denom
        del denom
        torch.cuda.empty_cache()
        return res

class SwiGLu_Forward(nn.Module):
    def __init__(self, params:Parameters):
        super().__init__()
        self.hidden_dim = params.ffn_hidden_dim
        self.device = params.device
        self.w1 = nn.Linear(params.emb_dim, self.hidden_dim).to(self.device)
        self.w2 = nn.Linear(params.emb_dim, self.hidden_dim).to(self.device)
        self.w3 = nn.Linear(self.hidden_dim, params.emb_dim).to(self.device)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.w3(self.w2(x)*nn.functional.silu(self.w1(x)))




class PatchEmbeddings(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()
        self.device = params.device
        self.emb_dim = params.emb_dim
        self.patch_size = params.patch_size
        self.max_height = params.max_im_height
        self.max_width = params.max_im_width
        assert self.max_height == self.max_width, "Width and height should be equal"
        assert self.max_height % self.patch_size == 0, "Patch size and image dims should be compatible"
        self.linear = nn.Linear(self.patch_size**2, self.emb_dim).to(self.device)
    
    def patchify(self, image: torch.Tensor):
        patches = []
        batch_size, height, width = image.size()
        for b in range(batch_size):
            for h in range(0, height, self.patch_size):
                for w in range(0, width, self.patch_size):
                    patch = image[b,h:h+self.patch_size, w:w+self.patch_size].float()
                    patches.append(patch)

        return torch.stack(patches, dim = 0).reshape(batch_size, -1, self.patch_size, self.patch_size)

    def forward(self, x:torch.Tensor)->torch.tensor:
        print(f"Size in patch: {x.size()}")
        batch_size, patch_height, patch_width = x.size()
        #print(patch_height, patch_width)
        assert patch_width == patch_height, "Uniform patch size should be provided"
        patches = self.linear(x.view(batch_size, -1, patch_width * patch_height)) 
        positions = torch.arange(patches.shape[1], dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(patches.shape[1], self.emb_dim)
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2).float() * (-math.log(10000.0) / self.emb_dim))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term) 
        
        pe = pe.repeat(batch_size, 1, 1).to(self.device)
        patches += pe
        del pe, positions, div_term
        torch.cuda.empty_cache()
        return patches


device = "cpu"
#tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

params = Parameters(device = device, use_cache = False, num_heads = 16, thresh = None, emb_dim = 256, max_seq_len = 256
                    ,ffn_hidden_dim = 512, batch_size = 8, div_batch = 8, 
                    tokenizer = tokenizer, vocab_size = tokenizer.vocab_size+1,
                    max_im_width = 480, max_im_height = 480, num_layers = 1, patch_size = 16, dataset_name = "ct_scan_data",
                    token_thresh = 0.2, imp_layer_hidden = 512)

dataloader = DataLoader(
    Medical_Data(params), 
    batch_size=params.batch_size, 
    collate_fn=Medical_Data(params).collate_fn
)

first_load = next(iter(dataloader))
pt = PatchEmbeddings(params)



