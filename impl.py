from utils import *
from data import *
from data_prep import *

class Encoder(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()
        self.device = params.device
        self.emb_dim = params.emb_dim
        self.thresh = params.thresh
        self.norm = RMSnorm(self.emb_dim, self.device, self.thresh)
        self.attention = Attention(params)
        self.ffn = SwiGLu_Forward(params)

    def forward(self, x:torch.Tensor, cur_pos: Optional[int])->torch.Tensor:
        first_layer = x + self.attention(self.norm(x), cur_pos) 
        second_layer = first_layer + self.ffn(self.norm(first_layer))

        del first_layer
        torch.cuda.empty_cache()
        return second_layer

class Transformer(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()
        self.layers_enc_text = nn.ModuleList()
        self.layers_enc_im = nn.ModuleList()
        for _ in range(params.num_layers):
            self.layers_enc_text.append(Encoder(params))
            self.layers_enc_im.append(Encoder(params))
        self.device = params.device
        self.emb_dim = params.emb_dim
        self.seq_len = params.max_seq_len
        self.vocab_size = params.vocab_size
        self.text_embeddings = nn.Embedding(self.vocab_size, self.emb_dim).to(self.device)
        self.thresh = params.thresh
        self.norm = RMSnorm(self.emb_dim, self.device, self.thresh) 
        self.div_batch = params.div_batch
        self.patch_embeddings = PatchEmbeddings(params)
        self.linear = nn.Linear(self.emb_dim, self.vocab_size).to(self.device)

        self.max_height = params.max_im_height
        self.max_width = params.max_im_width
        self.patch_size = params.patch_size
    
    def forward(self, x: torch.Tensor, cur_pos: Optional[int], im_inc: bool)->torch.Tensor:
        assert self.div_batch<=x.shape[0], "Batch serializer should not exceed tensor dimensions"
        if im_inc:
            im_seq_len = (self.max_height//self.patch_size)*(self.max_width//self.patch_size)
            res = self.patch_embeddings(x) 
            del im_seq_len
            torch.cuda.empty_cache()
            
            if cur_pos is None:
                res = self.norm(res)
                for layer in self.layers_enc_im:
                    res = layer(res, cur_pos)

                res = torch.chunk(res, self.div_batch, dim = 0)
        
                accumulated_output = None
                for chunk_idx in range(len(res)):
                    out = self.linear(res[chunk_idx])
                    if accumulated_output is None:
                        accumulated_output = out
                    else:
                        # Concatenate the current output with the accumulated output along the specified dimension
                        accumulated_output = torch.cat((accumulated_output, out), dim=0)
            
                    del out
                    torch.cuda.empty_cache()

                del accumulated_output
                torch.cuda.empty_cache()
            else:
                assert res.shape[1]==1, "Please pass one token at a time"
                res = self.norm(res)
                for layer in self.layers_enc_im:
                    res = layer(res, cur_pos)
                res = self.linear(res) 
        else:
            res = self.text_embeddings(x) 
            if cur_pos is None:
                res = self.norm(res)
                for layer in self.layers_enc_text:
                    res = layer(res, cur_pos)

                res = torch.chunk(res, self.div_batch, dim = 0)
        
                accumulated_output = None
                for chunk_idx in range(len(res)):
                    out = self.linear(res[chunk_idx])
                    if accumulated_output is None:
                        accumulated_output = out
                    else:
                        accumulated_output = torch.cat((accumulated_output, out), dim=0)
            
                    del out
                    torch.cuda.empty_cache()

                    del accumulated_output

            else:
                assert res.shape[1]==1, "Pass one token at a time"
                res = self.norm(res)
                for layer in self.layers_enc_text:
                    res = layer(res, cur_pos)
                res = self.linear(res)

        return res

class TokenFusion(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()
        self.device = params.device
        self.emb_dim = params.emb_dim
        self.token_thresh = params.token_thresh
        self.hidden_dim = params.imp_layer_hidden
        self.thresh = params.thresh
        self.norm = RMSnorm(self.emb_dim, self.device, self.thresh)
        self.vocab_size = params.vocab_size
        self.imp_layer1 = nn.Linear(self.vocab_size, self.hidden_dim).to(self.device)
        self.imp_layer2 = nn.Linear(self.hidden_dim, 1).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.transformer = Transformer(params)
      

    def forward(self, x:Optional[torch.Tensor], y:Optional[torch.Tensor], cur_pos: Optional[int], im_inc: bool):
        if cur_pos is None:
            x, y = self.transformer(x, cur_pos, False), self.transformer(y, cur_pos, im_inc)
            seq_len1, seq_len2 = x.shape[1], y.shape[1]
            min_seq_len = min(seq_len1, seq_len2)
            x_fuse, y_fuse, x_rem, y_rem = x[:,:min_seq_len,:], y[:,:min_seq_len,:], x[:,min_seq_len:,:], y[:,min_seq_len:,:]
            token_scores_x = self.sigmoid(self.imp_layer2(self.imp_layer1(x_fuse)))
            token_scores_y = self.sigmoid(self.imp_layer2(self.imp_layer1(y_fuse)))
            mask_x = (token_scores_x > self.token_thresh).int()
            inv_mask_x = 1-mask_x
            mask_y = (token_scores_y > self.token_thresh).int()
            inv_mask_y = 1-mask_y
            x_fin = x_fuse * mask_x + y_fuse * inv_mask_x.expand(x_fuse.shape[0],min_seq_len,x_fuse.shape[2])
            y_fin = y_fuse * mask_y + x_fuse * inv_mask_y.expand(y_fuse.shape[0],min_seq_len,y_fuse.shape[2])

            x_fin = torch.cat((x_fin, x_rem), dim=1).squeeze()
            y_fin = torch.cat((y_fin, y_rem), dim=1).squeeze()
            

            del x_fuse, y_fuse, x_rem, y_rem, token_scores_x, token_scores_y, mask_x, mask_y, inv_mask_x, inv_mask_y
            torch.cuda.empty_cache()
            return x_fin, y_fin
        else:
            if im_inc:
                return self.transformer(y, cur_pos, im_inc)
            else:
                return self.transformer(x, cur_pos, im_inc)




