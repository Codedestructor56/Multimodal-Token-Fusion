from impl import *
from utils import *
from data import *
from data_prep import *
from tqdm import tqdm


def _sample_top_p(probs, p):
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative_probs <= p
    top_p_mask[..., 0] = True
    filtered_probs = sorted_probs * top_p_mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(filtered_probs, num_samples=1)
    next_token = torch.gather(sorted_indices, -1, next_token)
    return next_token


def pad_im(img, max_height, max_width, device): 
    img_gray = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape[:2]
    pad_height = max(0, max_height - height)
    pad_width = max(0, max_width - width)
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    img_padded = cv2.copyMakeBorder(img_gray, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    img_cropped = img_padded[:max_height, :max_width]

    img_tensor = torch.tensor(np.array(img_cropped), dtype=torch.uint8).to(device)
    return img_tensor


def infer(tokenizer, prompts: list[str], params: Parameters, model: TokenFusion = None, temp: float = 0.3, top_p: float = 0.8):
    max_len = params.max_seq_len
    batch_size = params.batch_size
    device = params.device
    image_paths = []
    text_descriptions = []
    for prompt in prompts:
        if "image: " in prompt:
            image_part, text_part = prompt.split(" text: ")
            image_path = image_part.replace("image: ", "")
        else:
            image_path = None
            text_part = prompt.replace("text: ", "")
        
        image_paths.append(image_path)
        text_descriptions.append(text_part)
 
    prompts = [tokenizer.encode(prompt) for prompt in text_descriptions]
    assert len(prompts)<=batch_size, f"Too many prompts, they should be less than or equal to{batch_size}"
    max_prompt_len = max(len(prompt) for prompt in prompts)
    assert max_prompt_len<=max_len, f"Keep your prompt size below {max_len}"

    total_len = min(params.max_seq_len, max_len + max_prompt_len)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0 
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompts): 
        tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    
    only_text_indices = []
    for i in range(len(image_paths)):
        if image_paths[i] is not None:
            only_text_indices.append(i)

    only_text_indices = torch.tensor(only_text_indices).to(device)

    images = []
    for path in image_paths:
        if path is None:
            images.append(torch.zeros(params.max_im_height, params.max_im_width))
        else: 
            images.append(pad_im(cv2.imread(path), params.max_im_height, params.max_im_width, device))
    
    images = torch.stack(images, dim=0).to(device)
    images = torch.concat((torch.zeros(batch_size-images.shape[0],images.shape[1],images.shape[2]),images),dim=0)
    
    pt = PatchEmbeddings(params)
    patched_images = pt.patchify(images)
    eos_reached = torch.tensor([False] * batch_size, device=device)
    prompt_tokens_mask = tokens != pad_id  
    token_storage_multimodal = torch.zeros(patched_images.size(0), patched_images.size(1)).to(device)

    for cur_pos in tqdm(range(1, total_len), desc='Generating tokens'):
        with torch.no_grad():
            logits = model.forward(tokens[:, cur_pos-1:cur_pos], None, cur_pos, False)
        if temp > 0:
            probs = torch.softmax(logits[:, -1] / temp, dim=-1)
            next_token = _sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == tokenizer.eos_token_id)
        if all(eos_reached):
            break

    for cur_pos in tqdm(range(1, patched_images.size(1)), desc='Generating tokens'):
        with torch.no_grad():
            logits = model.forward(None, patched_images[:, cur_pos-1:cur_pos].squeeze(), cur_pos, True)
        if temp > 0:
            probs = torch.softmax(logits[:, -1] / temp, dim=-1)
            next_token = _sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        
        next_token = next_token.reshape(-1)
        token_storage_multimodal[:, cur_pos] = next_token
        eos_reached |=  (next_token == tokenizer.eos_token_id)
        if all(eos_reached):
            break
     
    out_tokens = []
    out_text = []
    for prompt_index, current_prompt_tokens in enumerate(token_storage_multimodal.tolist()):
        if tokenizer.eos_token_id in current_prompt_tokens:
            eos_idx = current_prompt_tokens.index(tokenizer.eos_token_id)
            current_prompt_tokens = current_prompt_tokens[:eos_idx]
        out_tokens.append(current_prompt_tokens)
        out_text.append(tokenizer.decode(torch.tensor(current_prompt_tokens, dtype=torch.int32)))

    for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
        if tokenizer.eos_token_id in current_prompt_tokens:
            eos_idx = current_prompt_tokens.index(tokenizer.eos_token_id)
            current_prompt_tokens = current_prompt_tokens[:eos_idx]
        out_tokens.append(current_prompt_tokens)
        out_text.append(tokenizer.decode(current_prompt_tokens))


    return (out_tokens, out_text)

device = "cpu"
#tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

#hf_checkpoint = "Intel/llava-llama-4-8b"
#tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
print(tokenizer.vocab_size)
#processor = AutoProcessor.from_pretrained(hf_checkpoint)
params = Parameters(device = device, use_cache = False, num_heads = 16, thresh = None, emb_dim = 256, max_seq_len = 256
                    ,ffn_hidden_dim = 512, batch_size = 8, div_batch = 8, 
                    tokenizer = tokenizer, vocab_size = tokenizer.vocab_size+1,
                    max_im_width = 240, max_im_height = 240, num_layers = 1, patch_size = 16, dataset_name = "ct_scan_data",
                    token_thresh = 0.3, imp_layer_hidden = 512)

dataloader = DataLoader(
    Medical_Data(params), 
    batch_size=params.batch_size, 
    collate_fn=Medical_Data(params).collate_fn
)

tk = TokenFusion(params)
print(infer(tokenizer,["text: Hello? How are you?","text: Medical Imaging right here", 
                        "image: LPMC/PMC908/PMC9088011_fimmu-13-881352-g002_undivided_1_1.jpg text: What does the image describe?"], params, tk))
