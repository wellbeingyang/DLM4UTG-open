import torch
import numpy as np
import random
from accelerator import tree_accelerator
import json

def read_jsonl(path):
    data=[]
    with open(path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data,path):
    with open(path,'w') as f:
        for d in data:
            f.write(json.dumps(d)+'\n')


def use_customized_code(model_path, code_file="generation_utils_diffucoder.py"):
    """
    Change the generation_utils.py file in the model path 
    """
    with open(code_file, "r") as f:
        new_code = f.read()
    with open(model_path + "/generation_utils.py", "w") as f:
        f.write(new_code)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@ torch.no_grad()
def generate_diffucoder(model, tokenizer, messages, n=3, accelerate=True, steps=32, gen_length=128, temperature=1.5, top_p=1, alg="maskgit_plus", alg_temp=0, threshold=0.02, seed=42, end_token_id=151645, pad_token_id=151667):
    setup_seed(seed)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device=model.device)
    attention_mask = inputs.attention_mask.to(device=model.device)

    def lin_increasing_points(total=64, m=8, q=4.0):
        """
        在[0,total]上取 m+1 个点（含0与total），
        相邻间隔线性增大，最后一个间隔是第一个的 q 倍。
        """
        a = 2*total / (m*(1+q))
        r = a*(q-1)/(m-1) if m > 1 else 0
        gaps = [a + i*r for i in range(m)]
        xs = [0]
        s = 0.0
        for g in gaps:
            s += g
            xs.append(int(s))
        return xs  # 长度 m+1，xs[0]=0, xs[-1]=total

    points = lin_increasing_points(total=steps, m=20, q=5.0)

    def diffucoder_accelerate(x, x0, confidence, step):
        mask_token_id = 151666
        x0 = torch.where((x == mask_token_id), x0, x)

        if step % 2 == 0:
        # if step in points:
            unmask_index = tree_accelerator(tokenizer.batch_decode(x0.reshape(-1)), x.reshape(-1) != mask_token_id).reshape(x0.shape)
            unmask_index_confidence = unmask_index & (confidence >= threshold)
            mask = (x == mask_token_id) & unmask_index_confidence
            all_tokens = tokenizer.batch_decode(x0[mask])
            x = torch.where(unmask_index_confidence, x0, x)
        else:
            unmask_index = x != mask_token_id
            all_tokens = []

        # unmask_index = x != mask_token_id
        # all_tokens = []

        mask1 = (((x == end_token_id) | (x == pad_token_id)) & unmask_index).cumsum(dim=1) > 0
        mask1 = mask1 & (~unmask_index)
        x[mask1] = pad_token_id

        return x, all_tokens

    output = model.diffusion_generate(
        input_ids,
        num_return_sequences=n,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        output_history=True,
        return_dict_in_generate=True,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        alg=alg,
        alg_temp=alg_temp,
        generation_tokens_hook_func = diffucoder_accelerate if accelerate else lambda **kwags: (kwags["x"], []),
    )

    return output

