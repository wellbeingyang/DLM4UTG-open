import time
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import read_jsonl, write_jsonl
import torch.multiprocessing as mp
import argparse
from utils_testeval import remove_extra, reformat_case_byrules, process_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Generate test cases")
    parser.add_argument('-n', '--num_tests', type=int, default=5, help='Number of test cases to generate for each function')
    parser.add_argument('-l', '--language', type=str, default='python', choices=['python', 'java', 'cpp'], help='Language to be tested')
    return parser.parse_args()

args = parse_args()

class CausalLMFlopsProfiler:
    def __init__(self, model):
        """
        初始化 Profiler
        model: HuggingFace AutoModelForCausalLM (CausalLM)
        """
        self.model = model
        self.config = model.config

        # transformer 基本参数
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.vocab_size = self.config.vocab_size
        self.intermediate_size = getattr(self.config, "intermediate_size", self.hidden_size * 4)

        self.flops_cache = []

        for i in range(4096):
            self.flops_cache.append(self.flops_full_forward(i))

    def flops_full_forward(self, seq_len: int) -> float:
        """
        计算单个 token 的 FLOPs（包括注意力 + FFN + 输出投影）
        seq_len: 当前输入序列长度（含 prompt + 已生成 token）
        返回值单位：FLOPs
        """

        d_model = self.hidden_size
        h = self.num_heads
        d_ff = self.intermediate_size
        L = seq_len
        n_layers = self.num_layers

        # -------------------------
        # 1) 自注意力 (scaled dot-product)
        # Q = xW_Q, K = xW_K, V = xW_V: 3*(2*L*d_model*d_model)
        # QK^T: 2*L*L*d_model, softmax(QK^T/\sqrt{d})V: 2*L*L*d_model
        # 注意力输出 + residual projection: 2*L*d_model*d_model
        attn_flops = n_layers * (8 * L * d_model * d_model + 4 * L * L * d_model) # 近似公式

        # 2) Feed-Forward Network: 两个线性层
        ffn_flops = n_layers * (4 * L * d_model * d_ff)

        # 3) LM 输出投影 (to vocab)
        lm_head_flops = 2 * L * d_model * self.vocab_size

        total_flops = attn_flops + ffn_flops + lm_head_flops
        return total_flops


    def estimate_flops_batch(self, prompt_tokens_batch: list, generated_tokens_batch: list) -> float:
        """
        支持 batch，生成不同长度
        prompt_tokens_batch: list[int] 每个样本 prompt 长度
        generated_tokens_batch: list[int] 每个样本生成长度
        """
        total_flops = 0
        for prompt_len, new_tokens in zip(prompt_tokens_batch, generated_tokens_batch):
            for seq_len in range(prompt_len, prompt_len + new_tokens):
                total_flops += self.flops_cache[seq_len]
        return total_flops

    @staticmethod
    def flops_to_tflops(flops: float) -> float:
        return flops / 1e12

    @staticmethod
    def flops_to_gflops(flops: float) -> float:
        return flops / 1e9

model_path = "Qwen/Qwen2.5-7B-Instruct"
model_abbrv = model_path.split('/')[-1]

language = args.language
n = args.num_tests

prompt_template = open(f'prompts/basic_{language}.txt').read()
system_message = open('prompts/system.txt').read().format(language=language)

output_dir = Path('outputs')

total_params = 7615616512

def generate_multiple(model, tokenizer, prompt, profiler, system_message=''):
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text]*n, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=1.0,
        use_cache=False,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    total_flops = profiler.estimate_flops_batch([model_inputs["input_ids"].shape[1]]*n, [g.shape[0] for g in generated_ids])
    total_tflops = profiler.flops_to_tflops(total_flops)

    return responses, total_tflops


def run_inference(rank, data):
    dataset = data[rank]
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    profiler = CausalLMFlopsProfiler(model)

    data_size = len(dataset)
    testing_results = []

    start = time.time()
    total_tflops = 0
    total_tokens = 0
    for data in tqdm(dataset, total=data_size, desc=f"Inference on GPU {rank}"):
        func_name = data['func_name']
        desc = data['description']
        code = data[f'{language}_solution']
        difficulty = data['difficulty']

        prompt = prompt_template.format(program=code, description=desc, func_name=func_name)
        generated_tests, tflops = generate_multiple(model, tokenizer, prompt, profiler, system_message)
        for test in generated_tests:
            total_tokens += len(tokenizer.encode(test, add_special_tokens=False))

        testing_data = {'task_num': data['task_num'], 'task_title': data['task_title'], 'func_name': func_name, 'difficulty': difficulty, 'code': code, 'tests': generated_tests}

        testing_results.append(testing_data)
        total_tflops += tflops
    eclipsed = time.time() - start
    print(f"GPU {rank} finished inference in {eclipsed:.2f} seconds.")
    print(f"GPU {rank} finished inference with {total_tflops} tflops.")
    print(f"GPU {rank} generated {total_tokens} tokens.")
    write_jsonl(testing_results, output_dir / f'testeval_{model_abbrv}_rank{rank}.jsonl')


def reformat_cov(datapath, newpath, language):
    data = read_jsonl(datapath)
    formatted_data = []
    for e in data:
        func_name = e['func_name']
        test_funcname = f'test_{func_name}'
        formatted_test_cases = []
        testcases = e['tests']
        for i, testcase in enumerate(testcases):
            extracted_testcase = remove_extra(testcase, language)
            reformatted_testcase = reformat_case_byrules(extracted_testcase, test_funcname, i, language)
            formatted_test_cases.append(reformatted_testcase)
        e['tests'] = formatted_test_cases
        formatted_data.append(e)
    write_jsonl(formatted_data, newpath)


def check_correctness(generated_data, language):
    """Compute syntactical and execution correctness (with coverage)."""
    total_cases = 0
    total_syn_correct = 0
    total_exec_correct = 0

    syn_fails = []
    exec_fails = []

    total_line_cov = 0
    total_branch_cov = 0

    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(partial(process_sample, language=language), generated_data), total=len(generated_data)))

    for res in results:
        total_cases += res['total_cases']
        total_syn_correct += res['syn_correct']
        total_exec_correct += res['exec_correct']
        syn_fails.extend(res['syn_fails'])
        exec_fails.extend(res['exec_fails'])
        total_line_cov += res['total_line_cov']
        total_branch_cov += res['total_branch_cov']

    shutil.rmtree('./tmp')

    syn_correct = total_syn_correct / total_cases
    exec_correct = total_exec_correct / total_cases
    print(f'Syntax Correctness: {syn_correct}')
    print(f'Executable Correctness: {exec_correct}')

    #compute coverage
    avg_line_cov = total_line_cov / len(generated_data)
    avg_branch_cov = total_branch_cov / len(generated_data)
    print(f'Average Line Coverage: {avg_line_cov}, Average Branch Coverage: {avg_branch_cov}')
    return {'syn_correct': syn_correct, 'exec_correct': exec_correct}, exec_fails


if __name__=='__main__':
    # inference
    gpu_num = torch.cuda.device_count()

    print('Model:', model_abbrv)

    data = read_jsonl('data/leetcode.jsonl')
    chunked_data = [data[i::gpu_num] for i in range(gpu_num)]

    output_dir.mkdir(exist_ok=True)

    mp.spawn(
        run_inference,
        args=(chunked_data,),
        nprocs=gpu_num,
        join=True
    )

    all_results = []
    for rank in range(gpu_num):
        result_file = output_dir / f'testeval_{model_abbrv}_rank{rank}.jsonl'
        if result_file.exists():
            all_results.extend(read_jsonl(result_file))
        result_file.unlink()
    write_jsonl(all_results, output_dir / f'testeval_{model_abbrv}_{language}.jsonl')

    # format
    file_name = f"testeval_{model_abbrv}_{language}.jsonl"
    formatted_file_name = f"testeval_{model_abbrv}_{language}_format.jsonl"
    print("Reformatting test cases...")
    print('input file:', output_dir / file_name)
    print('output file:', output_dir / formatted_file_name)
    reformat_cov(output_dir / file_name, output_dir / formatted_file_name, language=language)

    # evaluation
    predictions = read_jsonl(output_dir / formatted_file_name)
    check_correctness(predictions, language=language)

