import time
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import shutil
from transformers import AutoTokenizer, AutoModel
from utils import generate_diffucoder, read_jsonl, write_jsonl, use_customized_code
import torch.multiprocessing as mp
import argparse
from utils_testeval import remove_extra, reformat_case_byrules, process_sample
from accelerator import set_language

def parse_args():
    parser = argparse.ArgumentParser(description="Generate test cases with DiffuCoder")
    parser.add_argument('-p', '--model_path', type=str, default='/ckpts/DiffuCoder-7B-cpGRPO', help='Path to the model')
    parser.add_argument('-n', '--num_tests', type=int, default=5, help='Number of test cases to generate for each function')
    parser.add_argument('-s', '--steps', type=int, default=64, help='Steps to generate')
    parser.add_argument('-t', '--temperature', type=float, default=1.5, help='Sampling temperature')
    parser.add_argument('-l', '--language', type=str, default='python', choices=['python', 'java', 'cpp'], help='Language to be tested')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum generation length')
    parser.add_argument('--threshold', type=float, default=0.02, help='Confidence threshold to unmask')
    parser.add_argument('--accelerate', action='store_true', help='Whether to use accelerate for generation')
    parser.add_argument('--process', type=int, default=32, help='Number of processes for evaluation')
    parser.add_argument('--alg', type=str, default="maskgit_plus", help="maskgit_plus, eb_sampler, topk_margin, entropy")
    return parser.parse_args()

args = parse_args()

model_path = args.model_path
model_abbrv = model_path.split('/')[-1]

if model_abbrv == "DiffuCoder-7B-cpGRPO":
    end_token_id = 151645
    pad_token_id = 151667
elif model_abbrv == "Dream-v0-Instruct-7B":
    end_token_id = 151643
    pad_token_id = end_token_id
elif model_abbrv == "Dream-Coder-v0-Instruct-7B":
    end_token_id = 151643
    pad_token_id = end_token_id

language = args.language
set_language(language)
use_customized_code(model_path)

prompt_template = open(f'prompts/basic_{language}.txt').read()
system_message = open('prompts/system.txt').read().format(language=language)

output_dir = Path('outputs')


def generate_multiple(model, tokenizer, prompt, system_message=''):
    generated_tests=[]

    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    output = generate_diffucoder(model, tokenizer, messages=messages, n=args.num_tests, steps=args.steps, gen_length=args.max_length, temperature=args.temperature, alg=args.alg, alg_temp=0.2, threshold=args.threshold, accelerate=args.accelerate, end_token_id=end_token_id, pad_token_id=pad_token_id)
    outputs, tflops_per_step = output.history, output.tflops_per_step
    for output in outputs[-1][0]:
        generated_test = tokenizer.decode(output, skip_special_tokens=True)
        generated_tests.append(generated_test)

    return generated_tests, tflops_per_step * len(outputs)


def run_inference(rank, data):
    dataset = data[rank]
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
        generated_tests, tflops = generate_multiple(model, tokenizer, prompt, system_message)
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

    with Pool(processes=args.process) as pool:
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
