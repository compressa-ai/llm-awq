from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import argparse
import os
import json
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_in_model
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight, real_quantize_model_weight
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer


parser = argparse.ArgumentParser()
parser.add_argument('--qlora', type=bool, help='load qlora adapters')
parser.add_argument('--qlora_adapters_path', type=str, help='qlora adapters path')
parser.add_argument('--bnb', type=bool, help='load bnb model')
parser.add_argument('--model_path', type=str, help='path of the hf model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
# model config
parser.add_argument('--parallel', action='store_true',
                    help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument('--max_memory', type=str, nargs='*',
                    help="List of device_id:max_memory pairs to be parsed into a dictionary; " \
                        + "Example: 0:10GiB 1:10GiB cpu:30GiB; " \
                        + "mode details here: " \
                        + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling")
parser.add_argument('--auto_parallel', action='store_true',
                    help="automatically set parallel and batch_size")
# quantization config
parser.add_argument('--w_bit', type=int, default=None)
parser.add_argument('--q_group_size', type=int, default=-1)
parser.add_argument('--no_zero_point', action='store_true',
                    help="disable zero_point")
parser.add_argument('--q_backend', type=str,
                    default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument('--dump_quant', type=str, default=None,
                    help='save quantized model')
parser.add_argument('--load_quant', type=str, default=None,
                    help='load quantized model')
# apply/save/load awq
parser.add_argument('--run_awq', action='store_true',
                    help="perform awq search process")
parser.add_argument('--dump_awq', type=str, default=None,
                    help="save the awq search results")
parser.add_argument('--load_awq', type=str, default=None,
                    help="load the awq search results")
args = parser.parse_args()

max_memory = [v.split(':') for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k):v for k,v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization

}
print("Quantization config:", q_config)

# build model and tokenizer

def build_model_and_enc(model_path):
    # bnb args
    args.bf16 = False
    args.fp16 = False
    args.cache_dir = None
    args.double_quant = True
    args.quant_type = "nf4"
    args.trust_remote_code = True
    args.use_auth_token = False
    args.max_memory_MB = 80000

    max_memory = [v.split(':') for v in (args.max_memory or [])]
    max_memory = {(int(k) if k.isdigit() else k):v for k,v in max_memory}

    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if "mpt" in config.__class__.__name__.lower():
        enc = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    else:
        enc = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config=config,
                                                     torch_dtype=torch.float16, trust_remote_code=True)
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True)
        
        model.tie_weights()
        
        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"],
            **kwargs
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )

        if args.qlora:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, os.path.join(args.qlora_adapters_path, 'adapter_model'), is_trainable=True) 

            for name, module in model.named_modules():  # Different from original qlora
                if isinstance(module, LoraLayer):
                    module = module.to(torch.float16)

        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    elif args.bnb:
        print(f'loading base model {model_path}...')
        compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            
        max_memory = f'{args.max_memory_MB}MB'
        max_memory = {i: max_memory for i in range(n_gpus)}
        device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=args.cache_dir,
            load_in_4bit=args.w_bit == 4,
            load_in_8bit=args.w_bit == 8,
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.w_bit == 4,
                load_in_8bit=args.w_bit == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token
        )

        if args.qlora:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, os.path.join(args.qlora_adapters_path, 'adapter_model'), is_trainable=True) 

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if args.bf16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True, **kwargs)

        model.eval()

        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"
                        
            awq_results = run_awq(
                model, enc,
                w_bit=args.w_bit, q_config=q_config,
                n_samples=128, seqlen=512,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)
                
                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)
                
            exit(0)
                
        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert args.dump_quant is None, \
                    "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )
                if args.dump_quant:
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)
                    
                    print(
                        f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError
            
        if args.qlora:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, os.path.join(args.qlora_adapters_path, 'adapter_model'), is_trainable=True) 

            for name, module in model.named_modules():  # Different from original qlora
                if isinstance(module, LoraLayer):
                    module = module.to(torch.float16)

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {"max_memory": get_balanced_memory(model, max_memory if len(max_memory) > 0 else None)}
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"],
            **kwargs
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)

    if args.tasks is not None:
        task_names = args.tasks.split(",")

        lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=args.batch_size,
            no_cache=True,
            num_fewshot=args.num_fewshot,
        )

        print(evaluator.make_table(results))

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
