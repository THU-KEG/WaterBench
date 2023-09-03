import argparse
from tqdm import tqdm
import os
import torch.nn.functional as F
import json
import time
import torch
from old_watermark import BlacklistLogitsProcessor
from our_watermark import OurBlacklistLogitsProcessor
from gptwm import GPTWatermarkLogitsWarper
from watermark_v2 import WatermarkLogitsProcessor

                       


from transformers import T5Tokenizer, AutoTokenizer, \
    AutoModel, AutoModelForCausalLM, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, \
    StoppingCriteria, StoppingCriteriaList, \
    LlamaTokenizer, LlamaForCausalLM, \
    LogitsProcessorList

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



class StopSequences(StoppingCriteria):
    def __init__(self, stop_sequences_set):
        super().__init__()
        self.stop_sequences_set = stop_sequences_set
    
    def __call__(self, input_ids, scores):
        if input_ids[0][-1].item() in self.stop_sequences_set:
            return True
        return False

def load_model(model_name, test_input):
    # only load tokenizer for test_input mode

    # find model
    model_paths = [i + model_name for i in [
        '/data2/lhm/cookie_huggingface_models/',
        '/data2/cookie_huggingface_models/',
    ]]
    for model_path in model_paths:
        if os.path.exists(model_path):
            break
    else:
        print(f'Model {model_name} not found!')
        return None, None, None

    # load model from local files
    # all supported models are listed as following:
    if model_name == 'flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True
        ) if not test_input else None
    elif model_name == 'flan-ul2':                                                               
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True, torch_dtype=torch.bfloat16
        ) if not test_input else None
    elif model_name == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True
        ) if not test_input else None
    elif model_name == 'T0pp':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True
        ) if not test_input else None
    # elif model_name == 'opt-iml-30b':  # TODO
    #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_path, device_map='auto', return_dict_in_generate=True, output_scores=True, torch_dtype=torch.float16
    #     ) if not test_input else None
    elif model_name == 'alpaca-lora-7b':
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True, torch_dtype=torch.float16, load_in_8bit=True
        ) if not test_input else None
    elif model_name == 'chatglm-6b':
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True, trust_remote_code=True
        ).half() if not test_input else None
        
    elif model_name in [
        'gpt-neox-20b',
        'GPT-JT-6B-v1',
        'gpt-j-6b',
        'bloom-7b1',
        'llama-65b-hf',
        'LLaMA-13B',
        'vicuna-13b',
    ]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True
        ) if not test_input else None
    else:
        print(f'Model {model_name} not supported!')
        tokenizer, model = None, None

    max_length = {
        'flan-t5-xxl': 512,
        'ul2': 512,
        'flan-ul2': 2048,
        'GPT-JT-6B-v1': 2048,
        'gpt-j-6b': 2048,
        'gpt-neox-20b': 2048,
        'bloom-7b1': 10000,  # no length overflow till now
        'T0pp': 512,
        'llama-65b-hf': 10000,
        'alpaca-lora-7b': 10000,
        'chatglm-6b': 10000,
        'LLaMA-13B': 512, 
        'vicuna-13b':2048,
    }[model_name]

    return tokenizer, model, max_length


def load_dataset(dataset_name, layer):
    
    # find dataset
    dataset_path = f'/data2/cookie/input/{layer}/{dataset_name}/'
    if not os.path.exists(dataset_path):
        print(f'Dataset {dataset_name} not found!')
        return None, None
    
    # load dataset

    # datasets with only one file which means examples have been attached to instances
    # other datasets have test.json and train.json and their inputs need to be assembled from the two files
    preprocessed_datasets_to_file = {
        'hotpotqa': 'hotpotqa_sample.json',
        'musique' : 'musique_sample.json',
        'kqapro'  : 'kqapro_sample.json',
        '2wikimultihopqa': '2WikiMultihopQA_sample.json',
    }

    if dataset_name.split('/')[0] not in preprocessed_datasets_to_file:
        test_file = dataset_path + 'test.json'
    else:
        test_file = dataset_path + preprocessed_datasets_to_file[dataset_name]
    with open(test_file) as f:
        dataset_test = json.load(f)
    
    dataset_train = None
    if dataset_name.split('/')[0] not in preprocessed_datasets_to_file:  # preprocessed datasets have no train.json
        with open(dataset_path + 'train.json') as f:
            dataset_train = json.load(f)

    return dataset_test, dataset_train


def main(args):

    model_names, dataset_names, layer, test_input, num_shot, init_seed, dyna_seed, bl_proportion, bl_logit_bias, bl_type, num_beams, sampling_temp = args.models, args.datasets, args.layer, args.test_input, args.num_shot, args.initial_seed, args.dynamic_seed, 1-args.gamma, args.delta, args.bl_type, args.num_beams, args.sampling_temp
    
    for model_name in model_names:

        tokenizer, model, max_length = load_model(model_name, test_input)
        if not tokenizer and not model:
            return
        
        if not test_input:
            result_path = './our_scores/' + model_name + '/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)
                
        all_token_ids = list(tokenizer.get_vocab().values())
        vocab_size = len(all_token_ids)
        print(f"Vocabulary size: {vocab_size}")

        # replace datasets to their sub tasks, if exist
        datasets_with_sub_tasks = {
            'COPEN': ['cic', 'cpj', 'csj'],
            'FewNERD': ['inter', 'intra', 'supervised'],
            'KoRC': ['iid', 'ood'],
        }
        dataset_names_temp = []
        for i in dataset_names:
            if i not in datasets_with_sub_tasks:
                dataset_names_temp.append(i)
            else:
                dataset_names_temp.extend((f'{i}/{j}' for j in datasets_with_sub_tasks[i]))  # 'task/subtask'
        dataset_names = dataset_names_temp

        for dataset_name in dataset_names:
            
            dataset_test, dataset_train = load_dataset(dataset_name, layer)
            if not dataset_test:
                return
            
            spec = dataset_test['adapter_spec']
            instruction = spec['instructions']
            input_prefix = spec['input_prefix']
            input_suffix = spec['input_suffix']
            output_prefix = spec['output_prefix']
            output_suffix = spec['output_suffix']
            instance_prefix = spec['instance_prefix'] if 'instance_prefix' in spec else ''  # some earlier datasets don't have this field

            # every query start with instruction
            if model_name == 'ul2':
                query_prefix = f'[S2S] {instruction}'
            else:
                query_prefix = instruction

            # instances in following datasets have been preprocessed, only instruction needs to be attached
            preprocessed_datasets = set([
                '2wikimultihopqa',
                'hotpotqa',
                'kqapro',
                'musique',
            ])
            
            if dataset_name not in preprocessed_datasets and dataset_train:  # if not preprocessed, examples need to be concatenated
                for i in dataset_train['request_states'][:num_shot]:
                    i_input = i['instance']['input']['text']
                    i_output = i['instance']['references'][0]['output']['text']
                    query_prefix += instance_prefix
                    query_prefix += f'{input_prefix}{i_input}{input_suffix}'
                    query_prefix += f'{output_prefix}{i_output}{output_suffix}'
                
            query_prefix += instance_prefix + input_prefix

            if not test_input:
                max_new_tokens = dataset_test['adapter_spec']['max_tokens']
                stop_sequences_set = set(tokenizer.encode(i)[0] for i in dataset_test['adapter_spec']['stop_sequences'])
                stop_criteria = StopSequences(stop_sequences_set)
                print(f'Running inference on {dataset_name} with {model_name}')

            if test_input:
                lens = []
            
                 
            bl_processor = BlacklistLogitsProcessor(tokenizer=tokenizer,
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=all_token_ids, 
                                            all_vocab_size=vocab_size, 
                                            bl_proportion=bl_proportion,
                                            bl_logit_bias=bl_logit_bias,
                                            bl_type=bl_type, 
                                            initial_seed=init_seed, 
                                            dynamic_seed=dyna_seed)
            if args.mode == 'new': 
                bl_processor = OurBlacklistLogitsProcessor(tokenizer=tokenizer,
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=all_token_ids, 
                                            all_vocab_size=vocab_size, 
                                            bl_proportion=bl_proportion,
                                            bl_logit_bias=bl_logit_bias,
                                            bl_type=bl_type, 
                                            initial_seed=init_seed, 
                                            dynamic_seed=dyna_seed)
            logit_processor_lst = LogitsProcessorList([bl_processor])
                
            if args.mode == 'gpt':
                watermark_processor = GPTWatermarkLogitsWarper(vocab_size=len(list(tokenizer.get_vocab().values())),
                                                           fraction=args.gamma,
                                                           strength=args.delta)
                
                logit_processor_lst = LogitsProcessorList([watermark_processor])   
                
            if args.mode == 'v2':
                watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                           gamma=args.gamma,
                                                           delta=args.delta,
                                                           seeding_scheme=args.seeding_scheme,
                                                           select_green_tokens=args.select_green_tokens)
                logit_processor_lst = LogitsProcessorList([watermark_processor]) 
            
        
            # run inference
            for request in tqdm(dataset_test['request_states']):
                
                input_text = query_prefix + request['instance']['input']['text']
                if dataset_name not in preprocessed_datasets:
                    input_text += input_suffix + output_prefix

                if model_name == 'ul2':
                    input_text += " <extra_id_0>"
                input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')
                
                if test_input:  # just print one input without running inference
                    lens.append(len(input_ids[0]))
                    continue
                
                if len(input_ids[0]) > max_length:  # mark and skip
                    print(f'skip overlong instances: {len(input_ids[0])}')
                    request['request'] = {
                        'result': {
                            'success': False,
                            'completions': [{
                                'text': 'null_overlength',
                                'logprob': 0.,
                                'tokens': [],
                            }],
                            'cached': True,
                            'request_time': 0.,
                            'request_datetime': int(time.time()),
                    }
                    }
                    continue
                
                
                if args.mode == 'new':
                    example = {}
                    start_time = time.time()
                    # TODO: pad_token_id=tokenizer.eos_token_id
                    outputs = model.generate(
                        input_ids, max_new_tokens=max_new_tokens,
                        stopping_criteria=StoppingCriteriaList([stop_criteria]),
                        logits_processor = logit_processor_lst,
                        do_sample=True,
                        top_k=0,
                        temperature=sampling_temp
                    )
                    request_time = time.time() - start_time
                    request_datetime = int(time.time())

                    example.update({"bl_vocabularys":logit_processor_lst[0].get_and_clear_vocabularys()})
                    # remove the attached input from output for some model
                    scores = outputs.scores
                    output_ids = outputs.sequences[0, -len(scores):]
    
                    # remove the tail, if generation stops at any stop_sequences
                    if output_ids[-1].item() in stop_sequences_set:
                        scores = scores[:-1]
                        output_ids = output_ids[:-1]
    
                    # compute logprob for each token
                    completions_tokens = []
                    completions_logprob = 0
    
                    for score, token, vocabulary in zip(scores, output_ids, example["bl_vocabularys"], strict=True):
                        logprobs = F.log_softmax(score[0], dim=-1)
                        logprob = logprobs[token].item()
                        completions_tokens.append({
                            'text': tokenizer.decode(token),
                            'logprob': logprob,
                            'vocabulary': vocabulary,
                        })
                        completions_logprob += logprob
                    
                    completions_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
                    request['request'] = {
                        'result': {
                            'success': True,
                            'completions': [{
                                'text': completions_text,
                                'logprob': completions_logprob,
                                'tokens': completions_tokens,
                            }],
                            'cached': True,
                            'request_time': request_time,
                            'request_datetime': request_datetime,
                        }
                    }
            
                else:    
                
                    if args.mode == 'no':
                        start_time = time.time()
                        # TODO: pad_token_id=tokenizer.eos_token_id
                        outputs = model.generate(
                            input_ids, max_new_tokens=max_new_tokens,
                            stopping_criteria=StoppingCriteriaList([stop_criteria]),
                        )
                        request_time = time.time() - start_time
                        request_datetime = int(time.time())

                    elif args.mode == 'old':
                        start_time = time.time()
                        # TODO: pad_token_id=tokenizer.eos_token_id
                        outputs = model.generate(
                            input_ids, max_new_tokens=max_new_tokens,
                            stopping_criteria=StoppingCriteriaList([stop_criteria]),
                            logits_processor = logit_processor_lst,
                            do_sample=True,
                            top_k=0,
                            temperature=sampling_temp
                        )
                        request_time = time.time() - start_time
                        request_datetime = int(time.time())

                    elif args.mode == 'gpt':
                        start_time = time.time()
                        # TODO: pad_token_id=tokenizer.eos_token_id
                        outputs = model.generate(
                            input_ids, max_new_tokens=max_new_tokens,
                            stopping_criteria=StoppingCriteriaList([stop_criteria]),
                            logits_processor = logit_processor_lst,
                            do_sample=True,
                            top_k=0,
                            top_p=0.9
                        )
                        request_time = time.time() - start_time
                        request_datetime = int(time.time())

                    elif args.mode == 'v2':
                        start_time = time.time()
                        # TODO: pad_token_id=tokenizer.eos_token_id
                        outputs = model.generate(
                            input_ids, max_new_tokens=max_new_tokens,
                            stopping_criteria=StoppingCriteriaList([stop_criteria]),
                            logits_processor = logit_processor_lst,
                            do_sample=True,
                            top_k=0,
                            temperature=sampling_temp
                        )
                        request_time = time.time() - start_time
                        request_datetime = int(time.time())
        
                    # remove the attached input from output for some model
                    scores = outputs.scores
                    output_ids = outputs.sequences[0, -len(scores):]
    
                    # remove the tail, if generation stops at any stop_sequences
                    if output_ids[-1].item() in stop_sequences_set:
                        scores = scores[:-1]
                        output_ids = output_ids[:-1]
    
                    # compute logprob for each token
                    completions_tokens = []
                    completions_logprob = 0
    
                    for score, token in zip(scores, output_ids, strict=True):
                        logprobs = F.log_softmax(score[0], dim=-1)
                        logprob = logprobs[token].item()
                        completions_tokens.append({
                            'text': tokenizer.decode(token),
                            'logprob': logprob,
                        })
                        completions_logprob += logprob
                    
                    completions_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
                    request['request'] = {
                        'result': {
                            'success': True,
                            'completions': [{
                                'text': completions_text,
                                'logprob': completions_logprob,
                                'tokens': completions_tokens,
                            }],
                            'cached': True,
                            'request_time': request_time,
                            'request_datetime': request_datetime,
                        }
                    }
            
            dataset_name = dataset_name.replace('/', '++')  # rename sub task, e.g. KoRC/iid -> KoRC++iid, as filename
            if not test_input:
                prefix = 'r_' if layer == 'Rolling' else ''  # distinguish the datasets from Rolling, as they have the same names with the former ones 
                with open(result_path + f'{prefix}{dataset_name}_inference_base.json', 'w') as f:
                    json.dump(dataset_test, f, indent=2)
            else:
                print(input_text)
                print(sorted(lens))

parser = argparse.ArgumentParser(description="Process watermark to calculate z-score for every method")

parser.add_argument(
    "--mode",
    type=str,
    default="old",
    choices=["no", "old", "new", "v2", "gpt"],
    help="Which version of the watermark to generate",
)

parser.add_argument(
    '--datasets',
    type=str,
    required=True,
    nargs='+')

parser.add_argument(
    '--layer',
    required=True)

parser.add_argument(
    '--test_input',
    action='store_true')

parser.add_argument(
    '--num_shot',
    type=int,
    default=5)

parser.add_argument(
    '--models',
    type=str,
    default="vicuna-13b",
    required=True,
    nargs='+')

parser.add_argument(
    '--initial_seed',
    type=int,
    default=1234,
    help=("The initial seed to use in the blacklist randomization process.", 
    "Is unused if the process is markov generally. Can be None."),
    )

parser.add_argument(
    "--dynamic_seed",
    type=str,
    default="markov_1",
    choices=[None, "initial", "markov_1"],
    help="The seeding procedure to use when sampling the blacklist at each step.",
    )

parser.add_argument(
    "--gamma",
    type=float,
    default=0.5)

parser.add_argument(
    "--delta",
    type=float,
    default=5.0)

parser.add_argument(
    "--bl_type",
    type=str,
    default="soft",
    choices=["soft", "hard"],
    help="The type of blacklisting being performed.",
    )
parser.add_argument(
    "--num_beams",
    type=int,
    default=1,
    help="The number of beams to use where '1' is no beam search.",
    )
parser.add_argument(
    "--sampling_temp",
    type=float,
    default=0.7,
    help="The temperature to use when generating using multinom sampling",
    )
 





parser.add_argument(
    "--wm_key", 
    type=int, 
    default=0)

parser.add_argument(
    '--test_input',
    action='store_true')

parser.add_argument(
    "--threshold",
    type=float,
    default=6.0)

parser.add_argument(
    "--test_min_tokens",
    type=int, 
    default=2)

parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
)

parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )

parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )

parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )

args = parser.parse_args()

main(args)
