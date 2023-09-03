# micro lib to implement the watermarking extensions to LM generation
# as well as utils for eval/validaiton

from typing import List, Optional, Callable

import time
import random
import math
import torch
import json
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor, LogitsProcessorList, set_seed

class BaselineLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 tokenizer,
                 baseline_text
                 ):
        self.baseline_text = baseline_text
        self.tokenizer = tokenizer
        self.vocabularys = None
        
        self.baseline_ids = self.tokenizer.encode(self.baseline_text, return_tensors="pt")[0]
        
    def change_baseline_text(self, baseline_text):
        self.baseline_text = baseline_text
        self.baseline_ids = self.tokenizer.encode(self.baseline_text, return_tensors="pt")[0]
        
    def get_and_clear_vocabularys(self):
        old_vocabularys = self.vocabularys
        self.vocabularys = None
        return old_vocabularys
            
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size, vocabulary = cal_vocab(self.tokenizer, scores)
        if self.vocabularys is None:
            self.vocabularys = []
        self.vocabularys.append((vocabulary))
        
        #print("base vocabularys.num is ", len(self.vocabularys))
        #print("vocabularys[0] is ", self.vocabularys[0])
        last_id = input_ids[-1]
        
        baseline_id = self.baseline_ids[len(input_ids) - 1]
            
        # 如果输入的最后一个token和基线文本中的token不一致，说明输入不符合基线文本，返回一个空的分数张量
        if last_id != baseline_id:
            return torch.empty_like(scores)
        # 否则，计算出基线文本中的token在分数张量中的概率，并返回一个只包含这个概率的分数张量
        else:
            prob = scores.softmax(dim=-1)[baseline_id]
            return torch.tensor([prob])
        
def base_cal_main(tokenizer, example, model):
    prompts = example["truncated_input"]
    prompts1 = example["inputs"]
    baseline_completion = example["baseline_completion"]
    baseline_completion1 = example["my_baseout"]
    print("length of prompts1 is ", prompts1.shape[1])
    print("length of baseline_completion1 is ", baseline_completion1.shape[1])
    len_prompts = prompts1.shape[1]
    vocabularys = []
    # breakpoint()
    for i in range(baseline_completion1.shape[1]):
        
        # inputs = tokenizer.encode(prompts, return_tensors="pt", truncation=True)
        logits = model(prompts1.to(model.device)).logits
        # print("logits shape:", logits.shape) # 打印logits的形状
        scores = logits[:, -1, :]
        # print("scores shape:", scores.shape) # 打印scores的形状
        vocab_size, vocabulary = cal_vocab(tokenizer, scores)
        # print("vocab size:", vocab_size) # 打印词汇表大小
        # print("vocabulary:", vocabulary) # 打印词汇表内容
        vocabularys.append(vocabulary)
        # print("vocabularys size:", len(vocabularys)) # 打印词汇表大小
        c = torch.narrow(baseline_completion1,dim = 1,start=i,length=1)
        # print("c shape:", c.shape) # 打印c的形状
        prompts1 = torch.cat((prompts1, c), dim = 1)
        # print("prompts1 shape:", prompts1.shape) # 打印prompts1的形状
        
        
    print("base vocabularys.num is ", len(vocabularys))
    return vocabularys
    
          
def tokenize_and_truncate(example: dict,
                          completion_length: int = None,
                          prompt_length: int = None,
                          hf_model_name: str = None,
                          tokenizer = None,
                          model_max_seq_len: int = 4096,
                          base_processor: BaselineLogitsProcessor = None):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert "text" in example, "expects 'text' field to be present"
    # tokenize
    
    
    inputs = tokenizer.encode(example["text"], return_tensors="pt", truncation=True, max_length=model_max_seq_len)
    example.update({"untruncated_inputs": inputs})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        # print(f"fuck toke 1: {completion_length}, {prompt_length}")
        slice_length = min(inputs.shape[1]-1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        # print(f"fuck toke 2: {completion_length}, {prompt_length}")
        desired_comp_len = (inputs.shape[1]-1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError((f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                          f" but got completion_length:{completion_length},prompt_length:{prompt_length}"))

    # truncate
    my_baseout = inputs[:,inputs.shape[1]-slice_length:]
    inputs = inputs[:,:inputs.shape[1]-slice_length]
   
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name: 
        inputs[0,-1] = 1
    # else: pass
    example.update({"inputs": inputs})
    example.update({"my_baseout": my_baseout})
    return example

def cal_vocab(tokenizer, scores):
    #### 修改词汇 
        # 得到概率分布
        probs = scores.softmax(dim = -1)
        # 对概率分布按降序排列，得到前k个最高概率词的索引和概率值
        top_k_probs, top_k_indices = probs.topk(5, dim = -1, largest=True, sorted=True)
        # print("top_k_indices is ", top_k_indices)
        # print("top_k_probs is ", top_k_probs)
        top_k_words = []
        vocabulary = []
    
        for top_index in top_k_indices:
            # 调用tokenizer.decode方法，把索引转换成文本，并添加到列表中
            top_index = torch.squeeze(top_index)
            for index in top_index:
                # print("index2 is ", index)
                word = (tokenizer).decode(index)
                word = word.replace(" ", "")
                # print("word is " + word)
                vocabulary.append(word)
                top_k_words.append(word)
                # 获取word的同义词集
                synsets = wn.synsets(word)
                for synset in synsets:
                    name = synset.name()
                    parts = name.split(".")
                    word1 = parts[0]
                    # print ("word1 is " + word)
                    vocabulary.append(word1)
            
        
        # 输出    ,
        top_k_words = list(set(top_k_words))
        vocabulary = list(set(vocabulary))
        # print("top_k_words is " ,top_k_words)
        # print("vocabulary is " ,vocabulary)
        vocab_size = len(vocabulary)
        return vocab_size, vocabulary
        
        # return vocab_size
        

class WhitelistLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 tokenizer,
                 ):
        self.tokenizer = tokenizer
        self.vocabularys = None
        
    def get_and_clear_vocabularys(self):
        old_vocabularys = self.vocabularys
        self.vocabularys = None
        return old_vocabularys
            
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size, vocabulary = cal_vocab(self.tokenizer, scores)
        if self.vocabularys is None:
            self.vocabularys = []
        self.vocabularys.append((vocabulary))
        
        # print("wl vocabularys.num is ", len(self.vocabularys))
        # print("vocabularys[0] is ", self.vocabularys[0])
        return scores


class OurBlacklistLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the token ids of the words
            that should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, 
                tokenizer,
                bad_words_ids: List[List[int]], 
                eos_token_id: int,
                vocab: list[int], 
                all_vocab_size: int,
                bl_proportion: float=0.5,
                bl_logit_bias: float=1.0,
                bl_type: str = "hard", # "soft"
                initial_seed: int=None, 
                dynamic_seed: str=None, # "initial", "markov_1", None
                store_bl_ids: bool=True,
                store_spike_ents: bool = False,
                noop_blacklist: bool = False,
                ):
        
        self.vocab = vocab
        # self.vocab_size = vocab_size
        # self.vocabulary1 = list(tokenizer.get_vocab().values())
        # self.all_vocab_size = len(self.vocabulary1)
        self.tokenizer = tokenizer
        self.probs = None
        
        self.bl_proportion = bl_proportion
        self.bl_logit_bias = bl_logit_bias # \delta
        self.bl_type = bl_type
        # self.total_ex = 0
        self.vocabularys = None
        self.top_probs = None

        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed

        self.eos_token_id = eos_token_id
        # self.bad_words_id_length_1 = self._prepare_bad_words(bad_words_ids)

        self.bad_words_mask: Optional[torch.LongTensor] = None

        self.store_bl_ids = store_bl_ids
        self.bl_ids = None
        self.all_vocab_size = all_vocab_size

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None

        # hack to replace this with an approximation of infinite bias
        # so that the expectation coefficient will come out to 1.0
        if self.bl_type == "hard":
            self.bl_logit_bias = 10000 # FIXME to a value that is actually close to the largest soft watermark used

        alpha = torch.exp(torch.tensor(self.bl_logit_bias)).item()
        # gamma = self.bl_proportion
        gamma = 1.0-self.bl_proportion
        self.alpha = alpha
        self.gamma = gamma

        self.z_value = ((1-gamma)*(alpha-1))/(1-gamma+(alpha*gamma))
        self.expected_wl_coef = (gamma*alpha)/(1-gamma+(alpha*gamma))

        # catch for overflow when bias is "infinite"
        if self.alpha == torch.inf: # 如果无穷大
            self.z_value = 1.0
            self.expected_wl_coef = 1.0

        self.noop_blacklist = noop_blacklist
        if self.noop_blacklist: print(f"Blacklist processor for accounting only, no rescoring of logits")

        self.g_cuda = None
        self.large_prime = 15485863

    @property
    def blacklisted_ids(self):
        assert self.store_bl_ids, "Need to instantiate processor with `store_bl_ids` to be able to retrieve them later"
        # flatten the each indexes blacklist
        l_of_bl_ids = [[] for _ in range(len(self.bl_ids))]
        for b_idx, batch in enumerate(self.bl_ids):
            for l_of_l, seed in batch:
                bl_ids = [l[0] for l in l_of_l] # this was the main line, maybe unnecessary now?
                l_of_bl_ids[b_idx].append((bl_ids,seed))
        return l_of_bl_ids

    def get_and_clear_stored_bl_ids(self):
        old_bl_ids = self.bl_ids
        self.bl_ids = None
        return old_bl_ids

    def get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def get_and_clear_stored_spike_ents(self):
        spike_ents = self.get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1) # 也即p_k
        denoms = 1+(self.z_value*probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs
    
    def compute_ex(self, bl_ct):
        T = self.all_vocab_size
        eX = (T - bl_ct) / T
        return eX
    
    def get_vocabularys(self):
        vocabularys = [[] for _ in range(len(self.vocabularys))]
        
        for b_idx, vocabulary_list in enumerate(self.vocabularys):
            for vocabulary in vocabulary_list:
                vocabularys[b_idx].append(list(vocabulary))
        return vocabularys
    
    def get_and_clear_vocabularys(self):
        old_vocabularys = self.vocabularys
        self.vocabularys = None
        return old_vocabularys
    
    def get_and_clear_probs(self):
        old_probs = self.top_probs
        self.top_probs = None
        return old_probs


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        #### 修改词汇 
        # 得到概率分布
        probs = scores.softmax(dim = -1)
        # 对概率分布按降序排列，得到前k个最高概率词的索引和概率值
        top_k_probs, top_k_indices = probs.topk(5, dim = -1, largest=True, sorted=True)
        # print("top_k_indices is ", top_k_indices)
        # print("top_k_probs is ", top_k_probs)
        top_k_words = []
        vocabulary = []
    
        for top_index in top_k_indices:
            # 调用tokenizer.decode方法，把索引转换成文本，并添加到列表中
            top_index = torch.squeeze(top_index)
            for index in top_index:
                # print("index2 is ", index)
                word = (self.tokenizer).decode(index)
                word = word.replace(" ", "")
                # print("word is " + word)
                vocabulary.append(word)
                top_k_words.append(word)
                # 获取word的同义词集
                synsets = wn.synsets(word)
                for synset in synsets:
                    name = synset.name()
                    parts = name.split(".")
                    word1 = parts[0]
                    # print ("word1 is " + word)
                    vocabulary.append(word1)
            
        
        # 输出    ,
        top_k_words = list(set(top_k_words))
        vocabulary = list(set(vocabulary))
        print("top_k_words is " ,top_k_words)
        print("vocabulary is " ,vocabulary)
        vocab_size = len(vocabulary)
        if self.vocabularys is None:
            self.vocabularys = []
        self.vocabularys.append((vocabulary))
        
        print("vocabularys.num is ", len(self.vocabularys))
        print("vocabularys[0] is ", self.vocabularys[0])
        """
        vocab_size, vocabulary = cal_vocab(self.tokenizer, scores)
        if self.vocabularys is None:
            self.vocabularys = []
        self.vocabularys.append((vocabulary))
        
        # print("bl vocabularys.num is ", len(self.vocabularys))
        # print("vocabularys[0] is ", self.vocabularys[0])
        self.bad_words_id_length_1 = [None for _ in range(input_ids.shape[0])]

        if self.g_cuda is None:
            self.g_cuda = torch.Generator(device=input_ids.device)
            
        ### 修改
            
        vocabulary_ids_tensor_temp = torch.tensor(self.tokenizer.convert_tokens_to_ids(vocabulary), device=input_ids.device)
            
        uniques, cnt = vocabulary_ids_tensor_temp.unique(return_counts=True)
            
        vocabulary_ids_tensor = uniques[cnt == 1]
        vocab_size = len(vocabulary_ids_tensor)
        # print("vocabulary_ids.num is ", vocab_size)
        
        all_vocabularys = torch.tensor(self.vocab, device=input_ids.device)
        # input_ids.shape:(batch_size, sequence_length) 即一次生成的批次数量，sequence_length是每个输入序列的长度，没有指定批次数量默认是8
        # input_ids[b_idx]表示的是每一批次的输入序列，因为每一批次的输入序列是相同的
        # print("input_ids.shape[0] is ", input_ids.shape[0])
        for b_idx in range(input_ids.shape[0]):
            if self.dynamic_seed == "initial":
                self.g_cuda.manual_seed(self.large_prime*self.initial_seed)
            elif self.dynamic_seed == "markov_1":
                self.g_cuda.manual_seed(self.large_prime*input_ids[b_idx][-1].item())
            elif self.dynamic_seed is None:
                # let the rng evolve naturally - this is not a realistic setting
                pass

            
            
            # 方法1.1
            # bl_ct = int(vocab_size*self.bl_proportion)
            # blacklist_ids = torch.randperm(vocab_size, device=input_ids.device, generator=self.g_cuda)[:bl_ct] # ty Yuxin :]
            # 转换为 id 
            
            # 
            """
            bl_ct = int(vocab_size*self.bl_proportion)
            random_indices = torch.randperm(vocab_size, device=input_ids.device, generator=self.g_cuda)[:bl_ct]
            blacklist_ids = torch.index_select(vocabulary_ids_tensor, 0, random_indices)
            print("blacklist_ids is ", blacklist_ids)
            """
            
            
            
            
            # 方法1.2
            """
            wl_ct = int(vocab_size*(1 - self.bl_proportion))
            print("wl_ct is", wl_ct)
            total_bl_ct = self.all_vocab_size - wl_ct
            all_vocabularys = torch.tensor(self.vocab, device=input_ids.device)
            wl_random_indices = torch.randperm(vocab_size, device=input_ids.device, generator=self.g_cuda)[:wl_ct]
            whitelist_ids = torch.index_select(vocabulary_ids_tensor, 0, wl_random_indices)
            print("whitelist_ids is ", whitelist_ids)
            
            combined = torch.cat((all_vocabularys, whitelist_ids))
            uniques, counts = combined.unique(return_counts=True)
            blacklist_ids = uniques[counts == 1]
            print("bl_ct is", len(blacklist_ids))
            """
            
            # 方法2
            # print("all_vocabularys is", all_vocabularys)
            bl_ct = int(vocab_size* self.bl_proportion)
            bl_random_indices = torch.randperm(vocab_size, device=input_ids.device, generator=self.g_cuda)[:bl_ct]
            # print("vocabulary_ids_tensor is", vocabulary_ids_tensor)
            blacklist_ids = torch.index_select(vocabulary_ids_tensor, 0, bl_random_indices)
            # print("bl_random_indices is", bl_random_indices)
            # print("blacklist_ids1 is ", blacklist_ids)
            
            combined0 = torch.cat((vocabulary_ids_tensor, blacklist_ids))
            uniques0, counts0 = combined0.unique(return_counts=True)
            whitelist_ids1 = uniques0[counts0 == 1]
            
            combined = torch.cat((all_vocabularys, vocabulary_ids_tensor))
            uniques, counts = combined.unique(return_counts=True)
            rest_ids = uniques[counts == 1]
            
            rest_vocab_size = len(rest_ids)
            bl_ct1 = int(rest_vocab_size* self.bl_proportion)
            bl_random_indices1 = torch.randperm(rest_vocab_size, device=input_ids.device, generator=self.g_cuda)[:bl_ct1]
            blacklist_ids1 = torch.index_select(rest_ids, 0, bl_random_indices1)
            
            blacklist_ids = torch.cat((blacklist_ids, blacklist_ids1))
            
            
            
            # blacklist_ids = torch.complement(self.vocab, whitelist_ids)
            # print("bl_ct is ", bl_ct)
            
            
            # self.eX = self.compute_ex(bl_ct)
            # print("self.EX is ", self.eX)
            
            if self.store_bl_ids: 
                if self.bl_ids is None: self.bl_ids = [[] for _ in range(input_ids.shape[0])]
                self.bl_ids[b_idx].append((blacklist_ids,input_ids.tolist()[b_idx][-1]))

            if self.store_spike_ents: 
                if self.spike_entropies is None: self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self.compute_spike_entropy(scores[b_idx]))
            
            # self.bad_words_id_length_1[b_idx] = self._prepare_bad_words(blacklist_ids)
            # this logic may not really be necessary for our usecase
            self.bad_words_id_length_1[b_idx] = blacklist_ids
            
        # self.total_ex += self.eX
        if not self.noop_blacklist:
            self.bad_words_mask = self._calc_curr_bad_word_mask(scores)
            scores = self._set_scores_to_inf_for_banned_tokens(scores)
            
        if self.top_probs is None:
            self.top_probs = []
            
        ### if sampling
        
            
        probs = scores.softmax(dim=-1)
        top_id = probs.argmax().item()
        top_prob = probs.max().item()
        self.top_probs.append(top_prob)
        
        top_word = self.tokenizer.decode(top_id).strip()
        
        # print(f"top_word is {top_word}: prob is {top_prob}, {top_id in whitelist_ids1}")

        return scores

    def _prepare_bad_words(self, bad_words_ids: List[List[int]]) -> list[int]:
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [self.eos_token_id], bad_words_ids))
        return bad_words_ids
        # used to have more logic, not used now

    def _calc_curr_bad_word_mask(self, scores: torch.FloatTensor) -> torch.BoolTensor:
        bad_words_mask = torch.zeros_like(scores)
        for b_idx in range(len(self.bad_words_id_length_1)):
            bad_words_mask[b_idx][self.bad_words_id_length_1[b_idx]] = 1
        final_mask = bad_words_mask.bool()
        return final_mask

    def _set_scores_to_inf_for_banned_tokens(
        self, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
            # NOTE^^ Omitted logic for dynamic mask based on multi-token ban words
        """
        if self.bl_type == "hard":
            scores = scores.masked_fill(self.bad_words_mask, -float("inf"))
        elif self.bl_type == "soft":
            whitelist_mask = torch.logical_not(self.bad_words_mask)
            blacklist_mask = self.bad_words_mask
            scores[whitelist_mask] = scores[whitelist_mask] + self.bl_logit_bias
            # scores[blacklist_mask] = scores[blacklist_mask] - self.bl_logit_bias  # additive only
        else:
            raise NotImplementedError(f"unrecognized bl type {self.bl_type}!")          
        return scores


def score_sequence( vocabularys,
                    inputs: Tensor = None, 
                   outputs: Tensor = None, 
                   tokenizer: Tokenizer = None,
                #    logits_processor: LogitsProcessor = None,
                   initial_seed: int = None,
                   dynamic_seed: str = None,
                   bl_proportion: float = None,
                   use_cuda: bool = True,
                   record_hits: bool = False,
                   debug: bool = False,
                #    trim_tokens: int = 2,
                ):

    assert  (inputs is not None) and \
            (outputs is not None) and \
            (tokenizer is not None),"output tensor, tokenizer, and bl params req'd"
            # (logits_processor is not None), 
            
    # vocabularys = bl_processer.vocabularys
    
    
    # vocabulary = list(tokenizer.get_vocab().values())
    # vocab_size = len(vocabulary)
    

    model_generations = outputs.tolist()[0] # these are tensors unpack once for speed
    # toks_generated = model_generations[num_orig_input_tokens:]
    toks_generated = model_generations
    num_toks_generated = len(toks_generated)

    # num_toks_to_trim = trim_tokens*2
    # if (num_toks_generated-num_toks_to_trim > 0) == False: 
    #     return -1, -1
        
    # assert num_toks_generated > num_toks_to_trim, f"Need more than {num_toks_to_trim} toks total since we trim start and end a bit."

    # toks_generated = toks_generated[trim_tokens:-trim_tokens]
    # 该部分需要与__call__函数中相同保证每组的dynamic_seed相同
    if initial_seed is not None:
        random.seed(initial_seed)
    
    device = (torch.device("cuda") if use_cuda else torch.device("cpu"))
    g_cuda = torch.Generator(device=device)
    large_prime = 15485863

    bl_hits, hit_list = 0, []

    prev_token = inputs[0][-1].item()
    # prev_token = toks_generated[0] # haven't decided whether this edge effect matters

    wl_pro = []
    syn_hits, syn_hit_list = 0, []
    
    # for idx,tok_gend in enumerate(toks_generated[1:]):
    toks_gen_len = len(toks_generated)
    vocab_len = len(vocabularys)
    print("toks_generated.num is ", toks_gen_len)
    print("vocabularys.nums is", vocab_len)
    min_len = toks_gen_len
    if toks_gen_len != vocab_len:
        min_len = min(toks_gen_len, vocab_len)
    for idx,tok_gend in enumerate(toks_generated[:min_len]):
        
        # print("idx is ", idx)
        # print(f'tok_gend[{idx}] is ', tok_gend)
        # prev_token = model_generations[num_orig_input_tokens+idx-1]
        
        if dynamic_seed == "initial":
            g_cuda.manual_seed(large_prime*initial_seed)
        elif dynamic_seed == "markov_1":
            g_cuda.manual_seed(large_prime*prev_token)
        elif dynamic_seed is None:
            # let the rng evolve naturally - this is not a realistic setting
            pass
        
        vocabulary_ids_tensor_temp = torch.tensor(tokenizer.convert_tokens_to_ids(vocabularys[idx]), device=device)
        
        
        vocab = list(tokenizer.get_vocab().values())
        all_vocabularys = torch.tensor(vocab, device=device)
        uniques, cnt = vocabulary_ids_tensor_temp.unique(return_counts=True)
            
        vocabulary_ids_tensor = uniques[cnt == 1]
        vocab_size = len(vocabulary_ids_tensor)
        # print("vocabulary_ids.num is ", vocab_size)
        
        
        # bl_ct = int(vocab_size*bl_proportion)
        # posthoc_blacklist = torch.randperm(vocab_size, device=device, generator=g_cuda)[:bl_ct] # ty Yuxin :]
        # print("posthoc_blacklist is ", posthoc_blacklist)
        
        # 法1.1
        # vocabulary_ids_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(vocabularys[idx]))
        # bl_ct = int(vocab_size*bl_proportion)
        # random_indices = torch.randperm(vocab_size, device=device, generator=g_cuda)[:bl_ct]
        # posthoc_blacklist = torch.index_select(vocabulary_ids_tensor, 0, random_indices)
        # print("blacklist_ids is ", blacklist_ids)
        
        # 法1.2
        # vocabulary_ids_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(vocabularys[idx]))
        # wl_ct = int(vocab_size*(1 - bl_proportion))
        # print("wl_ct is", wl_ct)
        # # total_bl_ct = self.all_vocab_size - wl_ct
        # vocab = list(tokenizer.get_vocab().values())
        # all_vocabularys = torch.tensor(vocab, device=device)
        # wl_random_indices = torch.randperm(vocab_size, device=device, generator=g_cuda)[:wl_ct]
        # whitelist_ids = torch.index_select(vocabulary_ids_tensor, 0, wl_random_indices)
        # print("whitelist_ids is ", whitelist_ids)
        #     
        # combined = torch.cat((all_vocabularys, whitelist_ids))
        # uniques, counts = combined.unique(return_counts=True)
        # posthoc_blacklist = uniques[counts == 1]
        # print("bl_ct is", len(posthoc_blacklist))
        
        # 法2
        
        bl_ct = int(vocab_size * bl_proportion)
        bl_random_indices = torch.randperm(vocab_size, device=device, generator=g_cuda)[:bl_ct]
        blacklist_ids = torch.index_select(vocabulary_ids_tensor, 0, bl_random_indices)
        #print("blacklist_ids1 is ", blacklist_ids)
        print("blacklist_ids1_size is ", len(blacklist_ids))
        
        combined0 = torch.cat((vocabulary_ids_tensor, blacklist_ids))
        uniques0, counts0 = combined0.unique(return_counts=True)
        whitelist_ids1 = uniques0[counts0 == 1]
        
        combined = torch.cat((all_vocabularys, vocabulary_ids_tensor))
        uniques, counts = combined.unique(return_counts=True)
        rest_ids = uniques[counts == 1]
        
        rest_vocab_size = len(rest_ids)
        print("rest_vocab_size is ", rest_vocab_size)
        bl_ct1 = int(rest_vocab_size * bl_proportion)
        bl_random_indices1 = torch.randperm(rest_vocab_size, device=device, generator=g_cuda)[:bl_ct1]
        print("bl_random_indices1 size is", len(bl_random_indices1))
        blacklist_ids1 = torch.index_select(rest_ids, 0, bl_random_indices1)
        print("blacklist_ids1 size is", len(blacklist_ids1))
        posthoc_blacklist = torch.cat((blacklist_ids, blacklist_ids1))
        print("nums of posthoc_blacklist is ", len(posthoc_blacklist))


        tok_in_ph_bl = tok_gend in posthoc_blacklist
        if tok_in_ph_bl:
            bl_hits += 1
            hit_list.append(True)
        else:
            hit_list.append(False)
            
        if tok_gend in whitelist_ids1:
            syn_hits += 1
            syn_hit_list.append(True)
        else:
            syn_hit_list.append(False)
            
            
        if debug: 
            decoded_token = tokenizer.decode(tok_gend, skip_special_tokens=True)
            print(f"Token generated: '{decoded_token}' was in the blacklist {tok_in_ph_bl}")
        
        prev_token = tok_gend

    if debug: 
        print(f"wl hits / num tokens : {num_toks_generated-bl_hits}/{num_toks_generated} = {(num_toks_generated-bl_hits)/num_toks_generated:.02f}")
        print(f"bl hits / num tokens : {bl_hits}/{num_toks_generated} = {bl_hits/num_toks_generated:.02f}")

    
    wl_rate = num_toks_generated-bl_hits / num_toks_generated
    bl_rate = bl_hits / num_toks_generated
    wl_pro.append(wl_rate)
    
    if record_hits:
        return bl_hits, num_toks_generated, hit_list, wl_rate, syn_hit_list
    # bl_fraction = bl_hits/num_toks_generated
    return bl_hits, num_toks_generated, wl_rate, syn_hit_list


def tokenize_for_generation(example: dict, 
                            idx: int,
                            max_new_tokens: int=None,
                            min_prompt_tokens: int=None,
                            hf_model_name : str=None,
                            tokenizer: Tokenizer=None,
                            model: torch.nn.Module=None,
                            base_processor: BaselineLogitsProcessor=None):

    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    # preprocess for model generation/completion
    example = tokenize_and_truncate(example, 
                                    completion_length=max_new_tokens, 
                                    prompt_length=min_prompt_tokens,
                                    hf_model_name=hf_model_name,
                                    tokenizer=tokenizer,
                                    # model_max_seq_len=model.config.max_position_embeddings)
                                    model_max_seq_len=None,
                                    base_processor=base_processor)
    inputs = example["inputs"]
    # for calculating the baseline violation rate across the "gold" completion
    untruncated_inputs = example["untruncated_inputs"]

    # decode the preprocessed input to store for audit
    re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    example.update({"truncated_input":re_decoded_input})
    print("inputs_trun is " + str(re_decoded_input))
    print("len re_decoded_input is ", len(re_decoded_input))
    
    # also decode the original suffix of the input for audit as the baseline
    decoded_untruncated_input = tokenizer.batch_decode(untruncated_inputs, skip_special_tokens=True)[0]
    example.update({"baseline_completion":decoded_untruncated_input.replace(re_decoded_input,"")})
    #if base_processor:
    #    base_processor.change_baseline_text(decoded_untruncated_input.replace(re_decoded_input,""))
    print("len baseline_completion is ", len(example["baseline_completion"]))

    example.update({
        "orig_sample_length"            : untruncated_inputs.shape[1],
        "prompt_length"                 : inputs.shape[1],
        "real_completion_length"        : untruncated_inputs.shape[1] - inputs.shape[1],
    })
    return example


def generate_completions(example: dict, 
                        idx: int,
                        max_new_tokens: int=None,
                        hf_model_name : str=None,
                        tokenizer: Tokenizer=None,
                        model: torch.nn.Module=None,
                        no_bl_partial: Callable=None,
                        w_bl_partial: Callable=None, # 重要
                        base_partial: Callable=None,
                        # return_logits: bool=False,
                        bl_processor_list: LogitsProcessorList=None,
                        wl_processor_list: LogitsProcessorList=None,
                        base_processor_list: LogitsProcessorList=None):

    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    # # preprocess for model generation/completion
    # example = tokenize_and_truncate(example, 
    #                                 completion_length=max_new_tokens, 
    #                                 hf_model_name=hf_model_name,
    #                                 tokenizer=tokenizer,
    #                                 model_max_seq_len=model.config.max_position_embeddings)
    # inputs = example["inputs"]
    # # for calculating the baseline violation rate across the "gold" completion
    # untruncated_inputs = example["untruncated_inputs"]

    # # decode the preprocessed input to store for audit
    # re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    # example.update({"truncated_input":re_decoded_input})

    # # also decode the original suffix of the input for audit as the baseline
    # decoded_untruncated_input = tokenizer.batch_decode(untruncated_inputs, skip_special_tokens=True)[0]
    # example.update({"baseline_completion":decoded_untruncated_input.replace(re_decoded_input,"")})

    inputs = example["inputs"]
    # print ("inputs is ", tokenizer.batch_decode(inputs))
    re_decoded_input = example["truncated_input"]
    untruncated_inputs = example["untruncated_inputs"]
    baseline_completion = example["baseline_completion"]
    print ("truncated_input is ", re_decoded_input)

    # call the vanilla and watermarked generation function wrappers with the preprocessed inputs
    with torch.no_grad():

        samples_taken = 0
        max_retries = 10
        success = False
        while (success is False) and (samples_taken < max_retries):
            samples_taken += 1
            
            # set_seed(1234) # debugging the error when using sampling # leaving this off for now

            start_generation = time.time()
            example["base_vocabularys"] = base_cal_main(tokenizer, example, model)
            example["base_gen_time"] = time.time() - start_generation
            

            start_generation = time.time()
            outputs_no_bl = no_bl_partial(inputs.to(model.device))
            example["no_bl_gen_time"] = time.time() - start_generation

            # set_seed(1234) # debugging the error when using sampling

            start_generation = time.time()
            outputs_w_bl = w_bl_partial(inputs.to(model.device))
            example["w_bl_gen_time"] = time.time() - start_generation
            
            
            # outputs_base = base_partial(inputs.to(model.device))
            # example["base_gen_time"] = time.time() - start_generation

            # if return_logits:
            #     output_no_bl_dict = outputs_no_bl
            #     logits_no_bl = output_no_bl_dict.scores
            #     outputs_no_bl = output_no_bl_dict.sequences
            #     example["logits_no_bl"] = logits_no_bl

            #     output_w_bl_dict = outputs_w_bl
            #     logits_w_bl = output_w_bl_dict.scores
            #     outputs_w_bl = output_w_bl_dict.sequences
            #     example["logits_w_bl"] = logits_w_bl


            if bl_processor_list:
                if bl_processor_list[0].bl_ids is not None:
                    example["bl_ids"] = bl_processor_list[0].get_and_clear_stored_bl_ids()
                if bl_processor_list[0].spike_entropies is not None:
                    example["spike_entropies"] = bl_processor_list[0].get_and_clear_stored_spike_ents()
                if bl_processor_list[0].vocabularys is not None:
                    example["bl_vocabularys"] = bl_processor_list[0].get_and_clear_vocabularys()
                    
                if bl_processor_list[0].top_probs is not None:
                    example["bl_top_probs"] = bl_processor_list[0].get_and_clear_probs()
                    
            if wl_processor_list:
                if wl_processor_list[0].vocabularys is not None:
                    example["wl_vocabularys"] = wl_processor_list[0].get_and_clear_vocabularys()
                    
            # if base_processor_list:
            #     if base_processor_list[0].vocabularys is not None:
            #         example["base_vocabularys"] = base_processor_list[0].get_and_clear_vocabularys()
            try: 
                # decode and store the new generations for auditing
                no_bl_decoded_output = tokenizer.batch_decode(outputs_no_bl, skip_special_tokens=True)[0]
                example.update({"no_bl_output":no_bl_decoded_output.replace(re_decoded_input,"")})

                w_bl_decoded_output = tokenizer.batch_decode(outputs_w_bl, skip_special_tokens=True)[0]
                example.update({"w_bl_output":w_bl_decoded_output.replace(re_decoded_input,"")})
                
                #base_decoded_output = tokenizer.batch_decode#(outputs_base, skip_special_tokens=True)[0]
                #example.update({"base_output":base_decoded_output.replace(re_decoded_input,"")})
                
                success = True

            except:
                # log what happened
                print(f"Error while trying to decode the outputs of the model...")
                if samples_taken == 1:
                    print(f"truncated_input: {inputs.tolist()}")
                print(f"Result of attempt {samples_taken}")
                print(f"shape outputs_no_bl: {outputs_no_bl.shape}")
                no_bl_toks = outputs_no_bl.tolist()[0]
                print(f"outputs_no_bl: {no_bl_toks}")
                print(f"outputs_no_bl min: {min(no_bl_toks)}")
                print(f"outputs_no_bl max: {max(no_bl_toks)}")

                print(f"shape outputs_w_bl: {outputs_w_bl.shape}")
                w_bl_toks = outputs_w_bl.tolist()[0]
                print(f"outputs_w_bl: {w_bl_toks}")
                print(f"outputs_w_bl min: {min(w_bl_toks)}")
                print(f"outputs_w_bl max: {max(w_bl_toks)}")
        
        if success is False:
            print(f"Unable to get both a no_bl and w_bl output that were decodeable after {samples_taken} tries, returning empty strings.")
            example.update({"no_bl_output":""})
            example.update({"w_bl_output":""})
            if bl_processor_list:
                if bl_processor_list[0].bl_ids is not None:
                    example["bl_ids"] = []
                if bl_processor_list[0].spike_entropies is not None:
                    example["spike_entropies"] = []

    # Able to get lengths in here by checking 
    # truncated input shape versus the output shape

    example.update({
        "baseline_num_tokens_generated" : untruncated_inputs.shape[1] - inputs.shape[1], # want this earlier now
        "no_bl_num_tokens_generated"    : outputs_no_bl.shape[1] - inputs.shape[1],
        "w_bl_num_tokens_generated"     : outputs_w_bl.shape[1] - inputs.shape[1]
    })
    example.update({
        "no_bl_sec_per_tok"             : example["no_bl_gen_time"]/example["no_bl_num_tokens_generated"],
        "no_bl_tok_per_sec"             : example["no_bl_num_tokens_generated"]/example["no_bl_gen_time"],
        "w_bl_sec_per_tok"             : example["w_bl_gen_time"]/example["w_bl_num_tokens_generated"],
        "w_bl_tok_per_sec"             : example["w_bl_num_tokens_generated"]/example["w_bl_gen_time"],
    })
    
    # now done externally because these persist outside this func
    # # remove any fields we don't need to keep
    # del example["inputs"]
    # del example["untruncated_inputs"]
    
    return example


def compute_bl_metrics(example: dict, 
                        idx: int,
                        hf_model_name: str = None,
                        tokenizer: Tokenizer=None,
                        initial_seed: int = None,
                        dynamic_seed: str = None,
                        bl_proportion: float = None,
                        use_cuda: bool = None,
                        record_hits: bool = True,
                        limit_output_tokens: int = 0):

    # if example["idx"] == 3: breakpoint()
    # okay need to catch an odd bug here and fix things
    baseline_before = example["baseline_completion"]
    example["baseline_completion"] = baseline_before.replace(example["truncated_input"][:-1],"")
    if example["baseline_completion"] != baseline_before:
        print("baseline input replacement bug occurred!")
    
    no_bl_before = example["no_bl_output"]
    example["no_bl_output"] = no_bl_before.replace(example["truncated_input"][:-1],"")
    if example["no_bl_output"] != no_bl_before:
        print("no_bl_output input replacement bug occurred!")
    
    w_bl_before = example["w_bl_output"]
    example["w_bl_output"] = w_bl_before.replace(example["truncated_input"][:-1],"")
    if example["w_bl_output"] != w_bl_before:
        print("w_bl_output input replacement bug occurred!")
        
    bl_vocabularys = example["bl_vocabularys"]
    wl_vocabularys = example["wl_vocabularys"]
    base_vocabularys = example["base_vocabularys"]

    if ("w_bl_output_attacked" in example):
        w_bl_attacked_before = example["w_bl_output_attacked"]
        example["w_bl_output_attacked"] = w_bl_attacked_before.replace(example["truncated_input"][:-1],"")
        if example["w_bl_output_attacked"] != w_bl_attacked_before:
            print("w_bl_output_attacked input replacement bug occurred!")

    ##########

    # preprocess for model generation/completion
    inputs = tokenize_and_truncate({"text":example["truncated_input"]}, 
                                    completion_length=0, 
                                    hf_model_name=hf_model_name,
                                    tokenizer=tokenizer)["inputs"]

    baseline_outputs = tokenize_and_truncate({"text":example["baseline_completion"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]
    print("length of baseline_outputs is ", len(baseline_outputs))
    
    no_bl_outputs = tokenize_and_truncate({"text":example["no_bl_output"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]

    w_bl_outputs = tokenize_and_truncate({"text":example["w_bl_output"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]
    if "w_bl_output_attacked" in example:
        w_bl_attacked_outputs = tokenize_and_truncate({"text":example["w_bl_output_attacked"]}, 
                                            completion_length=0, 
                                            hf_model_name=hf_model_name,
                                            tokenizer=tokenizer)["inputs"][:,1:]
    else:
        w_bl_attacked_outputs = None

    if limit_output_tokens > 0:
        example["orig_baseline_completion"] = example["baseline_completion"]
        example["orig_real_completion_length"] = example["real_completion_length"]
        baseline_outputs = baseline_outputs[:,:limit_output_tokens]
        example["real_completion_length"] = baseline_outputs.shape[1]
        example["baseline_completion"] = tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)[0]

        example["orig_no_bl_output"] = example["no_bl_output"]
        example["orig_no_bl_num_tokens_generated"] = example["no_bl_num_tokens_generated"]
        no_bl_outputs = no_bl_outputs[:,:limit_output_tokens]
        example["no_bl_num_tokens_generated"] = no_bl_outputs.shape[1]
        example["no_bl_output"] = tokenizer.batch_decode(no_bl_outputs, skip_special_tokens=True)[0]

        example["orig_w_bl_output"] = example["w_bl_output"]
        example["orig_w_bl_num_tokens_generated"] = example["w_bl_num_tokens_generated"]
        w_bl_outputs = w_bl_outputs[:,:limit_output_tokens]
        example["w_bl_num_tokens_generated"] = w_bl_outputs.shape[1]
        example["w_bl_output"] = tokenizer.batch_decode(w_bl_outputs, skip_special_tokens=True)[0]

        example["orig_spike_entropies"] = example["spike_entropies"]
        example["spike_entropies"] = [example["spike_entropies"][0][:limit_output_tokens]]

        if "w_bl_output_attacked" in example:
            # raise NotImplementedError("Havent thought what to do yet for this")
            example["orig_w_bl_output_attacked"] = example["w_bl_output_attacked"]
            # example["orig_w_bl_attacked_num_tokens_generated"] = example["w_bl_attacked_num_tokens_generated"]
            w_bl_attacked_outputs = w_bl_attacked_outputs[:,:limit_output_tokens]
            example["w_bl_attacked_num_tokens_generated"] = w_bl_attacked_outputs.shape[1]
            example["w_bl_output_attacked"] = tokenizer.batch_decode(w_bl_attacked_outputs, skip_special_tokens=True)[0]

    # score the 3 sequence completions/outputs wrt to bl hits
    
    result = score_sequence(inputs=inputs,
                            outputs=baseline_outputs, # <-- real text completions
                            initial_seed=initial_seed,
                            dynamic_seed=dynamic_seed,
                            bl_proportion=bl_proportion, 
                            tokenizer=tokenizer,
                            use_cuda=use_cuda,
                            record_hits=record_hits,
                            debug=True,
                            vocabularys=base_vocabularys)
    if record_hits:
        bl_hits, num_toks_gend, hit_list, wl_rate, syn_hit_list = result
    else:
        bl_hits, num_toks_gend, wl_rate, syn_hit_list = result
        
    wl_hits = num_toks_gend - bl_hits
    example.update({"baseline_num_toks_gend_eq_0":(num_toks_gend == 0)})
    example.update({"base_wl_rate": wl_rate})
    example.update({"base_bl_hits": bl_hits})
    example.update({"base_num_toks_gend":num_toks_gend})
    example.update({"base_wl_hits":wl_hits})
    # if num_toks_gend < 0.99*example["real_completion_length"]: breakpoint()
    # if len(hit_list) < 0.99*example["real_completion_length"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    baseline_stats = {
        "baseline_whitelist_fraction": wl_frac,
        "baseline_blacklist_fraction": bl_frac
    }
    example.update(baseline_stats)
    
    base_bl_stats1 = {
        "baseline_whitelist_hits": wl_hits
    }
    example.update(base_bl_stats1)
    if record_hits: example.update({"baseline_hit_list":hit_list})
    
    #example.update({})
    
    # if "baseline_wl_fracs" not in 
    
    
    result = score_sequence(inputs=inputs,
                            outputs=no_bl_outputs, # <-- non-blacklisted version
                            initial_seed=initial_seed,
                            dynamic_seed=dynamic_seed,
                            bl_proportion=bl_proportion, 
                            tokenizer=tokenizer,
                            record_hits=record_hits,
                            debug=True,
                            vocabularys=wl_vocabularys)
    if record_hits:
        bl_hits, num_toks_gend, hit_list, wl_rate, syn_hit_list = result
    else:
        bl_hits, num_toks_gend, wl_rate, syn_hit_list = result
        
    wl_hits = num_toks_gend - bl_hits
    example.update({"no_bl_num_toks_gend_eq_0":(num_toks_gend == 0)})
    example.update({"wl_wl_rate": wl_rate})
    example.update({"no_bl_bl_hits": bl_hits})
    example.update({"no_bl_num_toks_gend":num_toks_gend})
    example.update({"no_bl_wl_hits":wl_hits})
    # if num_toks_gend < 0.99*example["no_bl_num_tokens_generated"]: breakpoint()
    # if len(hit_list) < 0.99*example["no_bl_num_tokens_generated"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    no_bl_stats = {
        "no_bl_whitelist_fraction": wl_frac,
        "no_bl_blacklist_fraction": bl_frac
    }
    example.update(no_bl_stats)
    no_bl_stats1 = {
        "no_bl_whitelist_hits": wl_hits
    }
    example.update(no_bl_stats1)
    if record_hits: example.update({"no_bl_hit_list":hit_list})
    
    result = score_sequence(inputs=inputs,
                            outputs=w_bl_outputs, # <-- blacklisted version
                            initial_seed=initial_seed,
                            dynamic_seed=dynamic_seed,
                            bl_proportion=bl_proportion, 
                            tokenizer=tokenizer,
                            record_hits=record_hits,
                            # breakpoint_on_hit=True, # banging head against wall
                            debug=True,
                            vocabularys=bl_vocabularys)
    if record_hits:
        bl_hits, num_toks_gend, hit_list, wl_rate, syn_hit_list = result
    else:
        bl_hits, num_toks_gend, wl_rate, syn_hit_list = result
        
    wl_hits = num_toks_gend - bl_hits
    example.update({"w_bl_num_toks_gend_eq_0":(num_toks_gend == 0)})
    example.update({"bl_wl_rate": wl_rate})
    example.update({"bl_bl_hits": bl_hits})
    example.update({"bl_num_toks_gend":num_toks_gend})
    example.update({"bl_wl_hits":wl_hits})
    
    example.update({"w_bl_syn_hit_list":syn_hit_list})
    # if num_toks_gend < 0.99*example["w_bl_num_tokens_generated"]: breakpoint()
    # if len(hit_list) < 0.99*example["w_bl_num_tokens_generated"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    w_bl_stats = {
        "w_bl_whitelist_fraction": wl_frac,
        "w_bl_blacklist_fraction": bl_frac
    }
    example.update(w_bl_stats)
    w_bl_stats1 = {
        "w_bl_whitelist_hits": wl_hits
    }
    example.update(w_bl_stats1)
    # record fracs
    if "wl_fracs" not in example:
        example["wl_fracs"] = [wl_frac]
    example["wl_fracs"].append(wl_frac)
    
    if record_hits: example.update({"w_bl_hit_list":hit_list})

    if w_bl_attacked_outputs is not None:
        result = score_sequence(inputs=inputs,
                                outputs=w_bl_attacked_outputs, # <-- blacklisted but attacked version
                                initial_seed=initial_seed,
                                dynamic_seed=dynamic_seed,
                                bl_proportion=bl_proportion, 
                                tokenizer=tokenizer,
                                record_hits=record_hits,
                                # breakpoint_on_hit=True, # banging head against wall
                                debug=True,
                                vocabularys=bl_vocabularys)
        if record_hits:
            bl_hits, num_toks_gend, hit_list = result
        else:
            bl_hits, num_toks_gend = result
        example.update({"w_bl_attacked_num_toks_gend_eq_0":(num_toks_gend == 0)})
        # if (num_toks_gend-bl_hits)/(num_toks_gend) < 1.0: breakpoint()

        if num_toks_gend == 0:
            # print("No tokens generated, odd, avoiding div by zero and returning -1's")
            wl_frac = -1
            bl_frac = -1
        else:
            wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
            bl_frac = bl_hits/num_toks_gend
        w_bl_attacked_stats = {
            "w_bl_attacked_num_tokens_generated": num_toks_gend,
            "w_bl_attacked_whitelist_fraction": wl_frac,
            "w_bl_attacked_blacklist_fraction": bl_frac
        }
        example.update(w_bl_attacked_stats)
        if record_hits: example.update({"w_bl_attacked_hit_list":hit_list})
    
    return example



def aggregate_bl_stats(example: dict, idx: int, stat_table: dict):
    
    stat_table["baseline_stats"]["whitelist_fraction"] += example["baseline_stats"]["whitelist_fraction"]
    stat_table["baseline_stats"]["blacklist_fraction"] += example["baseline_stats"]["blacklist_fraction"]
    
    stat_table["w_bl_stats"]["whitelist_fraction"] += example["w_bl_stats"]["whitelist_fraction"]
    stat_table["w_bl_stats"]["blacklist_fraction"] += example["w_bl_stats"]["blacklist_fraction"]

    stat_table["no_bl_stats"]["whitelist_fraction"] += example["no_bl_stats"]["whitelist_fraction"]
    stat_table["no_bl_stats"]["blacklist_fraction"] += example["no_bl_stats"]["blacklist_fraction"]

    stat_table["num_examples"] += 1

    return example


def compute_ppl_single(prefix_and_output_text = None,
                        output_text = None,
                        oracle_model_name = None,
                        oracle_model = None,
                        oracle_tokenizer = None):

    with torch.no_grad():
        tokd_prefix = tokenize_and_truncate({"text":prefix_and_output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]
        tokd_inputs = tokd_prefix
        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenize_and_truncate({"text":output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]

        tokd_inputs = tokd_inputs.to(oracle_model.device)
        # make labels, mark if not including all positions
        tokd_labels = tokd_inputs.clone().detach()
        tokd_labels[:,:tokd_labels.shape[1]-tokd_suffix.shape[1]+1] = -100

        outputs = oracle_model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss # avg CE loss all positions (except -100, TODO plz check that this is working correctly)
        ppl = torch.tensor(math.exp(loss))
    
    return loss.item(), ppl.item()


def evaluate_generation_fluency(example: dict, 
                                idx: int,
                                oracle_model_name = None,
                                oracle_model = None,
                                oracle_tokenizer = None):

    # pull out the required fields from the pipeline results
    inputs_plus_baseline_output = f"{example['truncated_input']}{example['baseline_completion']}"
    baseline_output = f"{example['baseline_completion']}"

    inputs_plus_no_bl_output = f"{example['truncated_input']}{example['no_bl_output']}"
    no_bl_output = f"{example['no_bl_output']}"

    inputs_plus_w_bl_output = f"{example['truncated_input']}{example['w_bl_output']}"
    w_bl_output = f"{example['w_bl_output']}"

    # add metrics
    loss, ppl = compute_ppl_single(inputs_plus_baseline_output, baseline_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["baseline_loss"] = loss
    example["baseline_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_no_bl_output, no_bl_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["no_bl_loss"] = loss
    example["no_bl_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_w_bl_output, w_bl_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["w_bl_loss"] = loss
    example["w_bl_ppl"] = ppl

    # del any temp values
    return example


def add_idx(example,idx):
    example.update({"idx":idx})
    return example


def check_input_lengths(example,idx, min_sample_len=0, min_prompt_len=0, min_completion_len=0, max_input_tokens=2000):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["real_completion_length"]

    # breakpoint()

    conds = all([
        orig_sample_length >= min_sample_len,
        prompt_length >= min_prompt_len,
        real_completion_length >= min_completion_len,
        orig_sample_length <= max_input_tokens,
    ])
    return conds


def check_output_lengths(example,min_output_len=0):
    no_bl_output_len = example["no_bl_num_tokens_generated"]
    w_bl_output_len = example["w_bl_num_tokens_generated"]
    conds = all([
        no_bl_output_len >= min_output_len,
        w_bl_output_len >= min_output_len,
    ])
    return conds


class NewWatermarkDetector():
    """
    Class for detecting watermarks
    """
    
    def __init__(self,
                 tokenizer,
                 vocab: list[int] = None,
                 gamma: float = 0.5,
                 delta: float = 5.0,
                 hash_key: int = 15485863,
                 initial_seed: int=None,
                 dynamic_seed: str=None, # "initial", "markov_1", None
                 device: torch.device = None,
                 select_green_tokens: bool = True,
                 ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.device = device
        self.tokenizer = tokenizer
        self.select_green_tokens = select_green_tokens
        # self.vocabularys = vocabularys
        
        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed
        self.rng = torch.Generator(device=self.device)

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = math.sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z
    
    def detect(self,
               tokens,
               tokenized_text: list[int]=None,
               debug: bool=True,
               return_scores: bool = True,
               inputs: list[int]=None):
        assert tokenized_text is not None, "Must pass tokenized string"
        
        assert inputs is not None,  "Must pass inputs"

        
        #if self.tokenizer is not None:
        #if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
        #    print("bos removed!!!")
        #    tokenized_text = tokenized_text[1:]
        
            
        green_token_count, green_token_mask = 0, []
        print("tokenized_text is ", tokenized_text)
        input_sequence = tokenized_text
        prev_token = inputs[-1]
        print("prev_token is: ", prev_token)
        
        # print("len of input sequence is ", len(input_sequence))
        # print("input_sequence is ", input_sequence)
        prev_token = input_sequence[0]
        for idx, tok_gend in enumerate(input_sequence[1:]):
            if self.dynamic_seed == "initial":
                self.rng.manual_seed(self.hash_key*self.initial_seed)
                
            elif self.dynamic_seed == "markov_1":
                self.rng.manual_seed(self.hash_key*prev_token)
                
            elif self.dynamic_seed == "None":
                pass
            
            if debug:
                decoded_token = self.tokenizer.decode(tok_gend, skip_special_tokens=True)
                print(f"Token generated: '{decoded_token}'")
                print("cur token decode is: ", decoded_token)
            
            vocabulary_ids_tensor_temp = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens[idx]["vocabulary"]), device=self.device)
            print("idx is ", idx)

        
            vocab = list(self.tokenizer.get_vocab().values())
            all_vocabularys = torch.tensor(vocab, device=self.device)
            uniques, cnt = vocabulary_ids_tensor_temp.unique(return_counts=True)

            vocabulary_ids_tensor = uniques[cnt == 1]
            vocab_size = len(vocabulary_ids_tensor)

            bl_ct = int(vocab_size * self.gamma)
            bl_random_indices = torch.randperm(vocab_size, device=self.device, generator=self.rng)[:bl_ct]
            blacklist_ids = torch.index_select(vocabulary_ids_tensor, 0, bl_random_indices)
            #print("blacklist_ids1 is ", blacklist_ids)
            # print("blacklist_ids1_size is ", len(blacklist_ids))
        
            combined0 = torch.cat((vocabulary_ids_tensor, blacklist_ids))
            uniques0, counts0 = combined0.unique(return_counts=True)
            whitelist_ids1 = uniques0[counts0 == 1]
        
            combined = torch.cat((all_vocabularys, vocabulary_ids_tensor))
            uniques, counts = combined.unique(return_counts=True)
            rest_ids = uniques[counts == 1]
        
            rest_vocab_size = len(rest_ids)
            # print("rest_vocab_size is ", rest_vocab_size)
            bl_ct1 = int(rest_vocab_size * self.gamma)
            bl_random_indices1 = torch.randperm(rest_vocab_size, device=self.device, generator=self.rng)[:bl_ct1]
            # print("bl_random_indices1 size is", len(bl_random_indices1))
            blacklist_ids1 = torch.index_select(rest_ids, 0, bl_random_indices1)
            # print("blacklist_ids1 size is", len(blacklist_ids1))
            posthoc_blacklist = torch.cat((blacklist_ids, blacklist_ids1))
            print("nums of posthoc_blacklist is ", len(posthoc_blacklist))

            
            #greenlist_size = int(self.vocab_size*self.gamma)
            #
            #vocab_permutation = torch.randperm(self.vocab_size, device=self.device,generator=self.rng)
            
            #greenlist_ids = vocab_permutation
            #if self.select_green_tokens: # directly
            #    greenlist_ids = vocab_permutation[:greenlist_size] # new
            #else: # select green via red
            #    greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]  # legacy behavior
                
            tok_in_ph_gl = tok_gend in posthoc_blacklist
            
            if tok_in_ph_gl:
                
                green_token_mask.append(False)
                
            else:
                green_token_count += 1
                green_token_mask.append(True)
                
            if debug:
                decoded_token = self.tokenizer.decode(tok_gend, skip_special_tokens=True)
                print(f"Token generated: '{decoded_token}' was in the blacklist {tok_in_ph_gl}")
                print("prev token decode is: ", decoded_token)
            
                
            
            prev_token = tok_gend
            print("prev_token is: ", prev_token)
        
        z_score = self._compute_z_score(green_token_count, len(input_sequence))
        
        return z_score


