from tqdm import tqdm
import os
import torch.nn.functional as F
from watermark.old_watermark import BlacklistLogitsProcessor
from watermark.our_watermark import OurBlacklistLogitsProcessor
from watermark.gptwm import GPTWatermarkLogitsWarper
from watermark.watermark_v2 import WatermarkLogitsProcessor
from transformers import  LogitsProcessorList
    
    
class Generator():
    def __init__(self, args, tokenizer, model) -> None:
        self.mode = args.mode # watermark mode
        self.init_seed, self.dyna_seed, self.gamma, \
        self.delta, self.bl_type, self.num_beams, self.sampling_temp = args.initial_seed, args.dynamic_seed, args.gamma, args.delta, args.bl_type, args.num_beams, args.sampling_temp
        self.tokenizer = tokenizer
        self.model = model # language model
        self.all_token_ids = list(tokenizer.get_vocab().values())
        self.vocab_size = len(self.all_token_ids)
        self.bl_processor = BlacklistLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed)
        self.logit_processor_lst = LogitsProcessorList([self.bl_processor])
        if args.mode == 'new': 
            self.bl_processor = OurBlacklistLogitsProcessor(tokenizer=tokenizer,
                                        bad_words_ids=None, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        vocab=self.all_token_ids, 
                                        all_vocab_size=self.vocab_size, 
                                        bl_proportion=1-self.gamma,
                                        bl_logit_bias=self.delta,
                                        bl_type=self.bl_type, 
                                        initial_seed=self.init_seed, 
                                        dynamic_seed=self.dyna_seed)
            self.logit_processor_lst = LogitsProcessorList([self.bl_processor])
            
        if args.mode == 'gpt':
            watermark_processor = GPTWatermarkLogitsWarper(vocab_size=len(list(tokenizer.get_vocab().values())),
                                                        fraction=args.gamma,
                                                        strength=args.delta)
            
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])   
            
        if args.mode == 'v2':
            watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        seeding_scheme=args.seeding_scheme,
                                                        select_green_tokens=args.select_green_tokens)
            self.logit_processor_lst = LogitsProcessorList([watermark_processor]) 
            
    def generate(self, input_ids, max_new_tokens):
        if self.mode == 'new':
            example = {}
            
            outputs = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                logits_processor = self.logit_processor_lst,
                do_sample=True,
                top_k=0,
                temperature=self.sampling_temp
            )

            example.update({"bl_vocabularys":self.logit_processor_lst[0].get_and_clear_vocabularys()})
            # remove the attached input from output for some model
            scores = outputs.scores
            output_ids = outputs.sequences[0, -len(scores):]


            # compute logprob for each token
            completions_tokens = []
            completions_logprob = 0

            for score, token, vocabulary in zip(scores, output_ids, example["bl_vocabularys"], strict=True):
                logprobs = F.log_softmax(score[0], dim=-1)
                logprob = logprobs[token].item()
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                    'logprob': logprob,
                    'vocabulary': vocabulary,
                })
                completions_logprob += logprob
            
            completions_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return completions_text, completions_tokens
        else:    
        
            if self.mode == 'no':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                )

            elif self.mode == 'old':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    temperature=self.sampling_temp
                )

            elif self.mode == 'gpt':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    top_p=0.9
                )

            elif self.mode == 'v2':
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    temperature=self.sampling_temp
                )

            # remove the attached input from output for some model
            scores = outputs.scores
            output_ids = outputs.sequences[0, -len(scores):]

            # compute logprob for each token
            completions_tokens = []
            completions_logprob = 0

            for score, token in zip(scores, output_ids, strict=True):
                logprobs = F.log_softmax(score[0], dim=-1)
                logprob = logprobs[token].item()
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                    'logprob': logprob,
                })
                completions_logprob += logprob
            
            completions_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            
            return completions_text, completions_tokens