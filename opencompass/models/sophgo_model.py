import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Union
from .base import BaseModel, LMTemplateParser

from .huggingface_above_v4_33 import _convert_base_messages, _convert_chat_messages, _get_meta_template

class SophgoModel(BaseModel):

    def __init__(self,
                 model_path: str,
                 sg_tokenizer_path: str,
                 max_seq_len: int = 2048,
                 device: str = "cuda",
                 devid: str = '0',
                 sample_kwargs: dict = dict(),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 **kwargs):
        self.device = device
        self.devid = devid
        self.max_seq_len = max_seq_len
        self.sample_kwargs = sample_kwargs
        self.generation_kwargs = generation_kwargs
        self.template_parser = _get_meta_template(meta_template)
        self.load_model(model_path, sg_tokenizer_path)
        self.eos_token_id = None

    def load_model(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.device == "gpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto"
            ).eval()
            self.model.generation_config.do_sample = False
        elif self.device == "tpu":
            path_chat = os.getenv("CHAT_PATH")
            if path_chat:
                sys.path.append(path_chat)
                import chat
            else:
                raise EnvironmentError("Environment variable CHAT_PATH not set!")
            devices = [int(d) for d in self.devid.split(",")]
            self.model = chat.Qwen()
            # init params
            self.model.init(devices, model_path)
            for key, value in self.sample_kwargs.items():
                assert hasattr(self.model, key)
                setattr(self.model, key, value)
        else:
            raise NotImplementedError("Unrecognized device, please specify it in model configs")

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings."""
        tokens = self.tokenizer([prompt], truncation=False)['input_ids']
        return len(tokens[0])

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        messages = _convert_chat_messages(inputs)
        batch_size = len(messages)
        assert batch_size == 1

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )
        messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
        tokenize_kwargs['add_special_tokens'] = False
        tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        generation_kwargs = self.generation_kwargs.copy()
        if max_out_len is not None:
            generation_kwargs['max_new_tokens'] = max_out_len
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id

        if self.device == "gpu":
            tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
            outputs = self.model.generate(**tokens, **generation_kwargs)
            outputs = outputs[:, tokens['input_ids'].shape[1]:]
            decodeds = self.tokenizer.batch_decode(outputs)
        elif self.device == "tpu":
            decodeds = self.inference(tokens['input_ids'][0].tolist(), generation_kwargs)

        return decodeds

    def inference(self, tokens, generation_kwargs):
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        token = self.model.forward_first(tokens)

        while token != self.tokenizer.eos_token_id and self.model.token_length < self.model.SEQLEN and tok_num < generation_kwargs['max_new_tokens']:
            word = self.tokenizer.decode(token, skip_special_tokens=True)
            self.answer_token += [token]
            tok_num += 1
            token = self.model.forward_next()

        self.answer_cur = self.tokenizer.decode(self.answer_token)

        return [self.answer_cur]

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs."""
        # reset template_parser
        self.template_parser = LMTemplateParser()

        # only support gpu for now
        assert self.device == "gpu"
        pad_token_id = self.tokenizer.pad_token_id
        messages = _convert_base_messages(inputs)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )

        tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)[0]

        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none').view(batch_size, seq_len - 1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss
