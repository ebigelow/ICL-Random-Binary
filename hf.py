import gc
import os
import json
import math
from tqdm import trange, tqdm
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def init_model(model_id='mistralai/Mixtral-8x7B-v0.1', device_map='auto', 
               torch_dtype=torch.float32, quantization_config=None, 
               tokenizer_kwargs={}, model_kwargs={}):
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, torch_dtype=torch_dtype, 
        quantization_config=quantization_config, **model_kwargs)
    return tokenizer, model


def clear_cuda():
    """Clear memory from CUDA - assuming all relevant object references are deleted."""
    # https://discuss.pytorch.org/t/48879/27        tip: shorten forum links by removing the text part
    gc.collect()
    torch.cuda.empty_cache()

    ## https://discuss.huggingface.co/t/18310/2
    #import numba.cuda
    #n_gpus = torch.cuda.device_count()
    #for gpu in range(n_gpus):
    #    numba.cuda.select_device(gpu).reset()


def outputs_to_cpu(outputs):
    out_seqs = outputs['sequences'].to('cpu')
    out_scores = [s.to('cpu') for s in outputs['scores']]

    del outputs
    clear_cuda()
    return out_seqs, out_scores



# Convert tensors of token indexes to equal-shape numpy arrays of tokens
## TODO: this is very inefficient. Instead, only 1 call to batch_decode
##       should be made, and X should be flattened then reshaped to its original
##       form.  But there might be a problem if tokenization changes -- maybe need
##       to make sure each sequence ends with a <eos> end of sent. token
def decode_2d(X, tokenizer, **kwargs):
    return [tokenizer.batch_decode(x[..., None], **kwargs) for x in X]

def decode_3d(X, tokenizer, **kwargs):
    return np.array([decode_2d(X_, tokenizer, **kwargs) for X_ in X])

def decode_4d(X, tokenizer, **kwargs):
    return np.array([[decode_2d(X__, tokenizer, **kwargs) 
                      for X__ in X_] for X_ in X])



def get_probs(input_texts, tokenizer, model):
    ### solution from: https://discuss.huggingface.co/t/11710/9
    ###      -->       https://discuss.huggingface.co/t/30075/17

    input_ids = tokenizer(input_texts, padding=True, return_tensors='pt').input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    
    # drop last index, since probability at index 0 corresponds to the token at index 1
    return probs[:, :-1, :]
    

def inference(input_text, tokenizer, model, logprobs=True, echo_inputs=False,
              num_return_sequences=100, max_new_tokens=10, 
              do_sample=True, temperature=1.0, **kwargs):
    """

    Returns:
        out_seqs   (Tensor : [n_inputs, n_returns, seq_len]): Token indexes for decoded sequence
        out_scores (Tensor : [n_inputs, n_returns, seq_len, vocab_size]): Token probabilities
    """
    inputs = tokenizer(input_text, padding=True, return_tensors='pt')
    inputs = inputs.to('cuda')
    
    if logprobs:
        # Also see:
        #   https://discuss.huggingface.co/t/30075
        #   https://github.com/huggingface/transformers/pull/21191/files#diff-  ...
        #     ...  26783ca033d92b4ce2e01eb691dbf5b05dd972b0cbfdc69fc726cf77a9dcb011R1016
        kwargs.update({'return_dict_in_generate': True, 'output_scores': True})
    
    outputs = model.generate(
        **inputs, **kwargs,
        num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=temperature, pad_token_id=tokenizer.eos_token_id
    )
    out_seqs, out_scores = outputs_to_cpu(outputs)
    out_scores = torch.stack(out_scores)
    
    # reshape output scores to (n_inputs, n_returns, seq_len, vocab_size)
    n_inputs = 1 if type(input_text) is str else len(input_text)
    seq_len, _, vocab_size = out_scores.shape
    
    out_seqs = out_seqs.reshape(n_inputs, num_return_sequences, -1)
    if not echo_inputs:
        out_seqs = out_seqs[:, :, inputs.input_ids.shape[1]:]   # trim to remove input text

    out_scores = out_scores.transpose(0, 1).reshape(n_inputs, num_return_sequences, seq_len, vocab_size)
    
    if echo_inputs:
        in_probs = get_probs(input_text, tokenizer, model)
        in_probs = in_probs.unsqueeze(1).repeat((1, num_return_sequences, 1, 1))   # add dim to match outputs
        out_scores = torch.concat([out_scores, in_probs], dim=2)

    return out_seqs, out_scores


def multiple_inference(input_texts, tokenizer, model,
                       in_batch_size=100, out_batch_size=100, 
                       num_return_sequences=100,
                       save_path='./hf_out/', overwrite=False, verbose=True,
                       top_logprobs=0, extra_args=None, **kwargs):
    """
    Run inference multiple times, saving results to file each iteration.

    This is so that memory can be cleared and old objects can be deleted.
    """
    # Record inference args to file
    input_args = kwargs.copy()
    input_args['in_batch_size'] = in_batch_size
    input_args['top_logprobs'] = top_logprobs
    input_args.update(extra_args or {})
    json.dump(input_args, open(f'{save_path}input_args.json', 'w'))

    logprobs = (top_logprobs > 0)

    # Setup batching loop
    bsi = in_batch_size
    nbi = math.ceil(len(input_texts) / bsi)
    batches_in = trange(nbi) if verbose else range(nbi)

    bso = min(out_batch_size, num_return_sequences)
    nbo = math.ceil(num_return_sequences / bso)
    batches_out = trange(nbo) if verbose else range(nbo)

    for b_i in batches_in:
        batch_texts = input_texts[b_i*bsi : (b_i+1)*bsi]

        for b_o in batches_out:

            # Skip this batch if we already have the file
            if overwrite and os.path.isfile(f'{save_path}seqs_idx-{b_i}-{b_o}.pt'):
                print(f'\t--> skipping  {save_path}seqs_idx-{b_i}-{b_o}.pt')
                continue  

            # Run inference for this batch
            out_seqs, out_scores = inference(batch_texts, tokenizer, model, 
                logprobs=logprobs, num_return_sequences=bso, **kwargs)

            # Save to file
            json.dump(batch_texts, open(f'{save_path}inputs-{b_i}-{b_o}.json', 'w'))
            torch.save(out_seqs, f'{save_path}seqs_idx-{b_i}-{b_o}.pt')
            # np.save(f'{save_path}seqs_text-{b_i}-{b_o}.npy', decode_3d(out_seqs, tokenizer))

            if logprobs:
                out_topk = torch.topk(out_scores, k=top_logprobs, dim=-1)
                torch.save(out_topk, f'{save_path}probs_idx-{b_i}-{b_o}.pt')
                # np.save(f'{save_path}probs_text-{b_i}-{b_o}.npy', decode_4d(out_topk.indices, tokenizer))

            # Clear objects from memory
            clear_cuda()



# ================================================================================================
# copied code from batch_prompt

def format_prompts(prompt, prompt_args=None):
    prompts, prompt_args = listify_prompts(prompt, prompt_args)
    formatted_prompts = [p.format(**kwargs) for p, kwargs in zip(prompts, prompt_args)]
    return prompts, prompt_args, formatted_prompts


def listify_prompts(prompt, prompt_args=None):        
    """
    Format prompt str with prompt args, and return (<list of prompts>, <list of kwargs for each>)
    
    Args:
        prompt (str | list[str]) : can be one prompt (str) or multiple prompts (list[str])
        prompt (dict | list[dict]) : can be one kwarg dict or a list of kwarg dicts
    """
    # Prompt args
    prompt_args = prompt_args or {}   # default value is empty dict
    prompt_args = prompt_args if type(prompt_args) in (list, tuple) else [prompt_args]  # convert to list
        
    # Format prompt / prompts
    if type(prompt) is str:
        prompts = [prompt for kwargs in prompt_args]
    else:
        prompts = [p for p in prompt for kwargs in prompt_args]
        prompt_args = [kwargs for p in prompt for kwargs in prompt_args]
        
    return prompts, prompt_args




if __name__ == '__main__':
    test_text1 = 'Hello my name is'
    test_text2 = 'My favorite food is'

    tokenizer, model = init_model()

    out_seqs, out_scores = inference([test_text1, test_text2], tokenizer, model, 
                                     logprobs=True, num_return_sequences=9, echo_inputs=True)

    out_topk = torch.topk(out_scores, k=100, dim=dim).values


    # for o in out_seqs[0]:
    #     print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print(tokenizer.decode(o, skip_special_tokens=True))
