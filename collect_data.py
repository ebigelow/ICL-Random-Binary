import argparse
import itertools
import math
import os
import pickle
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm, trange

import batch_prompt


def enum_01_flips(n):
    return [flips for i in range(n+1)
                  for flips in itertools.product([0, 1], repeat=i)]


def enum_flips(n):
    return [''] + [
        ', '.join(flips) + ',' for i in range(1, n+1)
         for flips in itertools.product(['Heads', 'Tails'], repeat=i)]


def gen_alt_flips(n, repeat_seq=['Heads', 'Tails']):
    flip_gen = list(repeat_seq) * 1000
    s = str(flip_gen[:n]).replace("'", '')[1:-1]
    if not s:
        return s
    return s + ','


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='flips_random')
    args = parser.parse_args()

    # p = """Q: Are the following coin flips from a random coin flip, or non-random coin flip? Why? [Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails]
    #
    # A: The flips are from a"""
    # res = call_openai(p, {'max_tokens': 1})
    n_calls = 1  #20
    n_samples = 5   #50
    max_tokens = 300

    # TOKEN_LIMIT = 9000
    # n_calls = math.ceil((n_samples * max_tokens) / TOKEN_LIMIT)
    p_gen1 = """Q: Generate a sequence of 1000 random samples {source}.\n\nA: [{flips}"""
    p_gen2 = """Q: Generate a sequence of 1000 samples {source}:\n\nA: [{flips}"""

    context_flips = 'Heads,'
    source_coin = 'from a weighted coin, with {}% probability of Heads and {}% probability of Tails'
    source_fair = 'from a fair coin, with 50% probability of Heads and 50% probability of Tails'
    source_unk = 'that may be from a fair coin with no correlation, or from some non-random algorithm'

    prompt_instruct = 'Generate a sequence of 1000 random samples {source}.'
    prompt_context = '[{flips}'

    now = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    os.makedirs(f'out/{now}', exist_ok=True)
    print(f'out dir: out/{now}')


    # model_names = ['gpt-4-0613', 
    #                'gpt-4-0314', 
    #                'gpt-3.5-turbo-0613', 
    #                'gpt-3.5-turbo-0301',
    #                'text-davinci-003', 
    #                'text-davinci-002', 
    #                'text-davinci-001', 
    #                'text-curie-001', 
    #                'text-babbage-001', 
    #                'text-ada-001']
    model_names = ['gpt-3.5-turbo-instruct']
    ### p_tails_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    ### p_tails_list = [15, 25, 35, 45, 55, 65, 75, 85]
    p_tails_list = [5, 10, 20, 30, 40, 49, 50, 51, 60, 70, 80, 90, 95]

    if args.type == 'flips_random':
        for m in model_names:
            is_chat_llm = m.startswith('gpt') and ('instruct' not in m)
            print('~'*100, '\n\t', m, f'\t\tchat: {is_chat_llm}')
            res = []

            for p_tails in tqdm(p_tails_list):
                p_heads = 100 - p_tails
                ### source_txt = source_coin.format(p_heads, p_tails) if p_heads != 50 else source_fair
                source_txt = source_coin.format(p_heads, p_tails)

                # ---------------------------------------------------------------------------
                for _ in trange(n_calls):
                    if m.startswith('gpt') and 'instruct' not in m:
                        res_ = batch_prompt.chat_completions(
                            prompt_instruct, 
                            prompt_context,
                            system_prompt='Your responses will only consist of comma-separated "Heads" and "Tails" samples.' + \
                                          '\nDo not repeat the user\'s messages in your responses.',
                            instruct_args={'source': source_txt},
                            context_args={'flips': context_flips},
                            model_args={'max_tokens': max_tokens, 'n': n_samples, 'model': m})
                    else:
                        res_ = batch_prompt.completions(
                            p_gen1, 
                            {'flips': context_flips,
                             'source': source_txt},
                            {'max_tokens': max_tokens, 'n': n_samples, 'model': m, 

                            'logprobs': 3})   # TODO

                    for r in res_:
                        r['p_tails'] = p_tails
                    res += res_
                    pickle.dump(res, open(f'out/{now}/gen_flips_{m}.pk', 'wb'))

            print(f'{len(res)} results')


    # concepts = [
    #     (0,), (1,), 
    #     (0, 1), (1, 0),
    #     (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)  
    #     (0, 1, 1, 0), (0, 0, 1, 0), (1, 1, 0, 1),
    #     (0, 1, 1, 1, 0), (1, 0, 0, 0, 1), (0, 0, 1, 0, 0), (1, 1, 0, 1, 1)
    # ]

    # concepts = [
    #     (0, 1), 
    #     (1, 0), 
    #     (0, 1, 0), 
    #     (1, 0, 1),
    #     (0, 1, 1, 0),
    #     (0, 1, 1, 1, 0),
    #     (1, 0, 1, 1, 0),
    # ]

    concepts = [
        # (0, 1),
        # (0, 1, 0),
        # (0, 1, 1),
        # (0, 0, 1, 0),
        # (1, 0, 0, 1),
        # (1, 1, 0, 1),
        (0, 0, 1),
        (1, 0, 1),
        (1, 0, 0),
        (1, 1, 0),
        # (0, 1, 0, 0, 0),
        # (0, 0, 0, 1, 1),
        # (1, 0, 1, 0, 0),
        # (0, 1, 1, 1, 0),
        # (1, 1, 0, 1, 0),
        # (1, 1, 1, 0, 1)
    ]
    flip_strs = ['Heads', 'Tails']
    
    n_log_probs = 5
    x_len_max = 40

    n_chat_samples = 30
    args_per_call = 50
    n_depth = 5  # 4

    source = source_unk

    model_names = [#'gpt-4-0613', 
                   # 'gpt-4-0314', 
                   ####'gpt-3.5-turbo-0613', 
                   # 'gpt-3.5-turbo-0301',
                   'gpt-3.5-turbo-instruct-0914', 
                   'text-davinci-003', 
                   # 'text-davinci-002', 
                   # 'text-davinci-001', 
                   # 'text-curie-001', 
                   # 'text-babbage-001', 
                   # 'text-ada-001'
                   ]

    if args.type == 'tree_formal':
        for m in tqdm(model_names):
            is_chat_llm = m.startswith('gpt') and ('instruct' not in m)
            print('~'*100, '\n\t', m, f'\t\tchat: {is_chat_llm}')
            res = []

            for concept in tqdm(concepts):
                concept_ht = [flip_strs[i] for i in concept]

                enum_dict = defaultdict(list)
                for x_len in range(1, x_len_max + 1):
                    for d_flips in enum_flips(n_depth):
                        flips = gen_alt_flips(x_len, concept_ht)
                        if flips[-1] != ' ' and (len(d_flips) > 0 and d_flips[0] != ' '):
                            flips = flips + ' '
                        flips = flips + d_flips
                        enum_dict[flips].append((x_len, d_flips))

                flips_set = list(enum_dict.keys())

                nf = len(flips_set)
                n_calls = math.ceil(nf / args_per_call)

                # ---------------------------------------------------------------------------
                for i in trange(n_calls):
                    flips_list = flips_set[i*args_per_call : (i+1)*args_per_call]
                    # import ipdb; ipdb.set_trace()


                    if is_chat_llm:
                        res_ = batch_prompt.chat_completions(
                            prompt_instruct, 
                            prompt_context,
                            system_prompt='Your responses will only consist of comma-separated "Heads" and "Tails" samples.' + \
                                          '\nDo not repeat the user\'s messages in your responses.',
                            instruct_args={'source': source},
                            context_args=[{'flips': flips} for flips in flips_list],
                            model_args={'max_tokens': 1, 'n': n_chat_samples, 'model': m})
                    else:
                        res_ = batch_prompt.completions(
                            p_gen2, 
                            [{'flips': flips, 'source': source} for flips in flips_list], 
                             {'max_tokens': 1, 'model': m, 'logprobs': n_log_probs})

                    for r in res_:
                        flips = r['context_args']['flips'] if is_chat_llm else r['prompt_args']['flips']
                        r['x_len'], r['depth'] = zip(*enum_dict[flips])
                        r['concept'] = concept
                    res += res_
                    pickle.dump(res, open(f'out/{now}/tree_formal_{m}.pk', 'wb'))

            print(f'{len(res)} results')



