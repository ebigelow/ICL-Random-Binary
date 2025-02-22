import itertools
import gzip
import pickle
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats
import scipy.optimize

import pomegranate
import networkx as nx
from networkx.drawing.nx_pydot import write_dot, graphviz_layout
from networkx.drawing.nx_agraph import pygraphviz_layout
import Levenshtein

import pygraphviz
from IPython.display import Image, display


source_coin = 'a weighted coin, with {}% probability of Heads and {}% probability of Tails'
source_fair = 'a fair coin, with 50% probability of Heads and 50% probability of Tails'
source_non = 'a non-random algorithmic process'

# Prompt templates
p_gen1 = """Q: Generate a sequence of 1000 random samples from {source}.

A: [{flips}"""
p_gen2 = """Generate a sequence of 1000 samples from {source}:

[{flips}"""

p_judge = """Q: Is the following sequence of coin flips generated by a random process with no pattern, or are they generated by a non-random algorithm? [{flips}]

A: The sequence was generated by a"""


# ---------------------------------------------------------------------------

def enum_01_flips(n):
    return [flips for i in range(n+1)
                  for flips in itertools.product([0, 1], repeat=i)]


def enum_flips(n):
    return [''] + [
        ', '.join(flips) + ',' for i in range(1, n+1)
         for flips in itertools.product(['Heads', 'Tails'], repeat=i)]


def gen_alt_flips(n, repeat_seq=['Heads', 'Tails']):
    flip_gen = (repeat_seq * 1000)
    s = str(flip_gen[:n]).replace("'", '')[1:-1]
    if not s:
        return s
    return s + ','


# ---------------------------------------------------------------------------

def parse_token_probs(r, chars=('h', 't'), print_misses=False):
    """Parse log probabilities into normalized 2-class probabilities."""
    h, t = chars
    
    P = []
    for d in r['choice']['logprobs']['top_logprobs']:
        heads = sum([np.exp(v) for k, v in d.items() if k.strip() and k.strip().lower()[0] == h])
        tails = sum([np.exp(v) for k, v in d.items() if k.strip() and k.strip().lower()[0] == t])

        misses = {k: v for k, v in d.items() if (not k.strip()) or k.strip().lower()[0] not in (h, t)}
        if print_misses:
            print('Missed tokens in `get_token_prob`:', misses)
        
        p = tails / (heads + tails)  if heads+tails>0  else np.nan
        P.append(p)
        
    return np.array(P)


def parse_flips(r, sep=',', chars=('h', 't'), print_misses=False):
    """Parse a model completion which is a list of flips into a list of 0|1."""
    c = r['choice']
    text = c['text'] if 'text' in r['choice'].keys() else c['message']['content']   # parse Completion or ChatCompletion
    flips = text.split(sep) if sep else [c for c in text]
    return parse_flip_list(flips, chars, print_misses)


# backward compatibilty    TODO: rename these uses
def get_prob_rand(r):
    return get_token_prob(r, ('n', 'r'))

get_token_prob = lambda *args, **kwargs: parse_token_probs(*args, **kwargs)[0]
res_gen_to_flips = lambda *args: parse_flips(*args).tolist()

# ---------------------------------------------------------------------------


def island_cumsum_vectorized(a):    
    # https://stackoverflow.com/q/42129021
    a = a[~np.isnan(a)]
    
    a_ext = np.concatenate(( [0], a, [0] ))
    idx = np.flatnonzero(a_ext[1:] != a_ext[:-1])
    a_ext[1:][idx[1::2]] = idx[::2] - idx[1::2]
    return a_ext.cumsum()[1:-1]

def island_cumsum(a):
    """
    Return a vector of incremental values for each successive flip of the same value.
    
    Used to compute maximum run length
    E.g. [0, 1, 1, 1, 0, 0, 0] -> [1, 1, 2, 3, 1, 2, 3]
    
    This helper function does max with flipping the 0's and 1's since the above function only counts 
      non-zero values, and I haven't bothered to modify the stackoverflow code
    """
    a = np.array(a, dtype=float)
    return np.max([island_cumsum_vectorized(a), island_cumsum_vectorized(1-a)])


def lev_dist_mat(flips_list):
    # Compute matrix of Levenshtein distances normalized to [0,1] for each pair of flip seqs.
    n_samples = len(flips_list)
    dists = np.zeros((n_samples, n_samples))

    for i, r in enumerate(flips_list):
        for j, r_ in enumerate(flips_list):
            dists[i, j] = Levenshtein.ratio(r, r_)
            
    return dists
        
def compute_running_prob(flips, init_val=None):
    if init_val is not None:
        flips = np.array([init_val] + list(flips))
            
    N = len(flips)
    run_avg = np.cumsum(flips, dtype=float) / np.arange(1, N+1)
    return run_avg


# ---------------------------------------------------------------------------

def get_rotations(tup, depth):
    tups = []
    for i in range(len(tup)):
        t = tup[i:] + tup[:i]
        t = (t * depth)[:depth]
        tups.append(t)
        
    return tups


def get_path_probs(res, depth):
    # Map from seqs to p(next-token = heads)
    seq_probs = {k: parse_token_probs(r)[0] for k, r in zip(enum_01_flips(depth-1), res)}
    
    # Dict mapping from sequences to the next-token prob for (x_t | x_1 , ... , x_t-1)
    dd      = {k + (0,): 1 - p     for k, p in seq_probs.items()}
    dd.update({k + (1,): p         for k, p in seq_probs.items()})
    
    # Compute path probabilities for each full path
    dd2 = {k:v for k,v in dd.items() if len(k) == depth}
    for k, p in dd.items():
        if len(k) < depth:
            for k_ in dd2:
                if k_[:len(k)] == k:
                    dd2[k_] *= p
    return dd2

def get_paths_p(res, true_path, depth=3):
    """
    Compute probabilities of true path in results for a probability transition tree.
    
    """
    if 2**depth > len(res) + 1:
        raise ValueError(f'input depth {depth} too large')
    
    dd2 =  get_path_probs(res, depth)

    # Enumerate all paths
    paths = get_rotations(true_path, depth)
    return sum(v for k, v in dd2.items() if k in paths)

# ---------------------------------------------------------------------------

def mc_param_to_dict(param):
    # Convert Pomegranate MarkovChain distribution to dict of state transition weights
    dist_dict = defaultdict(dict)
    for p in param:
        from_elems = tuple(p[:-2])
        to_elem = p[-2]
        prob = p[-1]
        dist_dict[from_elems][to_elem] = prob
    
    return dict(dist_dict)

# ---------------------------------------------------------------------------

def build_flip_network(result, res_to_prob, res_labels):
    """
    Convert prob tree results to a networkx weighted graph.
    
    TODO: maybe delete this if it's not used
    """
    G = nx.DiGraph()

    G.graph["graph"] = dict(rankdir="LR")
    flips_to_str = lambda ls: (''.join([str(i) for i in ls])) if ls else '[]'

    seq_probs = {k: res_to_prob(r) for k, r in zip(res_labels, result)}

    for flips in res_labels:
        p_heads = seq_probs[flips]
        fs = '' if flips_to_str(flips) == '[]' else flips_to_str(flips)
        G.add_edge(flips_to_str(flips), fs + '0', weight=p_heads, color='pink')
        G.add_edge(flips_to_str(flips), fs + '1', weight=(1-p_heads), color='lightblue')
        
    return G

# ---------------------------------------------------------------------------

def flips_to_mat(flip_vals):
    max_len = max(len(fv) for fv in flip_vals)
    
    m = []
    for fv in flip_vals:
        fv = np.array(fv, dtype=float)
        fv_ = np.pad(fv, (0, max_len - len(fv)), constant_values=np.nan)
        m.append(fv_)
                     
    return np.array(m)


def moving_average(X, n=3):
    X_ = X.copy()
    
    for i in range(1, X.shape[1]):
        i_ = max(i - n, 0)
        X_[:, i] = X[:, i_:i].mean(axis=1)
    
    return X_

##ddd = np.tile(np.arange(10)[None, ...], (3, 2))
##moving_average(ddd)

# ---------------------------------------------------------------------------

def res_to_flips(r, probs=False, print_misses=True):
    flips = parse_flips(r, print_misses=print_misses)
    context = r['context'] if 'context' in r else r['prompt_args']['flips']
    return {
        'context': parse_flip_list(context.split(',')),
        'flips': flips,                                        # 0/1 flips
        'avgs': compute_running_prob(flips),                   # running average of flips
        'probs': parse_token_probs(r) if probs else None  # token probabilities, if available
    }

def simulate_bernoulli(samples, seq_len, p):
    return (np.random.rand(samples, seq_len) < p).astype(float)

# ---------------------------------------------------------------------------

def parse_flip_list(flips, chars=('h', 't'), print_misses=False, 
                    start_tok='[', end_tok=']'):
    flips = [f.strip().replace(start_tok, '') for f in flips]   
    flips = [f for f in flips if f]  # drop empty strings

    end_toks = [i for i, f in enumerate(flips) if end_tok in f]
    if len(end_toks) > 0:
        i = end_toks[0]
        # if print_misses:
        #     print(f'Cropping {len(flips)} to length {i}')
        flips = flips[:i]
    
    flips_ = []
    misses = []
    
    for f in flips:
        if f.lower()[0] == chars[0]:
            flips_.append(0)
        elif f.lower()[0] == chars[1]:
            flips_.append(1)
        else:
            misses.append(f)
            
    # if print_misses:
    #     print(misses)
    
    return np.array(flips_)

# ---------------------------------------------------------------------------

def flips_to_probs(flip_seqs):
    probs = defaultdict(lambda: [0, 0])
    
    for flips in flip_seqs:
        prefix = tuple(flips)[:-1]
        flip = flips[-1]
        probs[prefix][flip] += 1
        
    return {p[0] / (p[0] + p[1]) for k, p in probs.items()}

def parse_context(r, sep=',', chars=('h', 't'), print_misses=False):
    text = r['choice']['message']['content']   # parse Completion or ChatCompletion
    context = r['context']

    split = lambda s: s.split(sep) if sep else [c for c in s]
    context = parse_flip_list(split(context), chars, print_misses)
    responses = parse_flip_list(split(text), chars, print_misses)
    return context, responses

def chat_res_to_probs(results, sep=',', chars=('h', 't'), print_misses=False):
    flip_seqs = []
    
    for r in results:
        context, response = parse_context(r, sep, chars, print_misses)
        flip_seqs.append(tuple(context) + tuple(response))

    return flips_to_probs(flip_seqs)

def comp_res_to_probs(results, sep=',', chars=('h', 't'), print_misses=False):
    res_probs = [res_to_flips(r, probs=True) for r in results]
    return {flips_to_str(rp['flips']): rp['probs'] for rp in res_probs}

# ---------------------------------------------------------------------------

def jitter(x, p=.2, log=False):
    if log:
        return x * (1 + np.random.normal(0, scale=p, size=x.shape))
    else:
        return x + np.random.normal(0, scale=p, size=x.shape)



def plot_cumulative_lines(res_flips, init_mean=.5, goal_p=.5, seq_len=50, figsize=(9, 5), title=None, alpha=0.1, show=True):
    plt.rcParams['figure.figsize'] = figsize

    for flips in res_flips:
        y = compute_running_prob(flips, init_mean)
        #y *= (1 + np.random.uniform(-.01, .01))    # jitter
        plt.plot(range(len(y)), y, alpha=alpha, linewidth=4, color='blue')

    plt.ylim(0, 1)
    plt.hlines(goal_p, 0, seq_len, linestyles='--', color='red', linewidth=2)
    plt.hlines(0.5, 0, seq_len, linestyles='--', color='grey', linewidth=2)

    plt.xlabel('Output Index i for $y_i$')
    plt.ylabel('Running Avg. of Generated Flips')
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Convert raw LLM-generated flips to summary stats df for each sequence x
#   e.g. mean(x), gzip(x), ...

def flips_to_str(flips):
    return ''.join([str(int(flip)) for flip in flips])

def str_to_flips(s):
    return [int(c) for c in s]

def gzip_compress(flips):
    return gzip.compress(flips_to_str(flips).encode())

def make_flips_row(flips, llm_name, model_name, goal_p, include_flips=True, **kwargs):
    flips_arr = flips if type(flips) is np.ndarray else np.array(flips, dtype=int)
    flips_ls = flips_arr.tolist()

    run_avg = compute_running_prob(flips_ls, init_val=np.mean(flips_ls))
    row = {
        'llm': llm_name,
        'model': model_name,
        'goal_p': goal_p,
        
        'gzip': len(gzip_compress(flips_ls)),
        'mean': flips_arr.mean(),
        'p_A': np.mean(abs(flips_arr[:-1] - flips_arr[1:])),
        'long_run': island_cumsum(flips_ls),

        'run_avg_std': np.std(run_avg),
        'run_avg_max': np.max(run_avg),
        'run_avg_min': np.min(run_avg),
    }
    if include_flips:
        row['flips'] = flips_to_str(flips_ls)
        row['run_avg'] = run_avg
    row.update(kwargs)
    return row


def get_results(llm_name, p_tails, llm_flips, fit_models={}, seq_len=50, n_samples=None):
    # Crop sequences and prune seqs that are too short
    llm_flips = [flips[:seq_len] for flips in llm_flips if len(flips) >= seq_len]
    
    goal_p = p_tails / 100
    n_samples = n_samples or len(llm_flips)
    gzip_agg = len(gzip_compress([f for flips in llm_flips for f in flips.tolist()]))
    
    llm_rows = [make_flips_row(flips, llm_name, 'llm', goal_p, gzip_agg=gzip_agg) for flips in llm_flips]
    df = pd.DataFrame(llm_rows)
    subs = []

    for model_name, model_fn in fit_models.items():
        if model_name == 'Ground Truth':
            model = Bernoulli(p=goal_p)
        else:
            model = model_fn(llm_flips)
        sim_flips = [model.sample(seq_len) for _ in range(n_samples)]
        
        gzip_agg = len(gzip_compress([f for flips in sim_flips for f in list(flips)]))

        rows = [make_flips_row(flips, llm_name, model_name, goal_p, gzip_agg=gzip_agg) for flips in sim_flips]
        df = pd.concat([df, pd.DataFrame(rows)])

    return df

# ---------------------------------------------------------------------------
# Prepare sub-string results for LLM flips in a different data format

def enum_substrs(ls, window_size=10):
    subs = []
    
    for flips in ls:
        flips = ''.join(str(i) for i in flips)
        for i in range(0, len(flips)-window_size):
            sub = flips[i:i+window_size]
            subs.append(sub)
            
    return subs

def count_substrs(ls, window_size=10):
    sub_counts = defaultdict(lambda: 0)
    
    for flips in ls:
        flips = ''.join(str(i) for i in flips)
        for i in range(0, len(flips)-window_size):
            sub = flips[i:i+window_size]
            sub_counts[sub] += 1
            
    return dict(sub_counts)

def make_subs_dict(flips_list, llm_name, model_name, goal_p, window_sizes=[5, 10, 15, 20, 25], include_flips=True, **kwargs):
    d = {
        'llm': llm_name,
        'model': model_name,
        'goal_p': goal_p,
        
        'lev_dists': lev_dist_mat(flips_list),                              # n!
        'subs': {w: count_substrs(flips_list, w) for w in window_sizes},    # n * (|x| / k)
        'run_avg': [compute_running_prob(flips, init_val=np.mean(flips)) for flips in flips_list],
        'window_avg': {w: moving_average(flips_to_mat(flips_list), n=w) for w in window_sizes},
    }
    if include_flips:
        d['flips'] = [flips_to_str(flips) for flips in flips_list]
    d.update(kwargs)
    return d

def get_sub_results(llm_name, p_tails, llm_flips, fit_models, seq_len=50, n_samples=None, **kwargs):
    """Top-level method to get all substring results for a given (LLM, p_tails)."""

    # Crop sequences and prune seqs that are too short
    llm_flips = [flips[:seq_len].tolist() for flips in llm_flips if len(flips) >= seq_len]
    
    goal_p = p_tails / 100
    n_samples = n_samples or len(llm_flips)
    
    rows = [make_subs_dict(llm_flips, llm_name, 'llm', goal_p, **kwargs)]

    for model_name, model_fn in fit_models.items():
        if model_name == 'Ground Truth':
            model = Bernoulli(p=goal_p)
        else:
            model = model_fn(llm_flips)
        
        sim_flips = [model.sample(seq_len) for _ in range(n_samples)]
        row = make_subs_dict(sim_flips, llm_name, model_name, goal_p, **kwargs)
        rows.append(row)

    return rows



# ---------------------------------------------------------------------------
# Define Bernoulli and window average models used in paper

class Bernoulli:
    def __init__(self, X=None, p=None):
        self.p = p
        if X is not None:
            flips = [flip for x in X for flip in x]
            self.p = np.mean(flips)

    def sample(self, n_samples):
        s = np.random.rand(n_samples) < self.p
        return s.astype(int)

    def log_probability(self, x):
        n = len(x)
        k = sum(x)
        p = scipy.stats.binom(p=self.p, n=n).pmf(k=k)
        return np.log(p)


MIN_PROB = 1e-9

def predict_run_avg(x, p=.5, window_size=5):
    x_w = np.mean(x[-window_size:])
    v = p + (p - x_w)
    return max(0, min(1, v))
    

class WindowAvgModel:
    def __init__(self, goal_p=.5, window_size=5, warn=False):
        if (window_size < 1/goal_p) and warn:
            print('Warning: window size too small to accurately represent probability')
        self.goal_p = goal_p
        self.window_size = window_size
        
    def sample(self, seq_len):
        x = [self.goal_p]
        for _ in range(seq_len):
            p = predict_run_avg(x, self.goal_p, self.window_size)
            x.append(int(np.random.rand() < p))
        return x[1:]
    
    def predictive_prob(self, x):
        p = predict_run_avg(x, self.goal_p, self.window_size)
        p = p if x[-1] == 1 else 1-p   # invert probability for heads vs. tails
        return max(p, MIN_PROB)
    
    def log_probability(self, x):
        probs = [self.predictive_prob(x[:t]) for t in range(1, len(x)+1)]
        return sum(np.log(p) for p in probs)




# ---------------------------------------------------------------------------
# Fit Markov chain models

eps = 1e-3

def mc_combination(p_0, p_alt):
    """
    Combination of outcome bias (p_0) and gambler's fallacy (p_alt) Markov chain models.

    See Falk & Konald (1997) for a single-parameter gambler's fallacy MC model.
    """
    d1 = pomegranate.DiscreteDistribution({
        0: p_0, 
        1: 1 - p_0
    })
    c00 = (1 - p_alt) * p_0      
    c01 = p_alt       * (1 - p_0)
    c10 = p_alt       * p_0      
    c11 = (1 - p_alt) * (1 - p_0)
    d2 = pomegranate.ConditionalProbabilityTable([
        [0, 0, c00 / (c00 + c01)],   # Normalize probabilities
        [0, 1, c01 / (c00 + c01)],
        [1, 0, c10 / (c10 + c11)],
        [1, 1, c11 / (c10 + c11)]
    ], [d1])
    model = pomegranate.MarkovChain([d1, d2])
    return model


def mc_fn_simple(data):
    # Wrapper func to get neg sum of log probs using Markov Chain with specified args
    def f(args):
        p_0, p_alt = args
        model = mc_combination(p_0, p_alt)
        likelihood = np.sum([model.log_probability(x) for x in data])
        return -likelihood
    return f

def mc_fn(data, beta_param=5):
    """Wrapper func to get negative log posterior, with Beta prior and MC likelihood"""
    def f(args):
        p_0, p_alt = args
        model = mc_combination(p_0, p_alt)
        prior_p_0 = scipy.stats.beta.logpdf(p_0, beta_param, beta_param)
        prior_p_alt = scipy.stats.beta.logpdf(p_alt, beta_param, beta_param)
        
        # Convert to tensor on-the-fly to handle lists of x with varying length |x|
        likelihood = np.sum([model.log_probability(x) for x in data])
        posterior = likelihood + prior_p_0 + prior_p_alt
        return -posterior
    return f


# ---------------------------------------------------------------------------
# Visualize Markov chains


def draw(dot, prog='dot', img_format='png'):
    """Draw dot format input string with graphviz."""
    return Image(pygraphviz.AGraph(dot).draw(format=img_format, prog=prog))


def draw_MC(model, k=1, edge_min=.2, edge_mult=3, edge_pow=1, img_format='png'):
    dd = model.distributions[k].parameters[0]
    p = mc_param_to_dict(dd)

    if k == 1:
        g = f"""digraph top {{
           0 -> 0 [penwidth={edge_min + edge_mult*p[(0,)][0]**edge_pow}, label="0"];
           0 -> 1 [penwidth={edge_min + edge_mult*p[(0,)][1]**edge_pow}, label="1"];
           1 -> 0 [penwidth={edge_min + edge_mult*p[(1,)][0]**edge_pow}, label="0"];
           1 -> 1 [penwidth={edge_min + edge_mult*p[(1,)][1]**edge_pow}, label="1"];
        }}"""
        prog = 'dot'
    
    elif k == 2:
        g = f"""digraph top {{
            00 -> 00 [penwidth={edge_min + edge_mult*p[(0, 0)][0]**edge_pow}, label="0"];
            00 -> 01 [penwidth={edge_min + edge_mult*p[(0, 0)][1]**edge_pow}, label="1"];
            01 -> 10 [penwidth={edge_min + edge_mult*p[(0, 1)][0]**edge_pow}, label="0"];
            01 -> 11 [penwidth={edge_min + edge_mult*p[(0, 1)][1]**edge_pow}, label="1"];
            10 -> 00 [penwidth={edge_min + edge_mult*p[(1, 0)][0]**edge_pow}, label="0"];
            10 -> 01 [penwidth={edge_min + edge_mult*p[(1, 0)][1]**edge_pow}, label="1"];
            11 -> 10 [penwidth={edge_min + edge_mult*p[(1, 1)][0]**edge_pow}, label="0"];
            11 -> 11 [penwidth={edge_min + edge_mult*p[(1, 1)][1]**edge_pow}, label="1"];
        }}"""
        prog = 'circo'
        
    elif k == 3:
        # g = f"""digraph top {{
        #     000 [pos="0,0!"];
        #     100 [pos="-2,1!"];
        #     001 [pos="2,1!"];
        #     010 [pos="0,2!"];
        #     101 [pos="0,4!"];
        #     110 [pos="-2,5!"];
        #     011 [pos="2,5!"];
        #     111 [pos="0,6!"];

        #     000 -> 000 [penwidth={edge_min + edge_mult*p[(0, 0, 0)][0]**edge_pow}, label="0"];
        #     000 -> 001 [penwidth={edge_min + edge_mult*p[(0, 0, 0)][1]**edge_pow}, label="1"];
        #     001 -> 010 [penwidth={edge_min + edge_mult*p[(0, 0, 1)][0]**edge_pow}, label="0"];
        #     001 -> 011 [penwidth={edge_min + edge_mult*p[(0, 0, 1)][1]**edge_pow}, label="1"];
        #     010 -> 100 [penwidth={edge_min + edge_mult*p[(0, 1, 0)][0]**edge_pow}, label="0"];
        #     010 -> 101 [penwidth={edge_min + edge_mult*p[(0, 1, 0)][1]**edge_pow}, label="1"];
        #     011 -> 110 [penwidth={edge_min + edge_mult*p[(0, 1, 1)][0]**edge_pow}, label="0"];
        #     011 -> 111 [penwidth={edge_min + edge_mult*p[(0, 1, 1)][1]**edge_pow}, label="1"];
        #     100 -> 000 [penwidth={edge_min + edge_mult*p[(1, 0, 0)][0]**edge_pow}, label="0"];
        #     100 -> 001 [penwidth={edge_min + edge_mult*p[(1, 0, 0)][1]**edge_pow}, label="1"];
        #     101 -> 010 [penwidth={edge_min + edge_mult*p[(1, 0, 1)][0]**edge_pow}, label="0"];
        #     101 -> 011 [penwidth={edge_min + edge_mult*p[(1, 0, 1)][1]**edge_pow}, label="1"];
        #     110 -> 100 [penwidth={edge_min + edge_mult*p[(1, 1, 0)][0]**edge_pow}, label="0"];
        #     110 -> 101 [penwidth={edge_min + edge_mult*p[(1, 1, 0)][1]**edge_pow}, label="1"];
        #     111 -> 110 [penwidth={edge_min + edge_mult*p[(1, 1, 1)][0]**edge_pow}, label="0"];
        #     111 -> 111 [penwidth={edge_min + edge_mult*p[(1, 1, 1)][1]**edge_pow}, label="1"];
        # }}"""
        # prog = 'neato'

        g = f"""digraph top {{
            fontsize=30;
        
            000 [pos="0,0!",  fontsize=25];
            100 [pos="1,2!",  fontsize=25];
            001 [pos="1,-2!", fontsize=25];
            010 [pos="2,0!",  fontsize=25];
            101 [pos="4,0!",  fontsize=25];
            110 [pos="5,2!",  fontsize=25];
            011 [pos="5,-2!", fontsize=25];
            111 [pos="6,0!",  fontsize=25];

            000 -> 000 [penwidth={edge_min + edge_mult*p[(0, 0, 0)][0]**edge_pow}, label="0", fontsize=25];
            000 -> 001 [penwidth={edge_min + edge_mult*p[(0, 0, 0)][1]**edge_pow}, label="1", fontsize=25];
            001 -> 010 [penwidth={edge_min + edge_mult*p[(0, 0, 1)][0]**edge_pow}, label="0", fontsize=25];
            001 -> 011 [penwidth={edge_min + edge_mult*p[(0, 0, 1)][1]**edge_pow}, label="1", fontsize=25];
            010 -> 100 [penwidth={edge_min + edge_mult*p[(0, 1, 0)][0]**edge_pow}, label="0", fontsize=25];
            010 -> 101 [penwidth={edge_min + edge_mult*p[(0, 1, 0)][1]**edge_pow}, label="1", fontsize=25];
            011 -> 110 [penwidth={edge_min + edge_mult*p[(0, 1, 1)][0]**edge_pow}, label="0", fontsize=25];
            011 -> 111 [penwidth={edge_min + edge_mult*p[(0, 1, 1)][1]**edge_pow}, label="1", fontsize=25];
            100 -> 000 [penwidth={edge_min + edge_mult*p[(1, 0, 0)][0]**edge_pow}, label="0", fontsize=25];
            100 -> 001 [penwidth={edge_min + edge_mult*p[(1, 0, 0)][1]**edge_pow}, label="1", fontsize=25];
            101 -> 010 [penwidth={edge_min + edge_mult*p[(1, 0, 1)][0]**edge_pow}, label="0", fontsize=25];
            101 -> 011 [penwidth={edge_min + edge_mult*p[(1, 0, 1)][1]**edge_pow}, label="1", fontsize=25];
            110 -> 100 [penwidth={edge_min + edge_mult*p[(1, 1, 0)][0]**edge_pow}, label="0", fontsize=25];
            110 -> 101 [penwidth={edge_min + edge_mult*p[(1, 1, 0)][1]**edge_pow}, label="1", fontsize=25];
            111 -> 110 [penwidth={edge_min + edge_mult*p[(1, 1, 1)][0]**edge_pow}, label="0", fontsize=25];
            111 -> 111 [penwidth={edge_min + edge_mult*p[(1, 1, 1)][1]**edge_pow}, label="1", fontsize=25];
        }}"""
        prog = 'neato'
    else:
        raise NotImplementedError()
        
    return draw(g, prog=prog, img_format=img_format)


# ---------------------------------------------------------------------------
# Draw red/blue binary tree for formal language learning in generation task


def draw_flips(results, depth=4, fig_size=(7, 7), edge_scale=9, edge_min=.5, show=True):
    """
    Draw red/blue circle arrow transitions, given a list of results with a tree of probabilities.
    """
    ####seq_probs = {k: res_to_prob(r) for k, r in zip(res_labels, results)}
    G = nx.DiGraph()

    plt.rcParams['figure.figsize'] = fig_size
    G.graph["graph"] = dict(rankdir="LR")
    

    flip_probs = {k: parse_token_probs(r)[0] for k, r in zip(enum_01_flips(depth-1), results)}
    flips_to_str = lambda ls: (''.join([str(i) for i in ls])) if ls else '[]'


    for flips, p_tails in flip_probs.items():
        fs = flips_to_str(flips)
        n_from = fs
        n_to0 = (fs if fs != '[]' else '') + '0'
        n_to1 = (fs if fs != '[]' else '') + '1'
        G.add_edge(n_from, n_to0, penwidth=edge_min + edge_scale * (1 - p_tails), color='pink')
        G.add_edge(n_from, n_to1, penwidth=edge_min + edge_scale * p_tails, color='lightblue')
    
    weights = list(nx.get_edge_attributes(G, 'penwidth').values())
    colors = list(nx.get_edge_attributes(G, 'color').values())

    pos = pygraphviz_layout(G, prog='dot')

    # Draw the nodes
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_size=1,
        node_color='white',
    )
    # Draw node labels
    node_label_handles = nx.draw_networkx_labels(
        G, 
        pos=pos
    )
    # Draw the edges and store the returned FancyArrowPatch list
    arrows = nx.draw_networkx_edges(
        G,
        pos=pos,
        arrows=True,
        width=weights,
        edge_color=colors,
        arrowstyle='-|>',

        min_source_margin=12,
        min_target_margin=12,
    )

    for a, w in zip(arrows, weights):
        a.set_mutation_scale(5 + w/2)   # min width: 5    scaling factor: w/2
        a.set_joinstyle('miter')
        a.set_capstyle('butt')
    
    if show:
        plt.show()