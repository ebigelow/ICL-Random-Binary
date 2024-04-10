
# In-Context Learning Dynamics with Random Binary Sequences

This repo contains code + data for the ICLR 2024 paper [In-Context Learning Dynamics with Random Binary Sequences](https://arxiv.org/abs/2310.17639)

### Files
- `Analysis.ipynb` - **Main analysis notebook generating all plots in the paper for OpenAI models.**
- `Analysis-HuggingFace.ipynb` - Generate plots for open-source huggingface LLMs.
- `ModelSelection.ipynb` - Minimal example of Bayesian model selection described in Figure 2 and Appendix D.
- `DistractMMLU.ipynb` - MMLU "distraction" task used for Appendix E.
- `MinimalExample.ipynb` - Minimal example of going from prompt, to querying openAI, to generating plots with Randomness Judgment task.
- `DataPreProcessing.ipynb` - Records how I converted raw openAI query data to the pickle objects loaded in `Analysis.ipynb`. I didn't add imports and haven't tested this like the other notebooks.

- `utils.py` - Utilities for plotting and processing data, mostly used in `Analysis.ipynb`.
- `hf.py` - Code for running inference on HuggingFace transformers models, and extracting token logits.
- `collect_data.py` - Script for querying OpenAI with all data for randomness generation. Set arg `type` to `'flips_random'` for generation with varying $p(Tails)$, or to `tree_formal` for generation with formal concept learning.
- `collect_hf.py` - Script for running inference on huggingface models and collecting outputs.


### Data

Data is available for download at TODO.


### Requirements

See `requirements.txt` for python package requirements.

Querying OpenAI requires the [`batch_prompt`](https://github.com/ebigelow/batch-prompt) package. The plotting code depends on an earlier version of this, which can be installed by:

```
git clone git@github.com:ebigelow/batch-prompt.git    # clone repo
git checkout 15925fb                                  # checkout specific earlier version
ln -s ./batch_prompt ../batch_prompt                  # add a symbolic link so package can be imported to notebooks, etc.
```


### Citation

Bigelow, E. J., Lubana, E. S., Dick, R. P., Tanaka, H., & Ullman, T. D. (2023). In-Context Learning Dynamics with Random Binary Sequences. In the proceedings of the 12th International Conference on Learning Representations (ICLR).



### Contact

If you encounter problems, submit an issue on github or contact me (see paper for email).

