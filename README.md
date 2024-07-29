# pt-nlp-gpt

This repository is a result of great hands-on experience I had while hacking around Shakespeare GPT pretraining codes in [nanoGPT](https://github.com/karpathy/nanoGPT). Thanks to the original source, I was able to:
* implement forward computation logic of causal attention from scratch to comprehend what's going on inside any high-level libraries like [transformers](https://huggingface.co/docs/transformers/index)(implementation and corresponding illustration can be found in this [module](https://github.com/sunsikim/pt-nlp-gpt/blob/master/gpt/model.py) and [notebook](https://github.com/sunsikim/pt-nlp-gpt/blob/master/notebooks/illustrate_causal_attention.ipynb) respectively).
* modify original custom code for DDP implementation to use [accelerate](https://huggingface.co/docs/accelerate/index) library(since pretraining Shakespeare GPT would only require single GPU) and experience the philosophy of the library that aims to provide unified and seamless API usages for diverse specific hardware use-cases.

