# :briefcase: Learning TSP Requires Rethinking Generalization

This repository contains code for the paper [**"Learning TSP Requires Rethinking Generalization"**](https://arxiv.org/abs/2006.07054) by Chaitanya K. Joshi, Quentin Cappart, Louis-Martin Rousseau, Thomas Laurent and Xavier Bresson.

## :newspaper: Overview

- End-to-end training of neural network solvers for combinatorial problems such as the **Travelling Salesman Problem** is intractable and inefficient beyond a few hundreds of nodes. 
While state-of-the-art Machine Learning approaches perform closely to classical solvers for trivially small sizes, they are **unable to generalize** the learnt policy to larger instances of practical scales.
- Towards leveraging transfer learning to **solve large-scale TSPs**, this paper identifies inductive biases, model architectures and learning algorithms that promote generalization to instances larger than those seen in training. 
Our controlled experiments provide the first principled investigation into such **zero-shot generalization**, revealing that extrapolating beyond training data requires rethinking the entire neural combinatorial optimization pipeline, from network layers and learning paradigms to evaluation protocols.

## :rocket: End-to-end Neural Combinatorial Optimization Pipeline

Towards a controlled study of **neural combinatorial optimization**, we unify several state-of-the-art architectures and learning paradigms into one experimental pipeline and provide the first principled investigation on zero-shot generalization to large instances.

![End-to-end neural combinatorial optimization pipeline](/img/pipeline.png)

1. **Problem Definition:** The combinatorial problem is formulated via a graph.
2. **Graph Embedding:** Embeddings for each graph node areobtained using a Graph Neural Network encoder.
3. **Solution Decoding:** Probabilities are assigned to each node for belonging to the solution set, either independent of one-another (i.e. Non-autoregressive decoding) or conditionally through graph traversal (i.e. Autoregressive decoding).
4. **Solution Search:** The predicted probabilities are converted intodiscrete decisions through classical graph search techniques such as greedy search or beam search.
5. **Policy Learning:** The entire model in trained end-to-end via imitating anoptimal solver (i.e. supervised learning) or through minimizing a cost function (i.e. reinforcement learning).

## :bulb: Constributions

Our findings suggest that learning scale-invariant TSP solvers requires rethinking the status quo of neural combinatorial optimization to **explicitly account for generalization**:
- The prevalent evaluation paradigm overshadows models' **poor generalization capabilities** by measuring performance on fixed or trivially small TSP sizes.
- Generalization performance of GNN aggregation functions and normalization schemes benefits from explicit redesigns which account for **shifting graph distributions**, and can be further boosted by **enforcing regularities** such as constant graph diameters when defining problems using graphs.
- Autoregressive decoding enforces a **sequential inductive bias** which improves generalization over non-autoregressive models, but is costly in terms of inference time.
- Models trained with supervision are more amenable to post-hoc search, while **reinforcement learning** approaches scale better with more computation as they do not rely on labelled data.
    
**We open-source our framework and datasets to encourage the community to go beyond evaluating performance on fixed TSP sizes, develop more expressive and scale-invariant GNNs, as well as study transfer learning for combinatorial problems.**

## :open_file_folder: Installation
We ran our code on Ubuntu 16.04, using Python 3.6.7, PyTorch 1.2.0 and CUDA 10.0. 
We highly recommend installation via Anaconda.

```sh
# Clone the repository. 
git clone https://github.com/chaitjo/learning-tsp.git
cd learning-tsp

# Set up a new conda environment and activate it.
conda create -n tsp python=3.6.7
source activate tsp

# Install all dependencies and Jupyter Lab (for using notebooks).
conda install pytorch=1.2.0 cudatoolkit=10.0 -c pytorch  
conda install numpy scipy cython tqdm scikit-learn matplotlib seaborn tensorboard pandas
conda install jupyterlab -c conda-forge
pip install tensorboard_logger

# Download datasets and unpack to the /data/tsp directory.
pip install gdown
gdown https://drive.google.com/uc?id=152mpCze-v4d0m9kdsCeVkLdHFkjeDeF5
tar -xvzf tsp-data.tar.gz ./data/tsp/
```


## :zap: Usage

For reproducing experiments, we provide a set of scripts for training, finetuning and evaluation in the `/scripts` directory. 
Pre-trained models for some experiments described in the paper can be found in the `/pretrained` directory.

Refer to `options.py` for descriptions of each option. 
High-level commands are as follows:
```sh
# Training
CUDA_VISIBLE_DEVICES=<available-gpu-ids> python run.py 
    --problem <tsp/tspsl> 
    --model <attention/nar> 
    --encoder <gnn/gat/mlp> 
    --baseline <rollout/critic> 
    --min_size <20/50/100> 
    --max_size <50/100/200>
    --batch_size 128 
    --train_dataset data/tsp/tsp<20/50/100/20-50>_train_concorde.txt 
    --val_datasets data/tsp/tsp20_val_concorde.txt data/tsp/tsp50_val_concorde.txt data/tsp/tsp100_val_concorde.txt
    --lr_model 1e-4
    --run_name <custom_run_name>
    
# Evaluation
CUDA_VISIBLE_DEVICES=<available-gpu-ids> python eval.py data/tsp/tsp10-200_concorde.txt
    --model outputs/<custom_run_name>_<datetime>/
    --decode_strategy <greedy/sample/bs> 
    --eval_batch_size <128/1/16>
    --width <1/128/1280>
```


## Resources

- [ArXiv paper](https://arxiv.org/abs/2006.07054)
- [Blog post on neural combinatorial optimization](http://chaitjo.github.io/neural-combinatorial-optimization/)
- [TSP datasets generated with Concorde](https://drive.google.com/uc?id=152mpCze-v4d0m9kdsCeVkLdHFkjeDeF5)


## :scroll: Citation
```
@article{joshi2020learning,
  title={Learning TSP Requires Rethinking Generalization},
  author={Joshi, Chaitanya K and Cappart, Quentin and Rousseau, Louis-Martin and Laurent, Thomas and Bresson, Xavier},
  journal={arXiv preprint arXiv:2006.07054},
  year={2020}
}
```

**Acknowledgement and Related Work:** Our codebase is a modified clone of [Wouter Kool's excellent repository](https://github.com/wouterkool/attention-learn-to-route) for the paper ["Attention, Learn to Solve Routing Problems!"](https://openreview.net/forum?id=ByxBFsRqYm), and incorporates ideas from the following papers, among others:
- [W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.](https://openreview.net/forum?id=ByxBFsRqYm)
- [M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.](https://link.springer.com/chapter/10.1007/978-3-319-93031-2_12)
- [C. K. Joshi, T. Laurent, and X. Bresson. An efficient graph convolutional network technique for the travelling salesman problem. arXiv preprint arXiv:1906.01227, 2019.](https://arxiv.org/abs/1906.01227)
- [A. Nowak, S. Villar, A. S. Bandeira, and J. Bruna. A note on learning algorithms for quadratic assignment with graph neural networks. arXiv preprint arXiv:1706.07450, 2017.](https://arxiv.org/abs/1706.07450v1)
- [I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio. Neural combinatorial optimization with reinforcement learning. In International Conference on Learning Representations, 2017.](https://arxiv.org/abs/1611.09940)
