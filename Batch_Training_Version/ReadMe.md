# Modified Version for STTN Model

**Several modifications have been made based on the code from https://github.com/wubin5/STTN**

A Pytorch version for paper 

[Spatial-Temporal Transformer Networks for Traffic Flow Forecasting]: https://arxiv.org/pdf/2001.02908.pdf



## Main Modification

- This version allows batch training to improve training efficiency and performance.
- Fix the bug in multi-head attention.
- Use *Chebyshev Polynomials* to model *Fixed Graph Convolution Layer*, which is the same as original paper.
- Provide positional embedding options with sine/cosine functions, which is the same as [Attention is All Your Need]: https://arxiv.org/abs/1706.03762.
- Add positional embedding to all three components: key, query, value.



## How to Run?

### prepareData.py

Data preprocessing. Adapted from [ASTGCN]: https://github.com/guoshnBJTU/ASTGCN-r-pytorch. 

- graph_signal_matrix_filename: The path for raw data, shape: [L, N]. Each row represents a  record, N means there are N traffic sensors in total.
- This code will return and save a processed .npz file in the same path.
- Other parameters you may refer to the comments in python file.



### train_batch.py

Training process. Adapted from [ASTGCN]: https://github.com/guoshnBJTU/ASTGCN-r-pytorch. 

- adj_mx: The path for adjacency matrix.

- params_path: The path for saving model parameters and training log.

- filename: The data we got from prepareData.py. You may also use your own preprocess function. The shape of our dataloader should be [Batch_size(B), number_of_sensors(N), input_channels(C), input_length(T)]

- All the training process will be saved.

- You should write down the best epochs for further test.

  

### predict_batch.py

Testing process.

- Similar setting as train_batch.py.
- It will return and save the prediction results for test set.

