# Deep-Transition RNN Project

This repository builds upon the code from [https://github.com/yoonkim/lstm-char-cnn](https://github.com/yoonkim/lstm-char-cnn), with significant enhancements and added features.

## Features and Enhancements

The following features have been added to the original codebase:

1. **Deep-Transition RNN/LSTM Variants**:
   - Implements several possible variants of the **deep-transition RNN/LSTM** architecture.
   - Based on ideas from the paper:  
     *Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio*  
     *"How to Construct Deep Recurrent Neural Networks"*   
     [Read the paper on arXiv](https://arxiv.org/abs/1312.6026)

2. **Hierarchical RNN/LSTM**:
   - Adds support for **hierarchical RNN/LSTM** structures.

3. **Additional modifications**:
   - Includes convenient functions for preprocessing dataset texts.
   - Logs the number of guesses an RNN takes to predict the next word.

## Key Modifications
- **Primary Changes**:
  - The main logic has been significantly updated in the file `main.lua`.

- **New Files**:
  - Additional utility files have been added under the `/model` directory, specifically those with "dt" in their filenames:
    - `LSTMTDNNdt.lua`: Deep-transition version of LSTM and DNN.
    - `RNNdt.lua`: Deep-transition version of RNN.
    - `LSTMTDNNdtDec.lua`: Deep-transition version of an LSTM Decoder.
    - `LSTMTDNNdtSeq.lua`: Deep-transition version of an LSTM Seq2Seq Model.

## Prerequisites
To run this project, you’ll need the following:
- **Torch**: The code is written in Lua for the Torch framework. 
- **Dependencies**: Install the required Lua libraries using `luarocks`:
    ```bash
    luarocks install nngraph
    luarocks install luautf8
    ```
  GPU usage will require cutorch and cunn packages:
    ```bash
    luarocks install cutorch
    luarocks install cunn
    ```
- **Data**: Data should be put into the data/ directory, split into train.txt, valid.txt, and test.txt. The English Penn Treebank (PTB) data is given as the default.

## Usage
### Training
**Training large character-level model**
  Example:
  ```bash
  th main.lua -savefile char-small -rnn_size 300 -highway_layers 1 -kernels '{1,2,3,4,5,6}' -feature_maps '{25,50,75,100,125,150}' -EOS '+'
  ```
**Training Word-level model**
  Example:
  ```bash
  th main.lua -savefile word-small -word_vec_size 200 -highway_layers 0 -use_chars 0 -use_words 1 -rnn_size 200 -EOS '+'
  ```
**Training Deep Transition RNN**  
  For training Deep Transition RNN, include options -ndt (deep-transition depth, 1 corresponds to a common LSTM), -ndt2 (deep-transition depth for sequence 2 sequence transitions).  
  To enable hierarchical sequence 2 sequence processing, use option -seq3seq.  
  
### Evaluation
Evaluation on test can be performed via the following script:
  ```bash
  th evaluate.lua -model model_file.t7 -data_dir data/ptb -savefile model_results.t7
  ```

## License
This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more details.
