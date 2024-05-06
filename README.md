# document-generator

## Getting Started

To start working with the scripts in this workspace, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies.
    pytorch v2.2.0

    
3. Run `train.py` with the data on which the model needs to be trained on saved as `input.txt` and present in the current workspace. This script trains the model based on the input data.

4. After the model is trained, run `generate.py` to generate the document based on the training data. This script uses the trained model to generate the document.


Usage
-----

### 1. Data Preparation

Place your input text data in a file named `input.txt` in the root directory of the project.

### 2. Training

To train the model, run the following command:

```bash
python train.py
```
This script will:

-Load the input data from `input.txt`
-Tokenize the data using the naive_tokenizer class
-Split the data into training and validation sets
-Train the BigramLanguageModel on the training data
-Save the trained model weights to `weights.pth`

You can monitor the training progress and loss values printed to the console.

### 3. Generation

To generate new text using the trained model, run the following command:

```bash
python generate.py
```
This script will:

-Load the trained model weights from `weights.pth`
-Load the tokenizer and vocabulary
-Generate a new text sequence of  tokens based on the model's predictions
-Print the generated text to the console

### 4. Configuration

You can modify the hyperparameters of the model by editing the `hyperparameters.yaml` file. The available hyperparameters are:

`batch_size`: Batch size for training
`block_size`: Maximum sequence length
`max_iters`: Maximum number of training iterations
`eval_interval`: Interval (in iterations) for evaluating the model on the validation set
`learning_rate`: Learning rate for the optimizer
`eval_iters`: Number of iterations for evaluating the model
`n_embd`: Embedding dimension
`n_head`: Number of attention heads
`n_layer`: Number of Transformer blocks
`dropout`: Dropout rate

## File Structure

-`tokenizer.py`: Contains the naive_tokenizer class for tokenizing text data
-`model.py`: Contains the implementation of the BigramLanguageModel and its components (attention, feed-forward, and Transformer block)
-`train.py`: Script for training the model
-`generate.py`: Script for generating new text using the trained model
-`hyperparameters.yaml`: Configuration file for model hyperparameters
-`weights.pth`: File for storing the trained model weights (generated during training)