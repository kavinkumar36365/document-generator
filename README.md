# document-generator

## Getting Started

To start working with the scripts in this workspace, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies.
    pytorch v2.2.0

    
3. Run `train.py` with the data on which the model needs to be trained on saved as `input.txt` and present in the current workspace. This script trains the model based on the input data.

4. After the model is trained, run `generate.py` to generate the document based on the training data. This script uses the trained model to generate the document.

## Files in the Repository

- `train.py`: This script is used to train the model based on the input data.
- `generate.py`: This script is used to generate the document using the trained model.
- `tokenizer.py`: This script provides functions for tokenizing text data.
- `model.py`: This script contains the implementation of the document generation model.

