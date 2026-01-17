# Digit Classifier

A neural network project for training and classifying handwritten digits.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd digit_classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

First, train the neural network on the MNIST dataset:

```bash
python train.py
```

This will:
- Load the MNIST dataset
- Train the model
- Save the trained model to `digit_model.pth`

### Step 2: Draw and Predict

Once the model is trained, draw a digit and get predictions:

```bash
python draw_digit.py
```

This will:
- Open a GUI window where you can draw a digit with your mouse
- Click "Save" to save your drawing as `digit.png`
- Use `predict.py` to classify your drawn digit

## Files

- `train.py` - Train the neural network model
- `draw_digit.py` - GUI tool to draw digits
- `predict.py` - Predict digit from an image
- `model.py` - Neural network model architecture
- `utils.py` - Utility functions
- `requirements.txt` - Project dependencies
