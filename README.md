# Dog Breed Classification with Pretrained ResNet-18

## What This Project Does

This project classifies dog breed images using PyTorch. The data pipeline loads dog images from breed folders, applies ImageNet-style preprocessing, and trains a pretrained CNN.

Important: the files `dataset.py`, `data_loader.py`, `transforms.py`, and `verify_pipeline.py` are not the pretrained model. They are the data pipeline. The pretrained model is created in `src/model.py` by loading ResNet-18 from `torchvision`.

## Why Use a Pretrained Model?

Training a CNN from scratch needs a lot of data and time. A pretrained ResNet-18 already learned useful visual features from ImageNet, such as edges, textures, fur patterns, ears, snouts, and body shapes. We replace its final classification layer so it predicts our dog breed classes instead.

This technique is called transfer learning.

## Project Structure

```text
deep-learning-team8/
  Images/                  # raw breed folders, optional
  data/
    train/
    val/
    test/
  src/
    dataset.py
    data_loader.py
    transforms.py
    verify_pipeline.py
    model.py
    train_pretrained.py
    evaluate.py
    predict.py
    scripts/
      download_and_split.py
  outputs/
    models/
    reports/
    plots/
```

## Run Locally

From the project folder:

```powershell
cd C:\Users\rn977\OneDrive\Desktop\demo.dog\deep-learning-team8
```

Create the train/validation/test split from `Images/`:

```powershell
python src\scripts\download_and_split.py --source-dir Images --data-dir data
```

Verify the data pipeline:

```powershell
python src\verify_pipeline.py
```

Run a small smoke test:

```powershell
python src\train_pretrained.py --data-dir data --epochs 1 --batch-size 4 --max-batches 2
```

Train the pretrained model:

```powershell
python src\train_pretrained.py --data-dir data --epochs 10 --batch-size 32 --lr 0.0001 --num-workers 2
```

Evaluate:

```powershell
python src\evaluate.py --checkpoint outputs\models\best_resnet18.pth --data-dir data --split test
```

Predict one image:

```powershell
python src\predict.py --checkpoint outputs\models\best_resnet18.pth --image data\test\n02085620-Chihuahua\n02085620_10131.jpg --top-k 5
```

## Run In Google Colab

Upload this folder to Google Drive, then create a Colab notebook.

Use GPU:

```text
Runtime -> Change runtime type -> T4 GPU
```

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Go to your project:

```python
%cd /content/drive/MyDrive/deep-learning-team8
```

Install dependencies:

```python
!pip install -q pillow matplotlib scikit-learn tqdm pandas
```

If you uploaded `Images/` but not `data/`, split the dataset:

```python
!python src/scripts/download_and_split.py --source-dir Images --data-dir data
```

Verify:

```python
!python src/verify_pipeline.py
```

Train with pretrained ResNet-18:

```python
!python src/train_pretrained.py --data-dir data --epochs 10 --batch-size 32 --lr 0.0001 --num-workers 2
```

Evaluate:

```python
!python src/evaluate.py --checkpoint outputs/models/best_resnet18.pth --data-dir data --split test --batch-size 32 --num-workers 2
```

##  Report

> We used transfer learning with a pretrained ResNet-18 CNN. ResNet-18 was first trained on ImageNet, so it already learned general image features. We replaced the final fully connected layer with a new layer matching the number of dog breeds in our dataset, then fine-tuned the model on our train split.

Also mention:

- `train` is used to learn model weights.
- `val` is used to choose the best checkpoint.
- `test` is used for final evaluation.
- Top-1 accuracy checks the highest prediction.
- Top-5 accuracy checks whether the correct breed appears in the five strongest predictions.
- The confusion matrix helps identify similar-looking breed mistakes.

 ## run this for setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
