# Dog Breed Classification using CNN

## Project Overview
This project focuses on image classification using a Convolutional Neural Network (CNN) to identify dog breeds from images. Our dataset contains 150 dog breeds, and the goal is to train a deep learning model that can distinguish between visually similar breeds.

Examples of challenging breed pairs include:
- Siberian Husky vs. Alaskan Malamute
- French Bulldog vs. Pug

The project is implemented using **PyTorch**.

---

## Objectives
- Build a CNN-based image classification model
- Classify dog images into one of 150 breed categories
- Evaluate performance using validation and test datasets
- Analyze challenges such as similar-looking breeds and class imbalance

---

## Future Extensions
If time allows, we may extend the project to:
- Detect whether a dog is present in an image
- Predict multiple breeds in the same image
- Estimate additional attributes such as age or health condition

---

## Team Responsibilities

### Member 1 — Data Gathering and Preprocessing
- Collect and organize dataset
- Clean and verify image data
- Create train/validation/test splits
- Implement preprocessing and augmentation pipeline

### Member 2 — Model Architecture
- Research CNN architectures
- Implement the baseline CNN model in PyTorch
- Improve architecture using regularization and deeper layers

### Member 3 — Training and Evaluation
- Implement training and validation loop
- Evaluate model performance
- Generate plots, metrics, and final results

---

## Project Structure

```bash
dog-breed-classification/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── config.py
│
├── outputs/
│   ├── models/
│   ├── plots/
│   └── logs/
│
├── README.md
├── requirements.txt
└── .gitignore
