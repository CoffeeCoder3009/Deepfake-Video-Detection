# Deepfake-Video-Detection

This repository contains the implementation of a DeepFake detection system based on spatial and temporal feature extraction. The project uses state-of-the-art machine learning models to identify DeepFake videos by analyzing irregularities in facial identity features across frames.

## Overview

DeepFake videos are artificially generated videos that replace a person's face with another's, often leading to misinformation or misuse. Detecting these videos has become a critical task in multimedia forensics.

Our approach builds on the work described in the paper ["It Wasn’t Me: Irregular Identity in Deepfake Videos"](https://github.com/HongguLiu/Identity-Inconsistency-DeepFake-Detection) by Honggu Liu et al., ICIP 2023. It incorporates spatial and temporal analysis to classify videos as real or fake.

## Methodology and Model Architecture

1. **Face Detection and Extraction**:  
   Faces are detected and cropped from every 25th frame of the video using RetinaFace.

2. **Spatial Features**:  
   - XceptionNet is used to extract embeddings for individual frames.  
   - Features are averaged across frames for robust spatial representation.

3. **Temporal Features**:  
   - Frame embeddings are passed through an LSTM model to capture temporal inconsistencies.  

4. **Classification**:  
   - Spatial and temporal embeddings are combined and passed through a Multi-Layer Perceptron (MLP) to classify videos as "Real" or "Fake".

![Model Architecture](https://drive.google.com/uc?id=1kS2ebxH__wYWyzfwYkbMwigtN3U4Kw-w)

## Project Structure
- **Temporal Branch:** Contains temporal and resnet embeddings
- **Spatial Branch:** Contains spatial embeddings
- **Papers:** Research papers and related resources.
- **Models:** Trained models used in this project.
- **jupyter_notebooks:** Jupyter notebooks with code and explanations.
- **Dataset:** Full dataset used for training and testing.
- **combined_embeddings:** Combined embeddings from different branches.
- **Check:** Any additional files for verification.

## Google Drive Link
All project resources can be accessed via this [Google Drive folder](https://drive.google.com/drive/folders/1lEL5rbFGgqa0Y9EvAMMIV1cj5huHG3bI?usp=sharing).

## Results

The model achieves **84% classification accuracy** by integrating spatial and temporal features. This high performance is a result of the combined strength of XceptionNet for spatial analysis and LSTMs for temporal feature extraction.

## Features

- **Explainability**: The use of semantic identity features makes the detection interpretable.  
- **Robustness**: The method works well with low-quality DeepFake videos.  
- **Efficiency**: Optimized for computational efficiency using multi-threading and multiprocessing.

## Installation

### Requirements
- Python 3.8+
- TensorFlow
- Keras
- OpenCV
- RetinaFace
- NumPy
- SciPy

## Authors
This project was a collaborative effort by the following team members:
- [**Purvanshi Nijhawan**](https://github.com/CoffeeCoder3009)
- [**Sohit Dhawan**](https://github.com/sohitdhawan)
