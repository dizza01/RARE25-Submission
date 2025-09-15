# RARE25 Baseline Submission

This repository provides a baseline solution for the RARE25 challenge, including code for data import, model training, inference, and Docker-based submission.

## Overview
- **Model:** 2D CNN (ResNet50) for image classification
- **Framework:** PyTorch
- **Containerization:** Docker
- **Submission:** Grand Challenge platform

## Workflow

### 1. Importing Data
- Training and test data are organized in the `test/input/` directory.
- Example input images and metadata are provided for testing the pipeline.

### 2. Training the Model
- Training scripts (not included in this baseline) are used to train a ResNet50 model on the public training data.
- The trained model weights are saved as `resources/resnet50.pth`.
- If you wish to retrain, use your own scripts and replace the weights file.

### 3. Inference
- The main inference logic is in `inference.py`.
- The model is loaded and run on test images from the `/input` directory.
- Outputs are written to the `/output` directory as required by the challenge.

### 4. Docker Container
- The repository includes a `Dockerfile` to build a containerized version of the algorithm.
- Scripts `do_build.sh`, `do_test_run.sh`, and `do_save.sh` automate building, testing, and exporting the Docker image.

#### Build the Docker Image
```sh
./do_build.sh
```

#### Test the Container Locally
```sh
./do_test_run.sh
```

#### Export the Docker Image for Submission
```sh
./do_save.sh
```
- This creates a `.tar.gz` file ready for upload to the Grand Challenge platform.

## Notes
- The container is designed to run as a non-root user, with no network access, and with all dependencies included.
- Model weights must be present as real files (not Git LFS pointers) before building the Docker image.
- For local testing on Apple Silicon (M1/M2), the container runs in CPU mode.

## Customization
- To improve the baseline, you can modify the model, training procedure, or inference logic.
- Ensure any new dependencies are added to `requirements.txt` and the Dockerfile.


