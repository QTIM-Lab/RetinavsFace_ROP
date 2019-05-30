# RetinavsFace_ROP

A Retina vs Face Classifier for Retinal Images, especially for ROP

**Requirements**: 
- Python 3.6.x
- PyTorch 1.0.0
- CUDA 9.0
- GPU support

**How to Run**:
Run via command line (preferably in a docker with above requirements) using main.py file

`python main.py [action] [data_directory] [model_path]`

Actions: train, eval, predict

Data Directory: images with file structure of [train, val, test]/[faces, retinas]/[images]

Model Path: (path and name of where to save model) OR (path and name of trained model