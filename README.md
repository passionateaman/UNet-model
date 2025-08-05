 Project Report: Colored Polygon Generation with Conditional UNet

## Project Overview

This project aimed to develop a deep learning model capable of generating colored polygon images based on an input grayscale polygon image and a specified color condition. The core of the solution is a modified U-Net architecture that incorporates the color information as a conditioning input. The model was trained on a custom dataset of polygon images and their corresponding colored versions. Experiment tracking was performed using Weights and Biases (wandb) to monitor training progress.

## Dataset

The dataset consists of pairs of input and output images, along with JSON annotations.
- Input Images: Grayscale images containing various polygon shapes.
- Output Images: The corresponding input polygons filled with a specific color.
- Annotations (`data.json`):A JSON file containing a list of dictionaries, where each dictionary links an input image filename (`input_polygon`) to its corresponding output image filename (`output_image`) and the target color (`colour`).

The dataset is split into training and validation sets, located in `/content/dataset/training` and `/content/dataset/validation` respectively. The `PolygonColorDataset` class was implemented to load this data, process the images, and convert the color names into one-hot encoded tensors for use as conditioning. It also handles matching input images to their annotations and corresponding output images based on the `data.json` file.

## Model Architecture: Conditional UNet

The model employed is a U-Net architecture enhanced with a conditioning mechanism to incorporate the desired color.

- **U-Net Base:** The model follows the standard U-Net structure with an encoder-decoder pathway and skip connections. The encoder progressively downsamples the input image, capturing hierarchical features, while the decoder upsamples the features to generate the output image.
- **Color Conditioning:** The color information, represented as a one-hot encoded vector, is embedded into a spatial map matching the input image dimensions using a fully connected layer. This color map is then concatenated with the input image channels before the first convolutional layer of the encoder. This allows the model to learn color-dependent features from the beginning of the network.
- **Input/Output Channels:** The input to the model is a 3-channel RGB image concatenated with the 1-channel color embedding, resulting in 4 input channels for the first layer. The output is a 3-channel RGB image representing the colored polygon.
- **Convolutional Blocks:** Each step in the encoder and decoder consists of convolutional blocks with two convolutional layers, batch normalization, and ReLU activation.
- **Skip Connections:** Skip connections are used to concatenate feature maps from the encoder to the corresponding layers in the decoder, helping to preserve spatial information lost during downsampling.
...
## Conclusion

The project successfully implemented and trained a conditional U-Net model for generating colored polygon images. The model effectively leverages color conditioning to produce outputs consistent with the input polygon and desired color. The use of wandb facilitated tracking the training process. Future work could involve adding a validation loop to monitor for overfitting, exploring different loss functions, and potentially incorporating other conditioning information like polygon shape or size.
