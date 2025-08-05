# Project Report: Colored Polygon Generation with Conditional UNet

## Project Overview

This project aimed to develop a deep learning model capable of generating colored polygon images based on an input grayscale polygon image and a specified color condition. The core of the solution is a modified U-Net architecture that incorporates the color information as a conditioning input. The model was trained on a custom dataset of polygon images and their corresponding colored versions. Experiment tracking was performed using Weights and Biases (wandb) to monitor training progress.

## Dataset

The dataset consists of pairs of input and output images, along with JSON annotations.
- Input Images:Grayscale images containing various polygon shapes.
- Output Images: The corresponding input polygons filled with a specific color.
- Annotations (`data.json`):** A JSON file containing a list of dictionaries, where each dictionary links an input image filename (`input_polygon`) to its corresponding output image filename (`output_image`) and the target color (`colour`).

The dataset is split into training and validation sets, located in `/content/dataset/training` and `/content/dataset/validation` respectively. The `PolygonColorDataset` class was implemented to load this data, process the images, and convert the color names into one-hot encoded tensors for use as conditioning. It also handles matching input images to their annotations and corresponding output images based on the `data.json` file.

## Model Architecture: Conditional UNet

The model employed is a U-Net architecture enhanced with a conditioning mechanism to incorporate the desired color.

- U-Net Base: The model follows the standard U-Net structure with an encoder-decoder pathway and skip connections. The encoder progressively downsamples the input image, capturing hierarchical features, while the decoder upsamples the features to generate the output image.
- Color Conditioning: The color information, represented as a one-hot encoded vector, is embedded into a spatial map matching the input image dimensions using a fully connected layer. This color map is then concatenated with the input image channels before the first convolutional layer of the encoder. This allows the model to learn color-dependent features from the beginning of the network.
- Input/Output Channels: The input to the model is a 3-channel RGB image concatenated with the 1-channel color embedding, resulting in 4 input channels for the first layer. The output is a 3-channel RGB image representing the colored polygon.
- Convolutional Blocks: Each step in the encoder and decoder consists of convolutional blocks with two convolutional layers, batch normalization, and ReLU activation.
- Skip Connections: Skip connections are used to concatenate feature maps from the encoder to the corresponding layers in the decoder, helping to preserve spatial information lost during downsampling.

## Hyperparameters

The following hyperparameters were used for training:

- Learning Rate: 1e-3
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSELoss)
- Epochs:40
- Batch Size: 8
- Color Dimension: 8 (corresponding to the number of unique colors in the dataset: red, green, blue, yellow, orange, purple, cyan, magenta)

## Training Dynamics

The model was trained for 20 epochs. The training loss, monitored using wandb, showed a decreasing trend over the epochs, indicating that the model was learning to generate colored polygons. The MSE loss measures the pixel-wise difference between the generated output and the ground truth colored image. A decreasing loss suggests that the model's generated images are becoming more similar to the desired output images.


## Key Learnings and Observations

- Effectiveness of Conditioning: The conditional U-Net architecture successfully utilizes the color information to guide the image generation process. By embedding and concatenating the color condition with the input image, the model learns to associate specific input polygons with the desired output colors. The visual examples generated during inference demonstrate the model's ability to color different polygons with the specified colors.
- Dataset Handling: Correctly parsing the `data.json` annotation file and aligning input images with their corresponding annotations and output images was crucial for successful data loading and training. Initial issues with dataset loading highlighted the importance of understanding the exact structure of the annotation data.
- Model Complexity: A U-Net based architecture is well-suited for this image-to-image translation task, allowing the model to capture both local and global features necessary for generating coherent colored polygons.
- Challenges: One challenge encountered was correctly interpreting the `data.json` file structure and ensuring that the `PolygonColorDataset` class accurately mapped input images to their annotations and output images. Another potential challenge, not fully explored in the provided code, is handling potential overfitting without explicit validation loss monitoring and techniques like early stopping.

## Conclusion

The project successfully implemented and trained a conditional U-Net model for generating colored polygon images. The model effectively leverages color conditioning to produce outputs consistent with the input polygon and desired color. The use of wandb facilitated tracking the training process. Future work could involve adding a validation loop to monitor for overfitting, exploring different loss functions, and potentially incorporating other conditioning information like polygon shape or size.


Wandb Project Link: https://wandb.ai/passionateaman148-studentuniverse/ayna-unet-color?nw=nwuserpassionateaman148
