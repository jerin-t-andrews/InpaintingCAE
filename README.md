# Goal
Train a convolutional autoencoder (CAE) using the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (images) for image inpainting (restore missing visual information in images by understanding the latent features of the image).

# Plan
### Overall Design
1) Preprocessing: 
    - Input is the original image and output is a "corrupted" image. 
        - An image is "corrupted" by placing a significant-sized black box randomly somewhere in the image.
2) Model: 
    - Input is a corrupted image and output is a reconstruction of the original image.
3) Training:
    - To train for the latent features of the original image through the corrupted image, we take the reconstructed image and compare it to the original image (input to the preprocessing block) rather than to the model input (the corrupted image).

### Data Preprocessing
1) Normalization: Alters the shape of the distribution to fit a normal distribution
    - In our case, we will initially start by normalizing with mean and standard deviation of 0.5 (just for simplicity)
    - After setting up our base model, when optimizing we can switch to data-specific mean and standard deviation (i.e. we compute it ourselves rather than using a constant)
2) Corruption: Randomly adding noise to our original image before sending it in as input to the CAE
    - In our case, the noise will be a 10x10 black pixel mask
        - Things to consider:
            - How does the size, color/alpha, and shape of the mask impact model training (consider these as hyperparameters that we can later fine-tune)

### Model Architecture
##### Encoder
The encoder will follow this layer pattern:
    - convolutional layer -> pooling (downsampling) -> convolutional layer -> pooling (downsampling)
    - Will later take into consider cutting out the last convolutional layer and pooling layer pair (or adding another layer pair)

##### Bottleneck
At the start of the bottleneck, we will have our initial latent representation of the input image from our encoder. For now, this layer will just be a single fully connected layer that contains the same number of nodes.

We can later consider the impact of adding additional fully connected layers and the number of nodes in each layer (i.e. reducing the number of nodes representing the latent representation -> further regularization).

##### Decoder
The decoder will follow this layer pattern:
    - convolutional layer -> upsampling -> convolutional layer -> upsampling
    - Will later take into consider cutting out the last convolutional layer and upsampling layer pair (or adding another layer pair)

### Results and Modifications