# Glaucoma-Detection
# U-Net for Glaucoma Detection: Optic Disc and Cup Segmentation

This project uses a U-Net architecture for segmenting the optic disc and cup in retinal fundus images for glaucoma detection. The U-Net model is a type of convolutional neural network that is widely used for biomedical image segmentation.

## Project Structure

The project is structured as follows:

- `UNET`: This class defines the U-Net model architecture.
- `DoubleConv`: This class defines the double convolution operation used in the U-Net model.
- `Dataset`: This class is used for loading the dataset of retinal images and their corresponding masks.
- `save_checkpoint` and `load_checkpoint`: These functions are used for saving and loading the model.
- `get_loaders`: This function is used for getting the data loaders for training and validation datasets.
- `check_accuracy`: This function is used for checking the accuracy of the model on a data loader.
- `save_predictions_as_imgs`: This function is used for saving the predicted masks as images.
- `train_fn`: This function is used for training the model on a data loader.
- `main`: This is the main function where the model is trained and evaluated.
- `calculate_white_area`: This function calculates the white pixel area in binary masks.
- `compare_images`: This function compares the optic cup and disc masks to generate a combined visualization.

## Usage

To use this project, you need to have a dataset of retinal images and their corresponding masks. The paths to the training and validation datasets are specified in the `TRAIN_IMG_DIR`, `TRAIN_MASK_DIR`, `VAL_IMG_DIR`, and `VAL_MASK_DIR` variables.

You can run the project by running the `main` function. This will train the model on the training dataset, evaluate it on the validation dataset, and save the model and the predicted masks.

Additionally, the project includes scripts for:
- Inferring optic disc and cup masks from new images.
- Calculating the cup-to-disc ratio (CDR) to assess glaucoma risk.
- Visualizing the overlap between optic cup and disc masks.

## Requirements

This project requires the following libraries:

- PyTorch
- torchvision
- albumentations
- PIL
- numpy
- tqdm

