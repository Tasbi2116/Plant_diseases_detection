# Plant Leaf Disease Detection

## Overview

This project implements a deep learning model to classify plant leaf images into various disease categories or healthy states using a fine-tuned VGG19 convolutional neural network (CNN). The model is trained on the **New Plant Diseases Dataset (Augmented)** from Kaggle, which contains images of plant leaves across 38 classes, including diseases and healthy states for plants such as tomatoes, apples, and grapes. The Jupyter Notebook (`plant-leaf-diseases-detection.ipynb`) provides the complete workflow for data exploration, preprocessing, model evaluation, and prediction.

The model achieves a validation accuracy of **78.49%**, but misclassifications (e.g., predicting a healthy tomato leaf as a healthy strawberry leaf) suggest potential areas for improvement, such as addressing class imbalance or further fine-tuning.

## Dataset

The project uses the **New Plant Diseases Dataset (Augmented)** available on Kaggle:
- **Training Set**: 70,295 images across 38 classes.
- **Validation Set**: 17,572 images across 38 classes.
- **Classes**: Include diseases (e.g., `Tomato___Late_blight`, `Apple___Black_rot`) and healthy states (e.g., `Tomato___healthy`, `Apple___healthy`) for various plants.
- **Source**:  [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

Each class corresponds to a subdirectory in the dataset, containing images of plant leaves labeled by plant type and condition.

## Project Structure

The main file in this repository is:
- `plant-leaf-diseases-detection.ipynb`: A Jupyter Notebook containing the complete code for data exploration, preprocessing, model evaluation, and prediction.

### Notebook Workflow
1. **Library Imports**: Imports essential libraries like NumPy, Pandas, Matplotlib, and TensorFlow/Keras for data manipulation, visualization, and deep learning.
2. **Exploratory Data Analysis (EDA)**: Lists the 38 classes in the training dataset and verifies the dataset structure.
3. **Data Preprocessing**:
   - Uses `ImageDataGenerator` for data augmentation (zoom, shear, horizontal flip) on the training set and preprocessing for both training and validation sets.
   - Images are resized to 256x256 pixels and preprocessed to match VGG19’s input requirements.
4. **Data Loading**: Configures data generators to load training (70,295 images) and validation (17,572 images) datasets in batches of 32.
5. **Visualization**: Displays sample images from the training set to inspect the data.
6. **Model Loading**: Loads a pre-trained model (`best_model.h5`), likely a fine-tuned VGG19 model.
7. **Model Evaluation**: Evaluates the model on the validation set, achieving **78.49% accuracy**.
8. **Prediction**: Defines a function to classify a single image and tests it on a sample image (`TomatoHealthy1.JPG`), though it incorrectly predicts `Strawberry___healthy`.
9. **Class Mapping**: Creates a dictionary to map numerical predictions to class names for interpretable results.

## Requirements

To run the notebook, install the following dependencies:
```bash
pip install numpy pandas matplotlib tensorflow
```

The notebook was developed in a Kaggle environment with Python 3.11.11 and a GPU accelerator. Ensure access to a compatible environment or modify paths if running locally.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/plant-leaf-disease-detection.git
   cd plant-leaf-disease-detection
   ```

2. **Download the Dataset**:
   - Download the [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle.
   - Place the dataset in the appropriate directory or update the paths in the notebook to match your local setup.

3. **Run the Notebook**:
   - Open `plant-leaf-diseases-detection.ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure the `best_model.h5` file is available in the working directory or update the path in the notebook.
   - Run the cells sequentially to perform EDA, load data, evaluate the model, and make predictions.

4. **Make Predictions**:
   - Use the `prediction` function to classify new images by providing the path to an image file:
     ```python
     prediction("path/to/your/image.jpg")
     ```

## Results

- **Model Performance**: The model achieves a validation accuracy of **78.49%** with a loss of 5.6564.
- **Sample Prediction**: When tested on a healthy tomato leaf image (`TomatoHealthy1.JPG`), the model incorrectly predicts `Strawberry___healthy`, indicating potential misclassification issues.
- **Classes**: The model classifies images into 38 categories, covering diseases and healthy states for plants like apples, tomatoes, grapes, and more.

## Limitations and Future Improvements

- **Misclassification**: The incorrect prediction suggests issues like class imbalance, insufficient training, or overfitting. Further fine-tuning or data balancing techniques (e.g., oversampling minority classes) could improve performance.
- **Model Architecture**: The notebook uses a pre-trained VGG19 model, but experimenting with other architectures (e.g., ResNet, EfficientNet) may yield better results.
- **Data Augmentation**: Additional augmentation techniques (e.g., rotation, brightness adjustment) could enhance model robustness.
- **Hyperparameter Tuning**: Adjusting learning rates, batch sizes, or the number of fine-tuned layers in VGG19 could improve accuracy.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 style guidelines and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Kaggle**: For providing the New Plant Diseases Dataset (Augmented).
- **TensorFlow/Keras**: For the deep learning framework and VGG19 model.
- **Community**: For open-source contributions to libraries used in this project.
