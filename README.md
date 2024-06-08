# MRI Brain Tumor Classification

This project involves developing a machine learning model to classify MRI brain tumor images into 44 distinct categories. Using Convolutional Neural Networks (CNN) and leveraging pre-trained models with fine-tuning, the model achieves high classification accuracy and robust performance metrics.

## Features

- **Extensive Data Preprocessing and Augmentation:** 
  - Applied normalization, resizing, and augmentation techniques (rotation, flipping, zooming) to improve model training and performance.
  
- **Convolutional Neural Network (CNN):**
  - Implemented a CNN architecture with pre-trained models (such as VGG16 and ResNet50), followed by fine-tuning to adapt to specific data characteristics.

- **High Classification Accuracy:**
  - Achieved 87% accuracy in classifying 44 different brain tumor types.
  - Evaluated model performance using F1 score, sensitivity, and specificity metrics.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/MRIBrainTumorClassification.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd MRIBrainTumorClassification
    ```

3. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

4. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your dataset:** Organize MRI images into training and testing directories, structured by class labels.

2. **Train the model:**

    ```bash
    python train.py --data_dir path_to_your_data --epochs 50 --batch_size 32
    ```

3. **Evaluate the model:**

    ```bash
    python evaluate.py --model_path path_to_trained_model --test_dir path_to_test_data
    ```

4. **Classify new images:**

    ```bash
    python predict.py --model_path path_to_trained_model --image_path path_to_image
    ```

## Results

- **Accuracy:** 87%
- **Evaluation Metrics:** F1 Score, Sensitivity, Specificity

## Technologies

- **Python**
- **TensorFlow & Keras**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Matplotlib**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Name:** Adapa Sunilkumar
- **Email:** adapasunilkumar123@gmail.com
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/adapasunilkumar)
- **GitHub:** [GitHub Profile](https://github.com/SunilkumarAdapa)

---

![Visitor Count](https://visitor-badge.glitch.me/badge?page_id=your-username.MRIBrainTumorClassification)
