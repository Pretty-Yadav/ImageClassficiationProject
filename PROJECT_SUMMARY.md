# Face Image Classification Project

## 1. Project Goal

This project uses deep learning to classify detected face images into the two labels provided by the dataset: `female` and `male`. These are dataset labels, not a reliable way to determine a person's actual gender identity.

The project also includes a small Streamlit app where a user can upload a photo or capture a webcam image and see the model's prediction.

## 2. Dataset

The dataset is already organized into training and validation folders:

```text
data/
├── Training/
│   ├── female/
│   └── male/
└── Validation/
    ├── female/
    └── male/
```

Available images:

- Training: 23,243 female and 23,766 male images
- Validation: 5,841 female and 4,744 male images

We only use a small amount from here, to speed up training in exchange of efficiency:

- Training: 5,000 images per class, 10,000 total
- Validation: 1,000 images per class, 2,000 total
- Random seed: `42` 

## 3. Technologies Used

- **Python**: Main programming language
- **TensorFlow and Keras**: Build, train, save, and use the neural network
- **OpenCV**: Read images, resize face crops, and process camera frames
- **OpenCV YuNet**: Detect faces and reduce false detections from backgrounds or clothing
- **NumPy**: Store image arrays and prepare model input
- **Matplotlib**: Display sample images and training graphs
- **scikit-learn**: Create the classification report and confusion matrix
- **Streamlit**: Provide the webcam and image-upload interface
- **Jupyter Notebook**: Document and run the machine-learning workflow

## 4. Machine-Learning Workflow

1. Read the `Training` and `Validation` folders.
2. Use folder names as the class labels.
3. Select a subset of images.
4. Read images with OpenCV and convert them to RGB.
5. Resize every image or face crop to `128 x 128` pixels.
6. Apply training augmentation: horizontal flipping, small rotations, and small zooms.
7. Train a CNN to predict one of the two dataset labels.
8. Evaluate the model on images it did not train on.
9. Save the trained model as `models/gender_cnn.keras`.
10. Detect faces in uploaded or webcam images with YuNet, crop each face, and classify each crop.

## 5. CNN Model

The model is a simple convolutional neural network:

- Four convolution layers with 32, 64, 128, and 256 filters
- Max-pooling after each convolution block
- A flatten layer followed by a 256-unit dense layer
- Dropout of `0.4` to help reduce overfitting
- One sigmoid output for binary classification
- Input size: `128 x 128 x 3` RGB pixels
- About 2.75 million trainable parameters

Training settings included batch size `32`, at most `10` epochs, the Adam optimizer, binary cross-entropy loss, and early stopping based on validation loss.

## 6. Results

On the 2,000-image validation subset:

- **Accuracy: 94.55%**
- Best validation accuracy during training: **94.95%**
- Female precision: **0.95**, recall: **0.94**, F1-score: **0.95**
- Male precision: **0.94**, recall: **0.95**, F1-score: **0.95**

The confusion matrix shows how often female and male dataset labels were confused. Accuracy is useful, but it does not prove that the model works equally well for every person or every real-world situation.

## 7. OpenCV and the App

OpenCV YuNet detects likely faces first. Only detections with at least `85%` detector confidence are sent to the CNN. This prevents the app from drawing boxes around patterns that are not faces.

The Streamlit app supports two inputs:

- **Webcam**: capture an image through the browser after giving camera permission
- **Upload image**: select a JPG, JPEG, or PNG file

Both inputs use the same face detector and the same saved CNN model.

Run the app with:

```bash
streamlit run app.py
```

## 8. Final thoughts

The model makes still mistakes because of lighting, pose, image quality, face-detection errors, dataset bias, and differences between the training images and real camera images
