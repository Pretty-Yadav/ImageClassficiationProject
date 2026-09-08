# Gender Classification using CNN

This beginner-friendly project demonstrates a CNN image-classification pipeline using the labeled face images already downloaded into `data/`. The model learns the labels provided by the dataset; it does not determine a person's gender identity.

## Objective

The project demonstrates an explainable computer-vision workflow suitable for a college project or viva:

```text
Dataset -> reproducible subset -> preprocessing -> CNN -> evaluation
        -> saved model -> face detection -> image / video / webcam prediction
```

The model predicts the dataset's binary annotations. It must not be interpreted as determining a person's gender identity.

## Technologies

Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, scikit-learn, Pillow, and Jupyter Notebook.

## Dataset

The downloaded dataset is stored locally under `data/` with this structure:

```text
data/
├── Training/
│   ├── female/
│   └── male/
└── Validation/
	├── female/
	└── male/
```

The notebook checks the folders and counts images before training. It uses at most 5,000 images per training class and 1,000 per validation class, selected with random seed `42`. Increase the configurable limits later for a stronger experiment. The full dataset remains untouched.

## Project Layout

```text
notebooks/gender_classification.ipynb  # Main CNN implementation
models/gender_cnn.keras                 # Saved trained model
models/face_detection_yunet_2023mar.onnx # Robust face detector
app.py                                  # Streamlit webcam frontend
sample_images/                          # Local test-image placeholder
sample_videos/                          # Local video placeholder
requirements.txt
```

## How to Run

1. Activate the environment: `source .venv/bin/activate`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `notebooks/gender_classification.ipynb` in Jupyter, VS Code, or Google Colab.
4. Run the notebook from top to bottom. It checks the local dataset, selects the subset, trains, evaluates, and saves `models/gender_cnn.keras`.
5. Start the webcam frontend with `streamlit run app.py`, then allow camera access in your browser.

The full dataset is intentionally excluded from this repository. Keep the local `data/` folder in place when running the notebook.

The app uses OpenCV YuNet for face detection rather than the older Haar cascade, which reduces false detections on clothing and backgrounds. The detector model is stored in `models/face_detection_yunet_2023mar.onnx`.

## Webcam

The recommended webcam demo is the Streamlit app: run `streamlit run app.py` and allow the browser to use the camera. It captures a frame, detects faces, applies the saved CNN to each RGB face crop, and displays the annotated result. Browser camera access works even when the Python server cannot directly see a local `/dev/video0` device.

## Limitations

- Accuracy depends on dataset bias, lighting, pose, occlusion, image quality, and detection errors.
- The subset reduces training coverage and may not represent real-world video frames.
- The model can make classification errors and is not a reliable identity or gender system.
- The project is an educational demonstration of computer vision, CNNs, and image classification.

## Future Improvements

Possible next steps include a stronger face detector, model optimization, real-time smoothing, more robust evaluation, and larger or more diverse datasets.

## License and Data Notice

This project code is provided for educational use. Check the source dataset's own documentation and usage terms before redistribution or commercial use.