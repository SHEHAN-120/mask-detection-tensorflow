# Face Mask Detection Using CNN

A Convolutional Neural Network (CNN)-based binary image classification project to detect whether a person is wearing a face mask or not. Built using TensorFlow and trained on a custom dataset organized in folders.

---

## 🔗 Dataset

* [Face Mask Dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

---

## 🔧 Project Structure

```
D:\Tensorflow\archive\data\
├── Mask\
│   ├── image1.jpg
│   └── ...
└── NoMask\
    ├── image2.jpg
    └── ...
```

---

## 🧠 Techniques Used

* **TensorFlow 2.x (Keras API)** for building and training the CNN model
* **Image preprocessing**: normalization with `Rescaling(1./255)`
* **Data loading** using `image_dataset_from_directory` with validation split
* **CNN architecture**:

  * Conv2D + ReLU
  * MaxPooling2D
  * GlobalAveragePooling2D
  * Dense layers with sigmoid for binary classification
* **Prefetching** with `AUTOTUNE` for better pipeline performance


---

## 📊 Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

---

## 🔍 Challenges Faced

* **Class imbalance**: The dataset had more "Mask" images, causing biased predictions.
* **Image quality variation**: Different lighting and background in images affected accuracy.
* **Model overconfidence**: Model often predicted the majority class until class weighting and better data split were considered.
* **Prediction consistency**: Validated prediction logic with custom image loading.
* **Epoch tuning**: Choosing the right number of epochs was critical to avoid overfitting.

---

## 📈 Training Details

* **Epochs**: 10 (selected based on early convergence without overfitting)
* **Loss**: `binary_crossentropy`
* **Optimizer**: `adam`
* **Validation Split**: 20%
* **Batch Size**: 32

---

## ✅ How to Use

1. Clone this repo
2. Place dataset in the specified structure
3. Run the notebook to train the model
4. Use the prediction block to test your own image:

```python
img = image.load_img("myphoto.jpg", target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
pred = model.predict(img_array)
print("Mask" if pred[0][0] < 0.5 else "No Mask")
```


## 🏁 Future Work

* Convert model to `.tflite` for mobile deployment
* Improve data augmentation to boost generalization

---

## 🙋 Author

Shehan 

---

## 📜 License

MIT License
