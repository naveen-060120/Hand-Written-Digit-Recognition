# Handwritten Digit Recognition Web Application

## 📌 Project Overview

This project is a **Handwritten Digit Recognition System** built using **TensorFlow (CNN) and Flask**.
The system trains a **Convolutional Neural Network (CNN)** model using the **MNIST dataset** and provides a **web interface** where users can draw a digit and get the predicted result.

The trained model is integrated with a **Flask web application**, allowing real-time digit recognition through a browser.

---

## 🧠 Features

* Train a CNN model using the MNIST dataset
* Predict handwritten digits (0–9)
* Interactive web interface for digit input
* Real-time prediction using a trained model
* Easy setup and deployment

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* Flask
* NumPy
* Pandas
* Pillow
* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```
project-folder/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── mnist_train.zip
│   └── mnist_test.zip
│
├── templates/
│   └── index.html
│
└── static/
```

---

## ⚙️ Installation and Setup

Follow the steps below to run the project.

---

### Step 1: Extract the Dataset

Navigate to the **data** folder and extract the dataset files.

Dataset files:

* `mnist_train.zip`
* `mnist_test.zip`

After extraction, the dataset will be used for training and testing the model.

---

### Step 2: Install Required Packages

Open **Command Prompt / Terminal** and run the following command:

```bash
pip install -r requirements.txt
```

This command installs all the required Python libraries for the project.

---

### Step 3: Train and Test the Model

1. Open **Command Prompt / Terminal**
2. Navigate to the project directory
3. Run the training script:

```bash
python train_model.py
```

This step will:

* Load the MNIST dataset
* Train the CNN model
* Save the trained model for prediction

---

### Step 4: Run the Web Application

Start the Flask application using the following command:

```bash
python app.py
```

Once the server starts successfully, open your browser and go to:

```
http://127.0.0.1:5000
```

You can now draw a handwritten digit and the model will predict the number.

---

## 🚀 How the System Works

1. The MNIST dataset is extracted and loaded.
2. A CNN model is trained using the dataset.
3. The trained model is saved.
4. The Flask application loads the saved model.
5. Users draw digits on the web interface.
6. The model predicts the digit in real time.

---

## 📊 Dataset

This project uses the **MNIST Handwritten Digit Dataset**, which contains:

* **60,000 training images**
* **10,000 testing images**
* Images of digits from **0 to 9**

Each image is **28 × 28 pixels grayscale**.

---

## 👨‍💻 Author

**Naveen**

---

## 📜 License

This project is for **educational and learning purposes**.
