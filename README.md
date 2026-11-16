# 💬 SentimentAI – Social Media Sentiment Analysis Using Machine Learning

**SentimentAI** is a machine learning-based web application that analyzes social media text and classifies its sentiment as **Positive, Negative, or Neutral**.

The project uses Natural Language Processing (NLP), TF-IDF Vectorization and machine learning algorithms to understand the sentiment expressed in social media posts.

Users can analyze text manually or provide social media post URLs where platform access is supported.

---

## 🚀 Live Application

**Live Demo:** [Add your Streamlit application URL here]

---

## 📌 Project Objectives

- Analyze sentiment in social media content.
- Classify text into Positive, Negative and Neutral categories.
- Apply NLP techniques for text preprocessing.
- Train and evaluate multiple machine learning algorithms.
- Develop an interactive web application using Streamlit.
- Explore sentiment patterns in Twitter and Reddit datasets.

---

## ✨ Features

### 1. Sentiment Analysis
- Enter text manually.
- Predict Positive, Negative or Neutral sentiment.
- View model-estimated probabilities when supported.
- View preprocessed text.

### 2. Social Media Post Analysis
Supports platform-specific workflows for:

- Reddit
- Twitter / X
- Instagram
- YouTube

Automatic extraction depends on API access and platform permissions. Instagram currently supports manual caption/comment input.

### 3. Interactive Dashboard
- Overall sentiment distribution.
- Positive, Negative and Neutral statistics.
- Twitter vs Reddit sentiment comparison.
- Interactive charts and visualizations.

### 4. Machine Learning
- Naive Bayes
- Logistic Regression
- Decision Tree

The saved Logistic Regression model is used for predictions.

### 5. Model Performance
Displays information about the selected trained model.

---

## 🧠 Machine Learning Workflow

The project follows these steps:

1. **Dataset Collection** – Collect Twitter and Reddit sentiment datasets.
2. **Data Cleaning** – Handle missing values and duplicate records.
3. **Text Preprocessing** – Convert text to lowercase, remove URLs, mentions and unnecessary characters, and remove stopwords while preserving negations.
4. **Exploratory Data Analysis (EDA)** – Study sentiment distribution and visualize the data.
5. **Feature Extraction** – Convert text into numerical features using TF-IDF.
6. **Model Training** – Train Naive Bayes, Logistic Regression and Decision Tree classifiers.
7. **Model Evaluation** – Compare model performance using accuracy, precision, recall and F1-score.
8. **Prediction** – Use the selected trained model to classify new text.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine Learning |
| TF-IDF | Text feature extraction |
| Plotly | Interactive visualizations |
| Streamlit | Web application |
| Joblib | Saving and loading trained models |
| Git & GitHub | Version control and project hosting |

---

## 📂 Project Structure

```text
Social_Media_Sentiment_Analysis/
│
├── app.py
├── requirements.txt
│
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
│
├── Reddit_Data.csv
├── Twitter_Data.csv
│
├── README.md
└── Social_Media_Sentiment_Analysis.ipynb
```

---

## 📊 Dataset Information

The project uses two datasets:

**1. Twitter Dataset**
- Contains Twitter text.
- Text column: `clean_text`
- Sentiment column: `category`

**2. Reddit Dataset**
- Contains Reddit comments.
- Text column: `clean_comment`
- Sentiment column: `category`

### Sentiment Labels

| Label | Sentiment |
|---|---|
| 1 | Positive |
| 0 | Neutral |
| -1 | Negative |

---

## ⚙️ Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/vedpatil1208-art/Social_Media_Sentiment_Analysis.git
```

### Step 2: Open the project folder

```bash
cd Social_Media_Sentiment_Analysis
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment steps:

1. Push the project files to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the project repository.
5. Set `app.py` as the main file.
6. Deploy the application.

---

## ⚠️ Limitations

- The model was trained using the supplied Twitter and Reddit datasets.
- Predictions on Instagram and YouTube content use the same trained model and are not platform-specific models.
- Automatic extraction of social media content depends on API access, authentication and platform restrictions.
- Instagram currently requires manually providing the caption or comments in this application.
- Model predictions are estimates and may not correctly understand sarcasm, slang or complex context.

---

## 🔮 Future Improvements

- Improve sentiment classification accuracy.
- Add multilingual sentiment analysis.
- Implement more platform-specific API integrations.
- Add sentiment trends and historical analytics.
- Analyze larger collections of social media comments.
- Improve sentiment explanations and reporting.

---

## 👨‍💻 Author

**Ved Patil**

Computer Science Engineering Student

GitHub: [@vedpatil1208-art](https://github.com/vedpatil1208-art)

---

## 📜 License

This project is developed for academic and educational purposes.

---

**SentimentAI – Turning Social Media Text into Meaningful Insights.**
