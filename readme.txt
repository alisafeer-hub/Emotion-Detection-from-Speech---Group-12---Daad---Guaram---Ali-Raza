# ?? Detailed Project Report: Emotion Detection from Speech Features

This project successfully developed and evaluated a high-accuracy machine learning pipeline for classifying a speaker's emotional state (Mood) based on acoustic and contextual speech features.

The model developed—a **Random Forest Classifier**—achieved an accuracy of **94.40%** on the test data, successfully meeting the performance target of ? 90%.

---

## 1. Project Goal and Objectives

The primary goal was to build a robust **multi-class classification model** capable of accurately predicting one of ten emotional states based on speech features.

Key Objectives:
1.  **Feature Engineering:** Successfully process and integrate complex speech features, notably the **MFCCs**.
2.  **Model Training:** Achieve an overall model accuracy of **90% or higher** using an ensemble classifier.
3.  **Evaluation:** Provide a detailed assessment of performance across all individual mood classes.

---

## 2. Data and Feature Engineering

The project utilized the `mood_detection_data_realistic.csv` dataset, which contains **10,000 instances**.

### Key Feature Processing Steps

| Feature Category | Features Example | Preprocessing |
| :--- | :--- | :--- |
| **Spectral (MFCCs)** | String list of 13 coefficients | **Parsing & Expansion** into 13 separate numerical columns (`MFCC_1` to `MFCC_13`). |
| **Acoustic/Prosodic** | `Pitch`, `Jitter`, `Shimmer` | **StandardScaler** (Standardization). |
| **Contextual** | `Age Group`, `Gender`, `Language` | **OneHotEncoder** (for categorical variables). |
| **Target** | `Mood` (10 classes) | **LabelEncoder**. |

---

## 3. Methodology and Model Pipeline

A robust scikit-learn **Pipeline** was used to chain the preprocessing steps with the final classifier.

### Classification Model

| Model | Configuration | Purpose |
| :--- | :--- | :--- |
| **Random Forest Classifier** | $\text{n\_estimators}=300$, $\text{max\_depth}=20$ | Selected for high accuracy and robustness in multi-class classification. |
| **Data Split** | $80\%$ Training, $20\%$ Testing | Used a stratified split to ensure equal representation of all mood classes in the test set. |

---

## 4. Key Results and Performance

### Model Accuracy

| Metric | Result | Target Met? |
| :--- | :--- | :--- |
| **Overall Accuracy** (on Test Set) | **94.40%** | **Yes** |

### Detailed Classification Performance

The model shows strong performance (high F1-Scores) across most emotional states, though it slightly struggles with distinguishing a few closely related moods like **Confused** and **Excited**, where the recall is slightly lower.

| Mood Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Angry** | 0.9901 | 0.9950 | 0.9926 |
| **Bored** | 0.9265 | 0.9450 | 0.9356 |
| **Calm** | 0.9282 | 0.9798 | 0.9533 |
| **Confused** | 0.9883 | 0.8622 | 0.9210 |
| **Excited** | 0.9834 | 0.8768 | 0.9271 |
| **Fearful** | 0.9522 | 0.9803 | 0.9660 |
| **Happy** | 0.8230 | 0.9709 | 0.8909 |
| **Neutral** | 0.9757 | 0.9168 | 0.9454 |
| **Sad** | 0.9749 | 0.9507 | 0.9626 |
| **Surprised** | 0.8953 | 0.9329 | 0.9137 |

---

## 5. Correlation Analysis

Analysis of core acoustic features revealed significant relationships:

* **Jitter** and **Shimmer** (measures of voice instability) are highly correlated ($\mathbf{r=0.85}$), suggesting they capture a common underlying phonetic property.
* **Pitch** and **Speech Rate** showed a moderate positive correlation ($\mathbf{r=0.25}$), which is consistent with high-arousal emotions (like excitement or anger) where speech is often faster and higher in tone.

The final trained model (`random_forest_model.joblib`) and the label encoder (`label_encoder.joblib`) were saved to enable seamless, error-free mood predictions on new data.
