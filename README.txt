CLEARSPACE AI
CBSE CLASS XII ARTIFICIAL INTELLIGENCE PROJECT

ABOUT
ClearSpace AI is an educational text-classification website. It uses Natural
Language Processing (NLP), CountVectorizer and a supervised Decision Tree
Classifier to classify a text as:
0 = Hate speech
1 = Offensive language
2 = Neither

DATASET CHECK
The supplied twitter_data.csv contains 24,783 rows and the columns:
Unnamed: 0, count, hate_speech, offensive_language, neither, class, tweet.
The training script uses the required columns "tweet" and "class".
The class values were checked against the three vote-count columns:
0 = hate_speech, 1 = offensive_language, 2 = neither.
All 24,783 rows have complete values in the inspected columns.

MODEL EVALUATION
Using the supplied dataset and the included 75/25 stratified train/test split,
the included Decision Tree model achieved 87.23% test accuracy.
The test-set F1 scores were:
Hate speech: 0.37
Offensive language: 0.92
Neither: 0.86

IMPORTANT LIMITATION
The dataset is imbalanced: class 0 (hate speech) has 1,430 examples,
class 1 has 19,190, and class 2 has 4,163. The model's hate-speech recall
on the test split was 0.46. Therefore, the overall accuracy should not be
treated as proof that the model reliably detects hate speech. This is a
school project prototype, not a moderation or enforcement system.
Predictions can be wrong due to context, sarcasm, dialect, spelling and
bias in the dataset. Do not use the result to punish or report people
automatically.

RUN LOCALLY
1. Install Python.
2. Open a terminal in this folder.
3. Run: pip install -r requirements.txt
4. Run: streamlit run app.py
The trained model file hate_speech_model.pkl is included, so the website
can start without the CSV.

RETRAIN THE MODEL
Place twitter_data.csv in this folder, then run:
python train_model.py
This recreates hate_speech_model.pkl. The CSV is not included in this
deployment package.

DEPLOY ONLINE WITH STREAMLIT COMMUNITY CLOUD
1. Create a GitHub repository.
2. Upload app.py, requirements.txt, hate_speech_model.pkl and README.txt.
3. Do not upload the raw dataset unless you have permission to share it.
4. At https://share.streamlit.io/, connect GitHub and create an app.
5. Select the repository, branch "main", and main file "app.py".
6. Deploy.

PROJECT CONCEPTS
AI domain: Natural Language Processing
Learning type: Supervised Learning
Algorithm: Decision Tree Classifier
Feature extraction: CountVectorizer
Input: User-entered text
Output: Predicted text category

This project is an educational demonstration of text classification.
