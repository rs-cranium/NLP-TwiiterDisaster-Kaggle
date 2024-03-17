import azureml.core
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies
print(azureml.core.VERSION)

subscription_id="your id"
resource_group="your resource grpop"
workspace_name="your workspace"
compute="your compute name"

# Load the workspace from the saved config file
ws = Workspace.from_config()


#Create New Experiment
exp = Experiment(ws, name = "BasicNLP")

import datetime
run = exp.start_logging(snapshot_directory=None)
run.log("Experimentation Start Time",str(datetime.datetime.now()))




import os
from azureml.core import Workspace, Dataset

# Load AzureML workspace
ws = Workspace.from_config()

# Authenticate Kaggle (if necessary)
# You can authenticate Kaggle using your Kaggle API token or username and API key

# Set Kaggle configuration (replace 'username' and 'api_key' with your actual Kaggle credentials)
#os.environ['KAGGLE_USERNAME'] = 'your username'
#os.environ['KAGGLE_KEY'] = 'your key

# Download Kaggle dataset
#!kaggle competitions download -c nlp-getting-started

# Unzip downloaded dataset (if necessary)
#!unzip nlp-getting-started.zip

# Upload dataset to AzureML datastore
ds = ws.get_default_datastore()
#ds.upload(src_dir='./', target_path='DisasterTweet', overwrite=True, show_progress=True)

# Create dataset from the uploaded files
dataset = Dataset.File.from_files(path=(ds, 'DisasterTweet/*.csv'))

# Register dataset
#dataset = dataset.register(workspace=ws, name='DisasterTweet', description='DisasterTweet_dataset')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
#train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
#test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
default_ds = ws.get_default_datastore()
#dataset = Dataset.get_by_name(ws, name='DisasterTweet')
train_dataset = Dataset.Tabular.from_delimited_files(path=(default_ds, 'DisasterTweet/train.csv'))
train_df = train_dataset.to_pandas_dataframe()
test_dataset = Dataset.Tabular.from_delimited_files(path=(default_ds, 'DisasterTweet/test.csv'))
test_df = test_dataset.to_pandas_dataframe()
sub_dataset = Dataset.Tabular.from_delimited_files(path=(default_ds, 'DisasterTweet/sample_submission.csv'))
sub_df = sub_dataset.to_pandas_dataframe()
train_df.head()
train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]
count_vectorizer = feature_extraction.text.CountVectorizer()

## let's get counts for the first 5 tweets in the data
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())

train_vectors = count_vectorizer.fit_transform(train_df["text"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])
# Check for missing values in the entire DataFrame
missing_values = train_df.isnull().sum()

# Display the count of missing values for each column
print(missing_values)

# Check for missing values in the "text" column
missing_values_text = train_df["text"].isnull().sum()

# Display the count of missing values in the "text" column
print("Missing values in 'text' column:", missing_values_text)

import tensorflow as tf
print(tf.__version__)
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Assuming train_df['text'] contains your text data with missing values

# Step 1: Prepare the Data
train_df['text'].fillna("", inplace=True)  # Replace missing values with an empty string

# Step 2: Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])

# Step 3: Convert text to sequences
sequences = tokenizer.texts_to_sequences(train_df['text'])

# Step 4: Padding sequences
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Step 5: Model Training
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_len))
model.add(LSTM(100))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Assuming you have target values for supervised learning
# target_values = train_df['target'].values
# Convert target values to one-hot encoding if needed

# model.fit(padded_sequences, target_values, epochs=10, validation_split=0.2)

# Step 6: Generate Text for Missing Values
missing_indices = train_df[train_df['text'] == ""].index
for idx in missing_indices:
    sequence = tokenizer.texts_to_sequences([train_df.at[idx, 'text']])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    predicted_sequence = model.predict(padded_sequence)
    #predicted_text = tokenizer.sequences_to_texts([np.argmax(predicted_sequence)])[0]
    # Convert numpy array to list
    predicted_sequence_list = predicted_sequence[0].tolist()
    # Convert sequence to text
    predicted_text = tokenizer.sequences_to_texts([predicted_sequence_list])[0]


    train_df.at[idx, 'text'] = predicted_text

# Step 7: Imputation
# The missing values in the 'text' column have been filled with generated text

# Check for missing values in the "text" column
missing_values_text = test_df["text"].isnull().sum()

# Display the count of missing values in the "text" column
print("Missing values in 'text' column:", missing_values_text)
import tensorflow as tf
print(tf.__version__)
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Assuming test_df['text'] contains your text data with missing values

# Step 1: Prepare the Data
test_df['text'].fillna("", inplace=True)  # Replace missing values with an empty string

# Step 2: Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(test_df['text'])

# Step 3: Convert text to sequences
sequences = tokenizer.texts_to_sequences(test_df['text'])

# Step 4: Padding sequences
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Step 5: Model Training
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_len))
model.add(LSTM(100))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Assuming you have target values for supervised learning
# target_values = train_df['target'].values
# Convert target values to one-hot encoding if needed

# model.fit(padded_sequences, target_values, epochs=10, validation_split=0.2)

# Step 6: Generate Text for Missing Values
missing_indices = test_df[test_df['text'] == ""].index
for idx in missing_indices:
    sequence = tokenizer.texts_to_sequences([test_df.at[idx, 'text']])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    predicted_sequence = model.predict(padded_sequence)
    #predicted_text = tokenizer.sequences_to_texts([np.argmax(predicted_sequence)])[0]
    # Convert numpy array to list
    predicted_sequence_list = predicted_sequence[0].tolist()
    # Convert sequence to text
    predicted_text = tokenizer.sequences_to_texts([predicted_sequence_list])[0]


    test_df.at[idx, 'text'] = predicted_text

# Step 7: Imputation
# The missing values in the 'text' column have been filled with generated text

train_vectors = count_vectorizer.fit_transform(train_df["text"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])

## Our vectors are really big, so we want to push our model's weights
## toward 0 without completely discounting different words - ridge regression 
## is a good way to do this.
clf = linear_model.RidgeClassifier()
# Check for missing values in the "text" column
missing_values_text = train_df["target"].isnull().sum()

# Display the count of missing values in the "text" column
print("Missing values in 'text' column:", missing_values_text)
import numpy as np

# Replace missing values (NaNs) in the "target" column with an empty string
train_df['target'].fillna("", inplace=True)

# Filter out any non-string values and convert them to strings
train_df['target'] = train_df['target'].apply(lambda x: str(x) if isinstance(x, str) else "")

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['target'])

# Check for missing values in the "text" column
missing_values_text = train_df["target"].isnull().sum()

# Display the count of missing values in the "text" column
print("Missing values in 'text' column:", missing_values_text)
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores
clf.fit(train_vectors, train_df["target"])
# Assuming clf.predict(test_vectors) returns predictions for all rows in test_vectors

# Check the length of predictions
print(len(clf.predict(test_vectors)))

# Check the length of sub_df
print(len(sub_df))

# Make sure the lengths match

# Ensure proper alignment of DataFrame index and predictions
#sub_df["target"] = clf.predict(test_vectors)[:len(sub_df)].tolist()
# Ensure proper alignment of DataFrame index and predictions
sub_df["target"] = clf.predict(test_vectors)[:len(sub_df)].tolist()


import joblib
# Save the trained model in the outputs folder
import os
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'DisasterTweets_model.pkl')
joblib.dump(value=clf, filename=model_file)


# Register the model
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'DisasterTweet_model')
import tempfile
import os
# Fetches latest model
model = ws.models['DisasterTweet_model']
print(model.name, 'version', model.version)

# Download the model file to a temporary directory
temp_dir = tempfile.mkdtemp()
model.download(target_dir=temp_dir, exist_ok=True)

# Get the path to the downloaded model file
model_path = os.path.join(temp_dir, 'DisasterTweets_model.pkl')
loaded_model = joblib.load(model_path)
#y = loaded_model.predict([[0,148,72,35,0,33.6,0.627, 50]])
sub_df["target"] = clf.predict(test_vectors)[:len(sub_df)].tolist()
print("the Output is:", sub_df["target"])
run.log("experiment End Time:", str(datetime.datetime.now()))
run.complete()
print(run.get_portal_url)
#sub_dataset = Dataset.Tabular.from_delimited_files(path=(default_ds, 'DisasterTweet/sample_submission.csv'))
#sub_df = sub_dataset.to_pandas_dataframe()
#sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sub_df["target"] = clf.predict(test_vectors)

sub_df["target"].head()
sub_df.to_csv("submission.csv", index=False)
