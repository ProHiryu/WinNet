import pandas as pd
import numpy as np
import random

df = pd.read_csv('data/data.csv')

x = list(range(13,19)) + [0]
df_2 = df.drop(df.columns[x], axis=1, inplace=False)
df_2 = df_2.apply(lambda x: x.astype(str).str.lower())

data = []

for _, row in df_2.iterrows():
    if row[11] in ['true', 'false']:
        data.append(row)

data = pd.DataFrame(data)

data.to_csv('data/clean_data.csv', index=False, header=False)

labels_df = data[data.columns[11]]

data_df = data[data.columns[:11]]
print(len(data_df))

words = [word for _, row in data_df[data_df.columns[:11]].iterrows() for word in row]

# feel free to use this import 
from collections import Counter

## Build a dictionary that maps words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = []
for _, review in data_df.iterrows():
    reviews_ints.append([vocab_to_int[word] for word in review])

encoded_labels = np.array([1 if label == 'true' else 0 for label in labels_df])
encoded_labels

features = np.array(reviews_ints)
assert(len(features) == len(encoded_labels))
print(features[:30,:10])

split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(features)*0.8)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = train_x
y_train = train_y
x_test = val_x
y_test = val_y

model = Sequential()
model.add(Dense(64, input_dim=11, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=200,
          batch_size=20)
score = model.evaluate(x_test, y_test, batch_size=128)

print(score)