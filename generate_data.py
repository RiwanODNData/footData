#!/usr/bin/env python
# coding: utf-8

# # 2. List of different approaches of how such game can be re-created. You can also use consider the other type of the data than we are providing

# Multiple method could be used:
# 
# 1) data augmentation, noise injection to norm and repeat other param
# 2) Use markov chain to model action, then generate size of list using normal distribution, and generate norm value using normal distribution
# 3) Use markov chain to model action, then generate size of list using normal distribution, and generate norm value using GAN
# 4) Use markov chain to model action, then generate size of list using normal distribution, and generate norm value using LSTM
# 

# # Please fully describe at least one approach you would choose (in jupyter notebook or some additional pdf): 
# ## a. The chosen architecture/algorithm. Why the decision was made, why it makes sense, and what kind of input it assumes. If the mathematical theory for the chosen approach is too complicated, the flow chart is enough.
# ## b. The pre- / post-processing of the data

# A) 
# 
# 
# ## Summary of Solution:
# 
# I employed a multi-pronged approach to model and generate the desired sequences:
# 
# ### 1. **Markov Chain for Action Sequencing**:
# I utilized a Markov chain to model and generate action sequences. This approach ensures that the action labels are generated in a meaningful and coherent order, mimicking real-world scenarios.
# 
# ### 2. **Normal Distribution for List Size Determination**:
# Given the significance of maintaining the gait length, I chose to model the size of the list using a normal distribution. This approach ensures that the generated lengths closely resemble the real-world data distribution.
# 
# ### 3. **LSTM for Norm Value Generation**:
# For the generation of norm values, I deployed an LSTM (Long Short-Term Memory) model, which is adept at handling and generating sequences. The LSTM architecture comprised two layers:
# - A layer for norm values: This captures the sequence of previous values.
# - A layer for labels: Recognizing that the norm values are influenced by actions, this layer ensures that the generated norm values are consistent with the associated action labels.
# 
# ## Reflection:
# Upon the completion of this model, I believe that while the LSTM approach has its strengths, there might be areas of potential refinement. Some generated values appear unconventional, indicating the possibility of overfitting or over-complexity. We can delve deeper into these aspects and discuss potential improvements during our interview.

# B)
# 
# 
# 
# Proper preprocessing of the data was essential to prepare it for modeling. The following steps were undertaken:
# - **Encoding Labels**: The action labels were encoded to numerical values, making them suitable for model processing.
# - **Standardizing Norm Values**: To stabilize the LSTM and ensure consistent input, the norm values were standardized. This standardization helps in ensuring smoother training and better performance of the LSTM model.
# 

# 

# # 5. Parametrize your fitted algorithm/program for recreating the game in the following way:
# 
# Bonus: It will be possible to generate game with specific type of play - e.g
# more attacking game (there will be more passes, shots), defending game
# (more tackles, interceptions, etc.), or just normal game.

# ##  1. Adjusting the Markov Chain for Specific Actions:
# To generate specific actions with greater or lesser frequency, the transition probabilities in the Markov chain can be fine-tuned. For instance:
# 
# ### For a Striker: 
# After observing that the probability of a shot post a dribble is 0.09, one could reduce the probabilities of other subsequent actions post-dribble, transferring that proportion to the shooting probability. It's crucial to strike a balance to avoid excessive shooting actions, which may deviate from realistic play.
# 
# ### For a Defender: 
# To simulate defensive actions like tackling with greater frequency, the probabilities of actions like tackling could be increased post certain preceding actions.
# 
# ## 2. Balancing the Coefficients:
# The key challenge lies in adjusting the coefficients in a manner that the generated sequences remain realistic. While emphasizing a specific action, care should be taken to ensure that the player doesn't overly rely on that singular action, as this would distort the authenticity of the gameplay.

# 

# In[1]:


# Import necessary libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Concatenate, Input, Flatten, Reshape, RepeatVector, Dropout
from keras.layers import concatenate
from keras.optimizers import Adam

from keras.callbacks import Callback

import re
from keras.models import load_model

import json


# Define the file path. Replace with the actual path to the file if different.
file_path = 'data/match_1.json'
# Load the JSON file into a pandas DataFrame with records orientation.
match_1_data = pd.read_json(file_path, orient='records')

file_path = 'data/match_2.json'
match_2_data = pd.read_json(file_path, orient='records')


# Combiner les données des deux matches
data = pd.concat([match_1_data, match_2_data], ignore_index=True)
data


# In[2]:


def compute_transition_frequencies(data):
    """
    Compute the frequency of transitions between different actions in the data.
    
    Args:
    - data (DataFrame): The dataset containing labeled sequences of actions.

    Returns:
    - matrix (dict): A dictionary representing the frequency of transitions 
                     between different actions.
    """
    matrix = {}  # Initialize an empty dictionary to hold transition frequencies.
    
    # Loop through each row of the data except the last one
    for i in range(len(data)-1):
        # Get the current action from the 'label' column
        current_action = data.iloc[i]['label']
        # Get the next action from the 'label' column
        next_action = data.iloc[i+1]['label']
        
        # Check if the current action is already in the matrix
        if current_action not in matrix:
            matrix[current_action] = {}  # If not, initialize it with an empty dictionary.
        
        # Check if the next action is already a key under the current action
        if next_action not in matrix[current_action]:
            matrix[current_action][next_action] = 1  # If not, initialize it with a count of 1.
        else:
            matrix[current_action][next_action] += 1  # If yes, increment the count.
    
    # Loop to normalize the transition counts to get transition probabilities
    for current_action, transitions in matrix.items():
        # Calculate the total count of transitions from the current action
        total = sum(transitions.values())
        # Normalize each count by the total
        for next_action, count in transitions.items():
            matrix[current_action][next_action] = count / total
    
    return matrix


def generate_actions(transitions, n):
    """
    Generate a sequence of actions based on the transition probabilities.

    Args:
    - transitions (dict): A dictionary representing the transition probabilities.
    - n (int): Length of the sequence to be generated.

    Returns:
    - sequence (list): Generated sequence of actions.
    """
    # Start with the "walk" action
    current_action = "walk"
    sequence = [current_action]

    for _ in range(n-1):
        # Check if there are possible transitions from the current action
        if current_action in transitions and transitions[current_action]:
            next_actions = list(transitions[current_action].keys())
            next_probs = list(transitions[current_action].values())
            current_action = np.random.choice(next_actions, p=next_probs)
        else:
            # If no possible transitions, choose a random action (excluding "walk" to avoid repetition)
            current_action = np.random.choice([action for action in transitions.keys() if action != "walk"])
        sequence.append(current_action)

    return sequence





# In[3]:


def get_label_specific_stats(data):
    """
    Compute the mean and standard deviation of lengths for each label.

    Args:
    - data (DataFrame): The dataset containing sequences and their labels.

    Returns:
    - stats (dict): A dictionary where keys are labels and values are 
                    (mean, std) tuples for that label's lengths.
    """
    # Compute lengths for each row and then group by the label
    lengths_by_label = data.groupby('label')['norm'].apply(lambda x: x.apply(len))
    
    means = lengths_by_label.groupby('label').mean()
    stds = lengths_by_label.groupby('label').std()
    
    return {label: (mean, std) for label, mean, std in zip(means.index, means, stds)}

def generate_length_for_action(action, stats):
    """Generate a sequence length for a given action based on stored statistics."""
    mean, std = stats[action]
    return int(abs(np.random.normal(mean, std)))






# In[4]:


# Use the build_transition_matrix function to get transition probabilities
transitions = compute_transition_frequencies(data)

stats = get_label_specific_stats(data)


# In[5]:


# # Use the build_transition_matrix function to get transition probabilities
# transitions = compute_transition_frequencies(data)
# # Generate actions for 10 minutes
# new_actions = generate_actions(transitions, 5)  # Adjust this based on your data


# stats = get_label_specific_stats(data)
# print(new_actions)
# print(stats)
# # Generate new sequences
# new_sequences = []
# for action in new_actions:

#     length = generate_length_for_action(action, stats)
#     print(length)
#     new_sequences.append({"label": action, "norm": [0] * length})

# print(new_sequences)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Modeling part

# In[ ]:





# In[6]:


num_labels = len(set(data['label']))

# Define the input shapes
sequence_length = 20

# Sequence input
sequence_input = Input(shape=(sequence_length, 1), name='sequence_input')

# Label input and embedding
label_input = Input(shape=(1,), name='label_input')
label_embedding = Embedding(input_dim=num_labels, output_dim=8)(label_input)  # increased embedding size
label_embedding = Flatten()(label_embedding)

# Expand dimensions and tile the embedding to match the sequence length
label_embedding = RepeatVector(sequence_length)(label_embedding)

# Concatenate sequence and label embedding
merged_input = concatenate([sequence_input, label_embedding])

# LSTM layers with dropout
lstm_out = LSTM(32, return_sequences=True, dropout=0.0)(merged_input)  # adjusted LSTM units and dropout
lstm_out = LSTM(16, dropout=0.0)(lstm_out)  # adjusted LSTM units and dropout

# Dense layer to predict the next value in the sequence
output = Dense(1, activation='linear')(lstm_out)

# Compile the model with the default learning rate
optimizer = Adam(learning_rate=0.01)
model = Model(inputs=[sequence_input, label_input], outputs=output)
model.compile(optimizer=optimizer, loss='mse')


# In[ ]:





# In[7]:


sequences = data['norm'].tolist()

# Compute the mean and standard deviation of your sequences
all_values = [item for sublist in sequences for item in sublist]
mean = np.mean(all_values)
std = np.std(all_values)

# Standardize the sequences
sequences = [(np.array(seq) - mean) / std for seq in sequences]
sequences = [seq.tolist() for seq in sequences]

labels = data['label'].tolist()

# Convert labels to unique integers for embedding
label_to_int = {label: i for i, label in enumerate(set(labels))}
encoded_labels = [label_to_int[label] for label in labels]

def prepare_training_data(sequences, encoded_labels, sequence_length):
    X_seq, X_label, Y = [], [], []
    for seq, label in zip(sequences, encoded_labels):
        for i in range(1, min(len(seq), sequence_length)):
            padded_seq = seq[:i] + [0] * (sequence_length - i)
            if len(padded_seq) != sequence_length:
                print(f"Unexpected sequence length. Before padding: {len(seq[:i])}, After padding: {len(padded_seq)}")
                continue
            X_seq.append(padded_seq)
            X_label.append(label)
            Y.append(seq[i])
    X_seq = np.array(X_seq)
    X_label = np.array(X_label)
    Y = np.array(Y)
    return X_seq, X_label, Y

X_seq, X_label, Y = prepare_training_data(sequences, encoded_labels, sequence_length)


# In[8]:


class PredictSamples(Callback):
    def __init__(self, X_seq_sample, X_label_sample, Y_sample):
        super(PredictSamples, self).__init__()
        self.X_seq_sample = X_seq_sample
        self.X_label_sample = X_label_sample
        self.Y_sample = Y_sample

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict([self.X_seq_sample, self.X_label_sample])
        print("\nSample predictions after epoch {}:".format(epoch+1))
        for actual, predicted in zip(self.Y_sample, predictions):
            print("Actual: {:.3f}, Predicted: {:.3f}".format(actual, predicted[0]))
        print("----------------------")

# Choose a small sample of your data
sample_size = 10
X_seq_sample = X_seq[:sample_size]
X_label_sample = X_label[:sample_size]
Y_sample = Y[:sample_size]

# Use the custom callback during training
sample_callback = PredictSamples(X_seq_sample, X_label_sample, Y_sample)



# In[9]:


# # When training, include this callback
# history = model.fit([X_seq, X_label], Y, epochs=15, batch_size=32, validation_split=0.2, callbacks=[sample_callback])


# # Plotting the training and validation loss
# plt.figure(figsize=(8, 4))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation MSE')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Value')
# plt.legend()
# plt.show()


# In[10]:


# save my model:
# model.save('model/lstm_norm_values.keras')


# In[ ]:





# In[ ]:





# In[ ]:





# # Prediction part

# In[11]:


def warmup_sequence(model, sequence_length, action):
    # Initialize a zero sequence for warmup
    current_sequence = np.zeros(sequence_length).reshape(1, sequence_length)
    warmup_output = []
    
    # Get the encoded label for the action
    encoded_label = get_encoded_label_for_action(action)
    encoded_label_array = np.array([encoded_label]).reshape(1, -1)
    
    for _ in range(sequence_length):
        next_value = model.predict([current_sequence, encoded_label_array])
        warmup_output.append(next_value[0][0])
        current_sequence = np.append(current_sequence[0, 1:], next_value[0]).reshape(1, sequence_length)
    
    return np.array(warmup_output)

def fill_table_with_sequence(model, action_table_length, sequence_length, action, last_20_values):
    # Create a zero-filled action table based on the provided length
    action_table = np.zeros(action_table_length)
    
    # Check if the last_20_values are all zeros (first action scenario)
    if np.all(last_20_values == 0):
        warmup_output = warmup_sequence(model, sequence_length, action)
        action_table[:min(len(warmup_output), action_table_length)] = warmup_output[:action_table_length]
        current_sequence = warmup_output.reshape(1, sequence_length)
        start_idx = len(warmup_output)
    elif action_table_length >= sequence_length:
        current_sequence = last_20_values.reshape(1, sequence_length)
        action_table[:sequence_length] = last_20_values
        start_idx = sequence_length
    else:  # In case action_table_length is less than sequence_length
        assign_length = min(sequence_length, action_table_length)
        action_table[:assign_length] = last_20_values[-assign_length:]
        current_sequence = last_20_values[-sequence_length:].reshape(1, sequence_length)
        start_idx = assign_length

    # Get the encoded label for the action
    encoded_label = get_encoded_label_for_action(action)
    encoded_label_array = np.array([encoded_label]).reshape(1, -1)
    
    for i in range(start_idx, action_table_length):
        # Predict the next value using the LSTM model
        next_value = model.predict([current_sequence, encoded_label_array])
        action_table[i] = next_value[0][0]
        current_sequence = np.append(current_sequence[0, 1:], next_value[0]).reshape(1, sequence_length)

    return action_table, action_table[-sequence_length:]


def get_encoded_label_for_action(action):
    return label_to_int[action]


# In[21]:


def generateData(time_in_minutes, transitions, stats, model, sequence_length, last_20_values):
    # 50 samples/second * 60 seconds/minute = 3000 samples/minute
    total_samples_needed = time_in_minutes * 3000

    # Estimate average action length using stats
    avg_action_length = np.mean([mean for mean, std in stats.values()])

    # Estimate the number of actions needed
    num_actions = int(total_samples_needed / avg_action_length)

    # Generate action list
    action_list = generate_actions(transitions, num_actions)
    print("Number of action to generate "+str(len(action_list)))
    # Create a dataframe to store the results
    df = pd.DataFrame(columns=['label', 'norm'])
    
    for index, action in enumerate(action_list):
        print("Generating action number: "+str(index))
        action_table_length = generate_length_for_action(action, stats)
        
        # Ensure the first action has a minimum length of 20
        if index == 0 and action_table_length < 20:
            action_table_length = 20

        filled_table, new_last_values = fill_table_with_sequence(model, action_table_length, sequence_length, action, last_20_values)
        
        # Update last_20_values
        if len(filled_table) < sequence_length:
            last_20_values = np.concatenate((last_20_values[len(filled_table):], filled_table))
        else:
            last_20_values = new_last_values
        
        # Append to dataframe
        df.loc[len(df)] = [action, filled_table.tolist()]  # store the filled_table as a list

    # Return the dataframe
    return df


# In[20]:


def get_next_match_number(directory_path):
    # List all files in the directory
    files = os.listdir(directory_path)
    
    # Regular expression pattern to match our file names and extract the number
    pattern = re.compile(r'output_Match_(\d+)')
    
    # Extract all match numbers from filenames
    match_numbers = [int(match.group(1)) for file in files for match in [pattern.search(file)] if match]
    
    # Return the next match number
    if match_numbers:
        return max(match_numbers) + 1
    else:
        return 1


# In[19]:


import argparse

def main(time_in_minutes):
    last_20_values = np.zeros(20)
    model_path = "model/lstm_norm_values.keras"
    model = load_model(model_path)
    df_result = generateData(time_in_minutes, transitions, stats, model, sequence_length, last_20_values)

    df = df_result
    # mean and std are previously defined
    for idx, row in df.iterrows():
        norm_array = np.array(row['norm'])
        reversed_norm = (norm_array * std) + mean
        df.at[idx, 'norm'] = reversed_norm.tolist()


    def get_next_match_number(directory_path):
        files = os.listdir(directory_path)
        pattern = re.compile(r'output_Match_(\d+)')
        match_numbers = [int(match.group(1)) for file in files for match in [pattern.search(file)] if match]
        if match_numbers:
            return max(match_numbers) + 1
        else:
            return 1



    # Convert the dataframe to a list of dictionaries
    formatted_data = df.to_dict(orient='records')
    
    # Save to JSON file
    directory_path = "generatedData/"
    next_match_number = get_next_match_number(directory_path)
    filename = f"generatedData/output_Match_{next_match_number}_{time_in_minutes}_Min.json"
    with open(filename, 'w') as f:
        json.dump(formatted_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data for a specified time duration.')
    parser.add_argument('time_in_minutes', type=int, help='Duration for which data should be generated (in minutes).')
    args = parser.parse_args()
    main(args.time_in_minutes)


# If you want to generate from here

# In[25]:


# time_in_minutes = 1


# In[26]:


# # Usage
# last_20_values = np.zeros(20)



# model_path = "model/lstm_norm_values.keras"
# model = load_model(model_path)

# df_result = generateData(time_in_minutes, transitions, stats, model, sequence_length, last_20_values)
# print(df_result)

# df=df_result
# # mean and std are previously defined
# for idx, row in df.iterrows():
#     # Convert the list to numpy array
#     norm_array = np.array(row['norm'])
    
#     # Reverse the standardization
#     reversed_norm = (norm_array * std) + mean
    
#     # Update the 'norm' column in the dataframe
#     df.at[idx, 'norm'] = reversed_norm.tolist()




# # Display the updated dataframe
# print(df)





# # Convert the dataframe to a list of dictionaries
# formatted_data = df.to_dict(orient='records')

# # Save to JSON file
# directory_path = "generatedData/"
# next_match_number = get_next_match_number(directory_path)
# filename = f"generatedData/output_Match_{next_match_number}_{time_in_minutes}_Min.json"
# with open(filename, 'w') as f:
#     json.dump(formatted_data, f)


# In[ ]:




