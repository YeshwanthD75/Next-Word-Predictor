# Next-Word-Predictor
**Next Word Predictor with LSTM**

### Project Description
This project implements a Next Word prediction model using Long Short-Term Memory (LSTM) neural networks. The model is trained on a dataset of text sequences to predict the next word given a sequence of words. The primary objective is to learn and understand the context of word sequences and provide accurate predictions for the next word in a sentence. This project includes the preprocessing of text data, construction and training of the LSTM model, and evaluation and prediction functionalities.

### Layers Used in the Model
1. **Embedding Layer**:
   - Converts integer-encoded words into dense vectors of fixed size.
   - Captures semantic relationships between words.
   - `model.add(Embedding(num_classes, 128, input_length=max_len - 1))`
   
2. **LSTM Layer**:
   - Learns the temporal dependencies and context between words in the sequences.
   - Handles long-term dependencies effectively.
   - `model.add(LSTM(350))`
   
3. **Dense Output Layer**:
   - Transforms the output from the LSTM layer into a vector with the same size as the vocabulary.
   - Uses a softmax activation function to produce probabilities for each word in the vocabulary.
   - `model.add(Dense(num_classes, activation='softmax'))`
   
### Additional Features
- **Callbacks**: Utilizes callbacks such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint to improve training efficiency and model performance.
- **Trie Implementation**: Includes a Trie data structure for efficient word insertion and search.
- **Heap for Model Checkpoints**: Uses a custom callback with a heap to store and manage model checkpoints based on validation loss.

### Usage
- **Training**: The model is trained on the provided dataset, with input sequences generated and padded for uniformity.
- **Prediction**: After training, the model can be used to predict the next word in a given sequence, with the ability to generate text iteratively.

### Getting Started
1. **Mount Google Drive**:
   - Mount Google Drive to access the dataset and save the model.
   
2. **Prepare Dataset**:
   - Load and preprocess the dataset, including tokenization and sequence padding.
   
3. **Build and Train Model**:
   - Construct the model with embedding, LSTM, and dense layers.
   - Train the model with appropriate callbacks to optimize performance.
   
4. **Evaluate and Predict**:
   - Evaluate the model on the training data.
   - Use the trained model to predict the next word in a given text sequence.

This project provides a comprehensive implementation of a Next Word prediction model using LSTM, with detailed steps for preprocessing, model building, training, and prediction.
