Minibatching with PyTorch: Logistic Regression & LSTM Models for Sentiment Analysis

Task Overview:

The objective of this project is to understand and implement minibatching in PyTorch. I am to create both a logistic regression model and an LSTM model that employs minibatching for sentiment analysis. The primary goal is to ascertain whether a given text (like a sentence) exudes a positive or negative sentiment. The IMDB reviews dataset is used for this purpose. I've utilized a provided starter Colab notebook which loads the dataset, constructs, trains, and evaluates the model on single instances. It also incorporates a model correctness test to initialize the model randomly on sample instances and display the score.

Padding & Masking:

To ensure all inputs in a batch have the same length for the DataLoader, padding is used. It involves extending an input sequence with a padding token so it matches the length of the lengthiest input in that batch. Padding is the preferred method for this assignment. In numerous network configurations, masking is also required to exclude the additional padding tokens, ensuring network and loss computations remain consistent as if the sentence was processed without padding.

Example: A preprocessed sentence like [12, 1, 5, 6] with a maximum length of 10 in a batch and a padding id of 0 becomes: [12, 1, 5, 6, 0, 0, 0, 0, 0, 0].

PyTorch functions like pad_sequence(), pack_padded_sequence(), and pad_packed_sequence() (found in torch.nn.utils.rnn) can be used for padding.

Logistic Regression Model:

I implemented a logistic regression model for binary classification, which consists of:

An Embedding layer that converts input text into a sequence of vectors.
A Linear output layer applied to the vector to get a label.
Hyperparameters like training epochs, batch size, optimization method choice, loss function choice, dropout rate, L2 regularization strength, and learning rate were considered.

LSTM Feature Extractor Model:

I built a classifier model that integrates an LSTM layer with word embeddings. This model averages the pooling on the resulting hidden state vectors and forwards the vector to a feed-forward neural network for prediction. The LSTM layer was introduced between the embedding and linear output layers. In PyTorch, the embeddings are packed using packed_padded_sequence before passing them to the LSTM, enabling the LSTM to only deal with the non-padded elements. After processing, the LSTM gives back the packed output and the last hidden and cell states, which are then unpacked using pad_packed_sequence.

Just as with the logistic regression model, hyperparameter decisions need consideration, especially tuning the batch size and learning rate.




