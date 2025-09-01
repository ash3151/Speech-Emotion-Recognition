Identify the emotion of a given speech using this trained Deep-learning model. Given speech can be classified into anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral.

# Preparing dataset for training
- Audio files were loaded using Librosa package.
- From audio, mfcc's were extracted using `librosa.feature.mfcc()`.
- These mfcc's are used for training the model.

# Model Architecture
- It is a sequential model containing LSTM followed by 3 dense layers
- To reduce the overfitting observed I used Dropout layers with probability of 0.2 in sequence.
- Model has over 300K trainable parameters and 0 non-trainable parameters.

Dataset is taken from Kaggle `https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess`, here you can find complete information about the dataset.


