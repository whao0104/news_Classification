# BBC News Classification

The text data is stored in the BBC folder. 

Each txt file is a text file.

The text data is stored in the BBC folder. Each TXT file is a text file.

Functions of the project:

1. Text preprocessing

2. Feature extraction (TF-IDF, gram, word frequency)

3. Feature selection (chi square test)

4. Model training

5. Model test

The main process is as follows:

1.Text preprocessing: Run text_prepro.py, this file contains functions to preprocess text. It label the text in a txt file, splits words, removes punctuation and numbers, and writes them to a CSV file. CSV files will be stored in a folder named dataset.

2.Run the classifier: Run classfier.py, this file includes feature extraction, feature selection, model training and model testing. When classfier.Py runs, it will from calls get_data() function and prepare_datasets() function to realise reading dataset from files and dividing data set.
