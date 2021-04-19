# This file contains functions to preprocess text. It label the text in a txt file, splits words, removes punctuation and numbers, and writes them to a CSV file.
# csv files will be stored in a folder named dataset.
import time
import os
import nltk
import csv
import re

def TextPrepro(news_label):
    os.getcwd() # Get current path
    path = os.getcwd()+'/bbc/'+news_label # Get txt files storage location
    datalist = []
    for i in os.listdir(path):
        if os.path.splitext(i)[1] == '.txt':     # Select the file with the txt as suffix to join the datalist
            datalist.append(i)

    file_paths=[] # Get txt files paths
    for txt in datalist:
        data_path = os.path.join(path,txt)
        file_paths.append(data_path)

    with open(os.getcwd()+'/dataset/'+news_label+'_dataset.csv', 'a') as csv_file:
        a=['label','content']
        writer = csv.writer(csv_file)
        writer.writerow(a)

        for path in file_paths: # Read pending text
            with open(path, "r") as f:
                text = f.read()

            remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+' # Remove numbers and punctuation
            text = re.sub(remove_chars, '', text)

            word_list = nltk.tokenize.word_tokenize(text) # Splits words

            lemmatizer = nltk.stem.WordNetLemmatizer() # Word normalization
            word_list_lemmas_lower = []
            for word in word_list:
                word_list_lemmas_lower.append(lemmatizer.lemmatize(word).lower())

            stopwords = nltk.corpus.stopwords.words('english') # Remove stopword
            for w in [ '!', ',', '.', '?', '-s', '-ly', '</s>', '"','(',')', 's','``',"''","'s"]:
                stopwords.append(w)
            filtered_words = [word for word in word_list_lemmas_lower if word not in stopwords]

            dic = {news_label:' '.join(filtered_words)} # Write to CSV file
            for key, value in dic.items():
                writer.writerow([key, value])


def main():
    start_time = time.time()
    TextPrepro('business')
    TextPrepro('entertainment')
    TextPrepro('politics')
    TextPrepro('sport')
    TextPrepro('tech')
    print('Text preprocessing complete')
    end_time = time.time()
    print('Running time:',end_time-start_time,'s')

if __name__ == '__main__':
    main()