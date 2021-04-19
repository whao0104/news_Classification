# get_data() will read the result of text preprocessing in the csv file and form a data set.
# prepare_datasets() will divided dataset into training set, development set and test set

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_data():
    path = os.path.abspath('dataset') # Get dataset file path
    filename_extenstion = '.csv'
    file_allname = []
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == filename_extenstion:
            t = os.path.splitext(filename)[0]
            file_allname.append(t + filename_extenstion)
    data_list = []
    for fileitem in file_allname:
        tmp = pd.read_csv(path+'/' + fileitem)
        data_list.append(tmp)

    dataset = pd.concat(data_list,ignore_index = False) # Save dataset labels and data to list
    col = dataset.columns.values.tolist()
    col1 = col[0]
    col2 = col[1]
    label = np.array(dataset[col1])
    content = np.array(dataset[col2])

    return content , label # Return: content list and label list

def prepare_datasets(content, label):
    x_train, x_test, y_train, y_test = train_test_split(content, # data
                                                        label,   # label
                                                        test_size=0.2, # 20% of the data is used as the test set
                                                        random_state=42) # Random seed
    x_train, x_development, y_train, y_development = train_test_split(x_train, # data
                                                                    y_train,   # label
                                                                    test_size=0.2, # 20% of the train set data is used as the development set
                                                                    random_state=42) # Random seed

    return x_train, x_development, x_test, y_train, y_development, y_test