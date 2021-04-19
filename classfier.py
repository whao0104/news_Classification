# This file contains the main project of classification work

from sklearn.linear_model import SGDClassifier
from get_data_from_csv import get_data, prepare_datasets
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report

def main():
    corpus, labels = get_data() # Get data
    print("data number：", len(labels))
    train_corpus, development_corpus,test_corpus, train_labels,development_labels ,test_labels = prepare_datasets(corpus,labels) # Divided into training set, development set and test set
    print('train_corpus number:',len(train_corpus))
    print('development_corpus number:',len(development_corpus))
    print('test_corpus number:',len(test_corpus))

    # FeatureUnion is used for feature extraction.
    features_select = FeatureUnion([('tf',CountVectorizer(ngram_range=(1,1))), # term frequency
                                    ('bow,',CountVectorizer(ngram_range=(2,2))), # Bag-of-words model(Divide every two words)
                                    ('tfidf',TfidfVectorizer())]) # TF-IDF
    # Feature extraction of all data set
    train_feature = features_select.fit_transform(train_corpus)
    development_feature = features_select.transform(development_corpus)
    test_feature = features_select.transform(test_corpus)

    # SVM Model
    svm = SGDClassifier(loss='hinge', n_iter_no_change=100)

    # Feature selection using development set
    k_list=[500,1000,2000,5000]
    for k in k_list:
        fs_sentanalysis = SelectKBest(chi2, k=k).fit(train_feature, train_labels)
        new_train_feature = fs_sentanalysis.transform(train_feature)
        new_development_feature = fs_sentanalysis.transform(development_feature)
        new_test_feature = fs_sentanalysis.transform(test_feature)
        print('k=',k,"SVM model")
        svm_model = svm.fit(new_train_feature, train_labels)
        label_pred = svm_model.predict(new_development_feature)
        score = classification_report(development_labels, label_pred, target_names=['business', 'entertainment', 'politics', 'sprot','tech'])
        print(score)

    # test model
    test_k = int(input("Please input the K value with the best training effect:"))
    print('The score of the model on the test set：')
    fs_sentanalysis = SelectKBest(chi2, k=test_k).fit(train_feature, train_labels)
    new_train_feature = fs_sentanalysis.transform(train_feature)
    new_test_feature = fs_sentanalysis.transform(test_feature)
    svm_model = svm.fit(new_train_feature, train_labels)
    label_pred = svm_model.predict(new_test_feature)
    score_1 = classification_report(test_labels, label_pred, target_names=['business', 'entertainment', 'politics', 'sprot', 'tech'])
    print(score_1)

if __name__ == '__main__':
    main()