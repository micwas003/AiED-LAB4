# Algorytmy i Eksploracja Danych - Laboratorium 4 - Zadanie 1 i 2
import pandas as pd

import time
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.decomposition import PCA

import csv


# Zadanie 1 - ladowanie danych
def ReadDataSet():
    print("Zadanie 1....")
    print("Wczytywanie danych...")
    global Y_train, X_train, Y_test, X_test
    Y_train = pd.read_csv("dataset/train/y_train.txt", sep=' ', header=None)
    X_train = pd.read_csv("dataset/train/X_train.txt", sep=' ', header=None)
    Y_test = pd.read_csv("dataset/test/y_test.txt", sep=' ', header=None)
    X_test = pd.read_csv("dataset/test/X_test.txt", sep=' ', header=None)
    print("Wczytywanie danych zakonczone!")
    print()

   # print('Dataset Y train: ')
   # print(Y_train)
   # print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
   # print('Dataset X train: ')
   # print(X_train)
   # print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
   # print('Dataset Y test: ')
   # print(Y_test)
   # print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
   # print('Dataset X test: ')
   # print(X_test)


# Zadanie 2 - Redukcja Wielowymiarowości
def ReductionMultidimensionality():
    print("Zadanie 2....")
    print('Przeprowadzenie testu bez redukcji wielowymiarowosci...')
    start_time = time.process_time()
    classificationSVM = svm.SVC(probability=True)
    classificationSVM.fit(X_train, Y_train.values.ravel())
    time_train_without_reduction = time.process_time() - start_time
    start_time = time.process_time()
    top_score_without_reduction = cross_val_score(classificationSVM, X_test, Y_test.values.ravel(), cv=5)
    time_test_without_reduction = time.process_time() - start_time
    score_without_reduction = top_score_without_reduction.mean()
    print("Time train without redunction: {:.2f}".format(time_train_without_reduction))
    print("-.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.-")
    print("Time test without redunction: {:.2f}".format(time_test_without_reduction))
    print("-.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.-")
    print("Score value without redunction: {:.2f}".format(score_without_reduction))
    print()

    print('Przeprowadzenie testu z redukcja wielowymiarowosci metoda PCA...')
    pca_method = PCA(n_components=10)
    X_train_with_PCA = pd.DataFrame(pca_method.fit_transform(X_train))
    start_time = time.process_time()
    classificationSVM = svm.SVC(probability=True)
    classificationSVM.fit(X_train_with_PCA, Y_train.values.ravel())
    time_train_with_PCA = time.process_time() - start_time
    X_test_with_PCA = pd.DataFrame(pca_method.fit_transform(X_test))
    start_time = time.process_time()
    top_score_with_PCA = cross_val_score(classificationSVM, X_test_with_PCA, Y_test.values.ravel(), cv=5)
    time_test_with_PCA = time.process_time() - start_time
    score_with_PCA = top_score_with_PCA.mean()
    print("Time train with reduction method - PCA: {:.2f}".format(time_train_with_PCA))
    print("-.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.-")
    print("Time test with reduction method - PCA: {:.2f}".format(time_test_with_PCA))
    print("-.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.--.-.-.-.-")
    print("Score value with reduction method - PCA: {:.2f}".format(score_with_PCA))
    print()

    # Zapisywanie wynikow do pliku CSV
    print('Zapisywanie wynikow do pliku CSV...')
    file_name = 'dim_reduction.csv'
    with open(file_name, 'w', newline='') as f:
        fieldnames = ['Train time (before reduction)', 'Test time (before reduction', 'Score (before reduction)', 'Train time (after PCA reduction)', 'Test time (after PCA reduction)', 'Score (after PCA reduction)']
        thewriter = csv.DictWriter(f, fieldnames=fieldnames)

        thewriter.writeheader()
        thewriter.writerow({'Train time (before reduction)' : round(time_train_without_reduction, 2), 'Test time (before reduction' : round(time_test_without_reduction, 2), 'Score (before reduction)' : round(score_without_reduction, 2), 'Train time (after PCA reduction)' : round(time_train_with_PCA, 2), 'Test time (after PCA reduction)' : round(time_test_with_PCA, 2), 'Score (after PCA reduction)' : round(score_with_PCA, 2)})
    print('Zapisywanie danych do pliku ', file_name, ' - zakonczone!')

if __name__ == "__main__":
    ReadDataSet()
    ReductionMultidimensionality()