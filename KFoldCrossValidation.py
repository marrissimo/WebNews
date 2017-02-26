from sklearn.cross_validation import KFold
import numpy as np


file = open('Dataset_contenuti_TRUE.txt')

testo = file.read()
articles = np.array(testo.split('\n'))
for i in range(0,len(articles)):
    articles[i]+='\n'
count = len(articles)
file.close()

kf = KFold(count, n_folds=3)


count=0
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = articles[train_index], articles[test_index]

    train_string=""
    test_string=""
    for i in X_train:
        train_string += i
    for j in X_test:
        test_string += j

    out_file = open("./articoli/Train_"+str(count)+"_TRUE.txt", "w")
    out_file.write(train_string)
    out_file.close()

    out_file = open("./articoli/Test_"+str(count)+"_TRUE.txt", "w")
    out_file.write(test_string)
    out_file.close()
    count+=1

