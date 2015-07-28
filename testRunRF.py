#import sys
#sys.path.append('../RF/scikit-learn/sklearn')
import numpy as np
from io import BytesIO
from gzip import GzipFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import libmr
import time
import os
from  sklearn.datasets import load_svmlight_file
#start = time.clock()
datasize = 'openset'
forestsize = 20
n_classes = 4
threshold = 1

#file = open("../Data/tester.data",'r')
'''file2 = open("datasize_" + str(datasize) + "_forestsize_" +str(forestsize) + ".csv", 'w')
#f = BytesIO(file.read())
#Xy = np.genfromtxt(f, delimiter=',')
X_train, y_train = load_svmlight_file("../Data/letter_sample_20_files/letter_sample_training0")
X_test = [None] * 7
y_test = [None] * 7
X_test[0], y_test[0] = load_svmlight_file("../Data/letter_sample_20_files/letter_sample_testing0.t_plus_0")
X_test[1], y_test[1] = load_svmlight_file("../Data/letter_sample_20_files/letter_sample_testing0.t_plus_1")
X_test[2], y_test[2] = load_svmlight_file("../Data/letter_sample_20_files/letter_sample_testing0.t_plus_2")
X_test[3], y_test[3] = load_svmlight_file("../Data/letter_sample_20_files/letter_sample_testing0.t_plus_3")
X_test[4], y_test[4] = load_svmlight_file("../Data/letter_sample_20_files/letter_sample_testing0.t_plus_4")
X_test[5], y_test[5] = load_svmlight_file("../Data/letter_sample_20_files/letter_sample_testing0.t_plus_5")
X_test[6], y_test[6] = load_svmlight_file("../Data/letter_sample_20_files/letter_sample_testing0.t_plus_6")'''
counter = 0
X_train = []
y_train = []
for i in range(n_classes):
    file = open("../Data/CTG/CTG_train_" + str(i) + ".csv",'r')
    f = BytesIO(file.read())
    temp_X = np.genfromtxt(f, delimiter=',')
    for line in temp_X:
        X_train.append(line)
    temp_y = [i] * len(temp_X)
    y_train = y_train + temp_y
counter = 0
X_test = [[]] * 10
X_validate = [[]] * 10
y_test = [None] * 10
y_validate = [None] * 10
for i in range(10):
    index = 9 - i
    file = open("../Data/CTG/CTG_test_" + str(index) + ".csv")
    file2 = open("../Data/CTG/CTG_verify_" + str(index) + ".csv")
    f2 = BytesIO(file2.read())
    f = BytesIO(file.read())
    temp_X = np.genfromtxt(f, delimiter=',')
    temp_X2 = np.genfromtxt(f2, delimiter=',')
    X_test[i] = temp_X
    X_validate[i] = temp_X2
    temp_y = [index] * len(temp_X)
    temp_y2 = [index] * len(temp_X2)
    y_test[i] = temp_y
    y_validate[i] = temp_y2
 
def find_threshold(counter, points_in, points_out, pertinence):
    global threshold
    counter -= 1
    position = 0
    for point in points_in:
        if pertinence[point][1] > threshold:
            position += 0
        else :
            position -= 1
    for point in points_out:
        if pertinence[point][1] < threshold:
            position -= 0
        else:
            position += 1
    if position < 0:
        threshold -= 0.01
    else:
        threshold += 0.01
    if counter != 0:
        print threshold
        find_threshold(counter, points_in, points_out, pertinence)
            
    
def fit(size, trees, features, class_n):
    global threshold
    model = RandomForestClassifier(n_estimators = trees, max_features = features, min_samples_leaf = 5, oob_score = False)
    model.fit(X_train,y_train)
    classes = []
    for i in y_train:
        if i not in classes:
            classes.append(i)
    print classes
    X_validation = []
    y_validation = []
    for index in range(class_n):
        i = 9 - index
        for line in X_validate[i]:
            X_validation.append(line)
        y_validation = y_validation + y_validate[i]
    predictions, pertinence = model.evt_predict(X_validation)
    min_out = 0
    average_out = 0
    product_out = 0
    min_inn = 0
    average_inn = 0
    product_inn = 0
    counter1 = 0
    counter2 = 0
    points_in = []
    points_out = []
    for i in range(len(predictions)):
        if y_validation[i] not in classes:
            points_out.append(i)
            min_out += pertinence[i][0]
            average_out += pertinence[i][1]
            product_out += pertinence[i][2]
            counter1 += 1
        else:
            points_in.append(i)
            min_inn += pertinence[i][0]
            average_inn += pertinence[i][1]
            product_inn += pertinence[i][2]
            counter2 += 1
    known_min_pertinence = min_inn/ float(counter2)
    known_average_pertinence = average_inn/ float(counter2)
    known_product_pertinence = product_inn/ float(counter2)
    unknown_min_pertinence = min_out/ float(counter1)
    unknown_average_pertinence = average_out/ float(counter1)
    unknown_product_pertinence = product_out/ float(counter1)
    threshold = (known_average_pertinence + unknown_average_pertinence) / 2.0
    print known_average_pertinence
    print unknown_average_pertinence
    print threshold
    find_threshold(30, points_in, points_out, pertinence)
    
    
def runTest(size, trees, features, test):
    '''for i in range(10):

        foldX_train, foldX_test, foldy_train, foldy_test = train_test_split(X,y)
    
        print("train size: " + str(foldy_train.size))
        print("test size: " + str(foldy_test.size))
    
        model = RandomForestClassifier(n_estimators = trees, max_features = features, min_samples_leaf = 5, oob_score = False)
        
        model.fit(foldX_train,foldy_train)
        for i in y_train:
            if i not in classes:
                classes.append(i)
        model.evt_predict(X_test[test])'''
    model = RandomForestClassifier(n_estimators = trees, max_features = features, min_samples_leaf = 5, oob_score = False)
    model.fit(X_train,y_train)
    classes = []
    for i in y_train:
        if i not in classes:
            classes.append(i)
    print classes
    X_tests = []
    y_tests = []
    for index in range(test):
        i = 9 - index
        for line in X_test[i]:
            X_tests.append(line)
        y_tests = y_tests + y_test[i]
    og_score = model.score(X_tests,y_tests)
    print len(y_tests)
    print("random test: " + str(og_score))
    predictions, pertinence = model.evt_predict(X_tests)
    total = 0
    correct = 0
    min_out = 0
    average_out = 0
    product_out = 0
    min_inn = 0
    average_inn = 0
    product_inn = 0
    counter1 = 0
    counter2 = 0
    points_in = []
    points_out = []
    for i in range(len(predictions)):
        total += 1
        if pertinence[i][1] > threshold:
            if predictions[i] == y_tests[i]:
                correct += 1
        else:
            if y_tests[i] not in classes:
                correct += 1
        if y_tests[i] not in classes:
            points_out.append(i)
            min_out += pertinence[i][0]
            average_out += pertinence[i][1]
            product_out += pertinence[i][2]
            counter1 += 1
        else:
            points_in.append(i)
            min_inn += pertinence[i][0]
            average_inn += pertinence[i][1]
            product_inn += pertinence[i][2]
            counter2 += 1
    if counter2 > 0:
        min_pertinence = min_inn/ float(counter2)
        average_pertinence = average_inn/ float(counter2)
        product_pertinence = product_inn/ float(counter2)
        min_deviance = 0
        average_deviance = 0
        product_deviance = 0
        for i in points_out:
            min_deviance += (pertinence[i][0] - min_pertinence) ** 2
            average_deviance += (pertinence[i][1] - average_pertinence) ** 2
            product_deviance += (pertinence[i][2] - product_pertinence) ** 2
        min_deviance /= len(points_out) - 1
        average_deviance /= len(points_out) - 1
        product_deviance /= len(points_out) -1
        min_deviance = min_deviance ** 0.5
        average_deviance = average_deviance ** 0.5
        product_deviance = product_deviance ** 0.5
        print "average kown classes with min: " + str(min_pertinence) + " standard deviation: " + str(min_deviance)
        print "average kown classes with average: " + str(average_pertinence) + " standard deviation: " + str(average_deviance)
        print "average kown classes with product: " + str(product_pertinence) + " standard deviation: " + str(product_deviance)
    if counter1 > 0:
        min_pertinence = min_out/ float(counter1)
        average_pertinence = average_out/ float(counter1)
        product_pertinence = product_out/ float(counter1)
        min_deviance = 0
        average_deviance = 0
        product_deviance = 0
        for i in points_out:
            min_deviance += (pertinence[i][0] - min_pertinence) ** 2
            average_deviance += (pertinence[i][1] - average_pertinence) ** 2
            product_deviance += (pertinence[i][2] - product_pertinence) ** 2
        min_deviance /= len(points_out) - 1
        average_deviance /= len(points_out) - 1
        product_deviance /= len(points_out) -1
        min_deviance = min_deviance ** 0.5
        average_deviance = average_deviance ** 0.5
        product_deviance = product_deviance ** 0.5
        print "       average unknown classes with min: " + str(min_pertinence) + " standard deviation: " + str(min_deviance)
        print "       average unknown classes with average: " + str(average_pertinence) + " standard deviation: " + str(average_deviance)
        print "       average unknown classes with product: " + str(product_pertinence) + " standard deviation: " + str(product_deviance)
    EVT_score = float(correct)/total
    print("evt score: " + str(EVT_score))
    return (og_score,EVT_score)

og_average = 0
evt_average = 0
fit(datasize, forestsize, 10, 6)
print threshold
for i in range(4,10):
    test = runTest(datasize, forestsize, 10, i)
    og_average += test[0]
    evt_average += test[1]
    og = test[0]
    evt = test[1]
    '''   file2.write(str(10*i) + ',')
    file2.write(str(og) + ',')
    file2.write(str(evt) + '\n')'''
og_average /= 6.0
evt_average /= 6.0
print "original: " + str(og_average)
print "EVT: " + str(evt_average)
