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
import random
from  sklearn.datasets import load_svmlight_file
#start = time.clock()
datasize = 'openset'
forestsize = 40
n_classes = 4
n_validate = 3
min_threshold = 1
average_threshold = 1
product_threshold = 1

X_train = None
y_train = None
X_validate = [None] * (n_validate + n_classes)
y_validate = [None] * (n_validate + n_classes)
X_test = [None] * 10
y_test = [None] * 10
train_classes = []
validate_classes = []
    
#file = open("../Data/tester.data",'r')
file2 = open("_forestsize_" +str(forestsize) + "_n_classes_" + str(n_classes) + "_n_validate_" + str(n_validate) + ".csv", 'w')
#f = BytesIO(file.read())
#Xy = np.genfromtxt(f, delimiter=',')
def load():
    global X_train
    global y_train
    global X_validate
    global y_validate
    global X_test
    global y_test
    global train_classes
    global validate_classes
    
    X_train = None
    y_train = None
    X_validate = [None] * (10)
    y_validate = [None] * (10)
    X_test = [None] * 10
    y_test = [None] * 10
    train_classes = []
    validate_classes = []
    count = n_classes
    while count > 0:
        rand_class = random.randrange(10)
        if rand_class not in train_classes:
            train_classes.append(rand_class)
            count -= 1
    count = n_validate
    while count > 0:
        rand_class = random.randrange(0,9)
        if rand_class not in train_classes and rand_class not in validate_classes:
            validate_classes.append(rand_class)
            count -= 1
    
    for i in train_classes:
        X_temp, y_temp = load_svmlight_file("../Data/mnist/mnist_train_" + str(i))
        X_temp = X_temp.todense()
        X_temp = np.asarray(X_temp)
        if len(X_temp[0]) < 780:
            X_temp = np.hstack((X_temp, np.zeros((X_temp.shape[0], 780 - len(X_temp[0])), dtype=X_temp.dtype)))
                            
        if X_train == None:
            X_train = X_temp
            y_train = y_temp
        else:
            X_train = np.concatenate((X_train, X_temp))
            y_train = np.concatenate((y_train,y_temp))
    count = 0
    for i in train_classes + validate_classes:
        X_validate[i], y_validate[i] = load_svmlight_file("../Data/mnist/mnist_validate_" + str(i))
        X_validate[i] = X_validate[i].todense()
        y_validate[i] = np.asarray(y_validate[i])
        X_validate[i] = np.asarray(X_validate[i])
        if len(X_validate[i][0]) < 780:
            X_validate[i] = np.hstack((X_validate[i], np.zeros((X_validate[i].shape[0], 780 - len(X_validate[i][0])), dtype=X_validate[i].dtype)))
        count += 1
            

    for i in range(10):
        X_test[i], y_test[i] = load_svmlight_file("../Data/mnist/mnist_test_" + str(i))
        X_test[i] = X_test[i].todense()
        y_test[i] = np.asarray(y_test[i])
        X_test[i] = np.asarray(X_test[i])
        if len(X_test[i][0]) < 780:
            X_test[i] = np.hstack((X_test[i], np.zeros((X_test[i].shape[0], 780 - len(X_test[i][0])), dtype=X_test[i].dtype)))
        
'''counter_validate = 0
counter_train = 0
X_train = []
y_train = []
X_test = [[]] * 100
X_validate = [[]] * (n_validate + n_classes)
y_test = [None] * 100
y_validate = [None] * (n_validate + n_classes)
for i in range(100):
    file = open("../Data/Generated_Data/" + str(i+1) + "_generated_class.csv",'r')
    f = BytesIO(file.read())
    temp_X = np.genfromtxt(f, delimiter=',')[1:]
    trainer_X, other_X, trainer_y, other_y = train_test_split(temp_X, [i] * len(temp_X), train_size = 0.4, test_size = 0.6)
    validation_X, testing_X, validation_y, testing_y = train_test_split(other_X, other_y, train_size = 0.5, test_size = 0.5)
    if counter_train < n_classes:
        for line in trainer_X:
            X_train.append(line)
        y_train = y_train + trainer_y
    if counter_validate < n_classes+n_validate:
        y_validate[i] = validation_y    
        X_validate[i] = validation_X
    X_test[i] = testing_X
    y_test[i] = testing_y
    counter_validate += 1
    counter_train += 1'''

'''counter = 0
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
    y_validate[i] = temp_y2'''
 
def find_threshold(counter, points_in, points_out, pertinence):
    global min_threshold
    global average_threshold
    global product_threshold
    counter -= 1
    min_position = 0
    average_position = 0
    product_position = 0
    for point in points_in:
        if pertinence[point][0] > min_threshold:
            min_position += 0
        else :
            min_position -= 1
        if pertinence[point][1] > average_threshold:
            average_position += 0
        else :
            average_position -= 1
        if pertinence[point][2] > product_threshold:
            product_position += 0
        else :
            product_position -= 1
    for point in points_out:
        if pertinence[point][0] < min_threshold:
            min_position -= 0
        else:
            min_position += 1
        if pertinence[point][1] < average_threshold:
            average_position -= 0
        else:
            average_position += 1
        if pertinence[point][2] < product_threshold:
            product_position -= 0
        else:
            product_position += 1
    if min_position < 0:
        min_threshold -= 0.005
    else:
        min_threshold += 0.005
    if average_position < 0:
        average_threshold -= 0.005
    else:
        average_threshold += 0.005
    if product_position < 0:
        product_threshold -= 0.005
    else:
        product_threshold += 0.005
    if counter != 0:
        find_threshold(counter, points_in, points_out, pertinence)
            
    
def fit(model, class_n):
    global min_threshold
    global average_threshold
    global product_threshold
    classes = []
    for i in y_train:
        if i not in classes:
            classes.append(i)
    print classes
    X_validation = None
    y_validation = None
    for i in train_classes + validate_classes:
        if X_validation == None:
            X_validation = X_validate[i]
            y_validation = y_validate[i]
        else:
            X_validation = np.vstack((X_validation, X_validate[i]))
            y_validation = np.append(y_validation, y_validate[i])

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
    min_threshold = (known_min_pertinence + unknown_min_pertinence) / 2.0
    average_threshold = (known_average_pertinence + unknown_average_pertinence) / 2.0
    product_threshold = (known_product_pertinence + unknown_product_pertinence) / 2.0
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
    fit(model, n_classes)
    global min_threshold
    global average_threshold
    global product_threshold
    classes = []
    for i in y_train:
        if i not in classes:
            classes.append(i)
    X_tests = None
    y_tests = None
    unknown = []
    for i in range(10):
        if i not in train_classes and i not in validate_classes:
            unknown.append(i)
    print train_classes
    print validate_classes
    print unknown
    print train_classes +  unknown[:test]
    for i in train_classes +  unknown[:test]:
        if X_tests == None:
            X_tests = X_test[i]
            y_tests = y_test[i]
        else:
            X_tests = np.vstack((X_tests, X_test[i]))
            y_tests = np.append(y_tests, y_test[i])
    og_score = model.score(X_tests,y_tests)
    print("random test: " + str(og_score))
    predictions, pertinence = model.evt_predict(X_tests)
    total = 0
    min_correct = 0
    average_correct = 0
    product_correct = 0
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
        if pertinence[i][0] > min_threshold:
            if predictions[i] == y_tests[i]:
                min_correct += 1
        else:
            if y_tests[i] not in classes:
                min_correct += 1
        if pertinence[i][1] > average_threshold:
            if predictions[i] == y_tests[i]:
                average_correct += 1
        else:
            if y_tests[i] not in classes:
                average_correct += 1
        if pertinence[i][2] > product_threshold:
            if predictions[i] == y_tests[i]:
                product_correct += 1
        else:
            if y_tests[i] not in classes:
                product_correct += 1
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
    EVT_min_score = float(min_correct)/total
    EVT_average_score = float(average_correct)/total
    EVT_product_score = float(product_correct)/total
    print n_classes
    print("evt score: " + str(EVT_min_score))
    return (og_score,EVT_min_score,EVT_average_score,EVT_product_score)


og_average = [[]] * 4
evt_min_average = [[]] * 4
evt_average_average = [[]] * 4
evt_product_average = [[]] * 4
for w in range(5):
    load()
    for i in range(4):
        test = runTest(datasize, forestsize, 10, i)
        file2.write(str(i) + ',')
        file2.write(str(test[0]) + ',')
        file2.write(str(test[1]) + ',')
        file2.write(str(test[2]) + ',')
        file2.write(str(test[3]) + '\n')
        og_average[i].append( test[0])
        evt_min_average[i].append(test[1])
        evt_average_average[i].append(test[2])
        evt_product_average[i].append(test[3])
'''for i in range(4):
    for j in range(5):
        file2.write(str(i) + ',')
        file2.write(str(og_average[i][j]) + ',')
        file2.write(str(evt_min_average[i][j]) + ',')
        file2.write(str(evt_average_average[i][j]) + ',')
        file2.write(str(evt_product_average[i][j]) + '\n')'''

