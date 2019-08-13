import pickle
import matplotlib.pyplot as plt
import numpy as np
from visualization import plot_confusion_matrix, save
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
LABELS = ['right_standing', 'wrong_standing', 'right_turning', 'wrong_turning']


def main(num):
    pkl_file = open('features_vector.pkl', 'rb')
    ts_features, y = pickle.load(pkl_file)
    pkl_file.close()

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    accuracy = {}
    for name in names:
        accuracy[name] = []

    for i in range(num):
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        train_x, test_x, train_y, test_y = \
            train_test_split(ts_features, y, test_size=0.4)

        for name, clf in zip(names, classifiers):
            clf.fit(train_x, train_y)
            score = clf.score(test_x, test_y)
            if name == 'Decision Tree' or name == 'Naive Bayes':
                pred_y = clf.predict(test_x)
                plt.figure()
                plot_confusion_matrix([pred_y], [test_y], LABELS)
                save(name + '_confusion_matrix_' + str(i) + '.png')
            # print(name, ': ', score)
            accuracy[name].append(score)

    print(accuracy)
    for name in names:
        print(name+': ', np.array(accuracy[name]).mean())

if __name__ == '__main__':
    main(5)