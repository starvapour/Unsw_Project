import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def create_dataset():
    X, y = make_classification( n_samples=1250,
                                n_features=2,
                                n_redundant=0,
                                n_informative=2,
                                random_state=5,
                                n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 3 * rng.uniform(size = X.shape)
    linearly_separable = (X, y)
    X = StandardScaler().fit_transform(X)
    return X, y



def plotter(classifier, X, X_test, y_test, title, ax=None):
    # plot decision boundary for given classifier
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), 
                            np.arange(y_min, y_max, plot_step)) 
    Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax:
        ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        ax.set_title(title)
    else:
        plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        plt.title(title)

X, y = create_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.metrics import accuracy_score
train_sizes = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
train_subs = []
# create sub_train_sets
for size in train_sizes:
    if size != 1000:
        X_train_sub, drop_test, y_train_sub, drop_test = train_test_split(X_train, y_train, train_size=size)
        train_subs.append((size, X_train_sub, y_train_sub))
    else:
        train_subs.append((size, X_train, y_train))

print("start decision tree")
decision_accs = []
decision_times = []
# decision tree
for (size, X_train_sub, y_train_sub) in train_subs:
    repeat_accs = []
    repeat_time = []
    for i in range(10):
        start = time.time()
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train_sub,y_train_sub)
        repeat_accs.append(accuracy_score(decision_tree.predict(X_test), y_test))
        end = time.time()
        repeat_time.append(end-start)
    mean_acc = np.mean(np.array(repeat_accs))
    mean_time = np.mean(np.array(repeat_time))
    decision_accs.append(mean_acc)
    decision_times.append(mean_time)

print("start random forest")
random_forest_accs = []
random_forest_times = []
# random forest
for (size, X_train_sub, y_train_sub) in train_subs:
    repeat_accs = []
    repeat_time = []
    for i in range(10):
        start = time.time()
        random_forest = RandomForestClassifier()
        random_forest.fit(X_train_sub,y_train_sub)
        repeat_accs.append(accuracy_score(random_forest.predict(X_test), y_test))
        end = time.time()
        repeat_time.append(end-start)
    mean_acc = np.mean(np.array(repeat_accs))
    mean_time = np.mean(np.array(repeat_time))
    random_forest_accs.append(mean_acc)
    random_forest_times.append(mean_time)

print("start Ada Boost")
ada_boost_accs = []
ada_boost_times = []
# Ada Boost
for (size, X_train_sub, y_train_sub) in train_subs:
    repeat_accs = []
    repeat_time = []
    for i in range(10):
        start = time.time()
        ada_boost = AdaBoostClassifier()
        ada_boost.fit(X_train_sub,y_train_sub)
        repeat_accs.append(accuracy_score(ada_boost.predict(X_test), y_test))
        end = time.time()
        repeat_time.append(end-start)
    mean_acc = np.mean(np.array(repeat_accs))
    mean_time = np.mean(np.array(repeat_time))
    ada_boost_accs.append(mean_acc)
    ada_boost_times.append(mean_time)
    
print("start Logistic Regression")
log_re_accs = []
log_re_times = []
# Logistic Regression
for (size, X_train_sub, y_train_sub) in train_subs:
    repeat_accs = []
    repeat_time = []
    for i in range(10):
        start = time.time()
        log_re = LogisticRegression()
        log_re.fit(X_train_sub,y_train_sub)
        repeat_accs.append(accuracy_score(log_re.predict(X_test), y_test))
        end = time.time()
        repeat_time.append(end-start)
    mean_acc = np.mean(np.array(repeat_accs))
    mean_time = np.mean(np.array(repeat_time))
    log_re_accs.append(mean_acc)
    log_re_times.append(mean_time)

print("start Neural Network")
nn_accs = []
nn_times = []
# Neural Network
for (size, X_train_sub, y_train_sub) in train_subs:
    repeat_accs = []
    repeat_time = []
    for i in range(10):
        start = time.time()
        nn = MLPClassifier(max_iter=2000)
        nn.fit(X_train_sub,y_train_sub)
        repeat_accs.append(accuracy_score(nn.predict(X_test), y_test))
        end = time.time()
        repeat_time.append(end-start)
    mean_acc = np.mean(np.array(repeat_accs))
    mean_time = np.mean(np.array(repeat_time))
    nn_accs.append(mean_acc)
    nn_times.append(mean_time)

print("start SVM")
SVM_accs = []
SVM_times = []
# SVM
for (size, X_train_sub, y_train_sub) in train_subs:
    repeat_accs = []
    repeat_time = []
    for i in range(10):
        start = time.time()
        SVM = SVC()
        SVM.fit(X_train_sub,y_train_sub)
        repeat_accs.append(accuracy_score(SVM.predict(X_test), y_test))
        end = time.time()
        repeat_time.append(end-start)
    mean_acc = np.mean(np.array(repeat_accs))
    mean_time = np.mean(np.array(repeat_time))
    SVM_accs.append(mean_acc)
    SVM_times.append(mean_time)

# acc plot
l1, =plt.plot(train_sizes,decision_accs,color = "blue")
l2, =plt.plot(train_sizes,random_forest_accs,color = "yellow")
l3, =plt.plot(train_sizes,ada_boost_accs,color = "green")
l4, =plt.plot(train_sizes,log_re_accs,color = "red")
l5, =plt.plot(train_sizes,nn_accs,color = "fuchsia")
l6, =plt.plot(train_sizes,SVM_accs,color = "darkgoldenrod")
plt.legend(handles=[l1,l2,l3,l4,l5,l6],labels=["Decision Tree","Random Forest","Ada Boost","Logistic Regression", "Neural Network", "SVM"])
plt.title("accuracy")
plt.savefig("accuracy.png")
plt.show()

# time plot
l1, =plt.plot(train_sizes,decision_times,color = "blue")
l2, =plt.plot(train_sizes,random_forest_times,color = "yellow")
l3, =plt.plot(train_sizes,ada_boost_times,color = "green")
l4, =plt.plot(train_sizes,log_re_times,color = "red")
l5, =plt.plot(train_sizes,nn_times,color = "fuchsia")
l6, =plt.plot(train_sizes,SVM_times,color = "darkgoldenrod")
plt.legend(handles=[l1,l2,l3,l4,l5,l6],labels=["Decision Tree","Random Forest","Ada Boost","Logistic Regression", "Neural Network", "SVM"])
plt.title("time")
plt.savefig("time.png")
plt.show()
