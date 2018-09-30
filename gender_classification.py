#Gender Classification

# importing required  modules
from sklearn import tree
from sklearn.metrics import accuracy_score

# our data set to train
X_train =[[181,80,44],[177,70,43],[160,60,38],[154,54,37],
  [166,65,40],[190,90,47],[175,64,39]]
Y_train =['male','female','female','female','male',
  'male','male']

# testing dataset

X_test = [[177,70,40],[159,55,37],
  [171,75,42],[181,85,43]]
Y_test = ['female','male','female',
  'male']

#Decision Tree

#training and predicting using our classifier

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

prediction = clf.predict(X_test)

print("accuracy score is :",accuracy_score(Y_test,prediction))

"""# ***Random forest***"""

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(X_test)

print("accuracy score is :",accuracy_score(Y_test,prediction))

"""# ***K Nearest Neighbours***"""

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 3)
clf = clf.fit(X,Y)

prediction = clf.predict(X_test)

print("accuracy score is :",accuracy_score(Y_test,prediction))

"""# ***Gaussian NB***"""

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X,Y)


prediction = clf.predict(X_test)

print("accuracy score is :",accuracy_score(Y_test,prediction))
