import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

# load and organize dataset
# get x values and get y Values
# get only the data points of two classes, setosa, and versicolor
# so until index 0 to 99 only
df_raw = datasets.load_iris()
df = pd.concat([pd.DataFrame(df_raw['data'][:100, :], columns=['sepal length', 'sepal width', 'petal length', 'petal width']), 
    pd.DataFrame({'y': df_raw['target'][:100]})], axis=1)
print(df)

# note that the 0, 1, and 2 classes are setosa, versicolor, and virginica
# the three classes, in this case only classes setosa and versicolor are used
X, y = df[['sepal length', 'sepal width', 'petal length', 'petal width']], df['y']
print(X, X.shape)
print(y, y.shape)

# plot the features x_1, x_2, and x_3 on the x, y, z axis of a 3d
# cartesian plane respectively
fig_1 = plt.figure(figsize=(7, 5))
ax_1 = fig_1.add_subplot(111, projection='3d')
ax_1.scatter(X['sepal length'][:50], X['sepal width'][:50], X['petal length'][:50], c='#4d17ff', label='setosa')
ax_1.scatter(X['sepal length'][50:100], X['sepal width'][50:100], X['petal length'][50:100], c='#0fffa7', label='versicolor')

# labeling axes with X_1, X_2, X_3
ax_1.set_xlabel("X\u2081")
ax_1.set_ylabel("X\u2082")
ax_1.set_zlabel("X\u2083")

plt.legend()
plt.show()



# figure 2
fig_2 = plt.figure(figsize=(7, 5))
ax_2 = fig_2.add_subplot()
ax_2.scatter(X['sepal length'][:50], X['sepal width'][:50], c='#4d17ff', label='setosa')
ax_2.scatter(X['sepal length'][50:100], X['sepal width'][50:100], c='#0fffa7', label='versicolor')

# labeling axes with X_1, X_2, X_3
ax_2.set_xlabel("X\u2081")
ax_2.set_ylabel("X\u2082")

plt.legend()
plt.show()



# instantiate model
lr = LogisticRegression()

# split dataset into a 80 to 20 percent ratio of training and testing
X_trains, X_tests, y_trains, y_tests = train_test_split(X, y, test_size=0.20, random_state=0)
print(X_trains, X_trains.shape, end="\n")
print(y_trains, y_trains.shape, end="\n")
print(X_tests, X_tests.shape, end="\n")
print(y_tests, y_tests.shape, end="\n")

# train model
model = lr.fit(X_trains, y_trains)

# extract optimized coefficients/parameters
theta_0 = model.intercept_[0]
theta_1, theta_2, theta_3, theta_4 = model.coef_[0]
print("final model coefficients: {} {} {} {}".format(theta_1, theta_2, theta_3, theta_4))
print("final model bias coefficient: ", theta_0)

# predict y values based on test data set X
y_pred = model.predict(X_tests)
print(y_pred, len(y_pred), end='\n')
y_pred_2 = model.predict(X_trains)
print(y_pred_2, len(y_pred_2), end='\n')

# measure its accuracy by score and matrix
cm = confusion_matrix(y_tests, y_pred)
print(cm)
accuracy = accuracy_score(y_tests, y_pred)

# figure 3
fig_3, ax_3 = plt.subplots()
im = ax_3.imshow(cm, cmap='magma')

# Show all ticks and label them with the respective list entries
ax_3.set_xticks(np.arange(cm.shape[0]), labels=unique_labels(y))
ax_3.set_yticks(np.arange(cm.shape[1]), labels=unique_labels(y))

# Rotate the tick labels and set their alignment.
plt.setp(ax_3.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text = ax_3.text(j, i, cm[i, j], ha="center", va="center", color="#3274bf")

ax_3.set_title("correctly classified")
fig_3.tight_layout()
plt.show()



# figure 4
fig_4 = plt.figure(figsize=(7, 5))
ax_4 = fig_4.add_subplot()

# pass the train and test datapoints with setosa label
ax_4.scatter(X['sepal length'][:50], X['sepal width'][:50], c='#4d17ff', label='setosa')
# ax_4.scatter(X['sepal length'][:50], X['sepal width'][:50], c='#4d17ff', label='setosa')

# plot the train and test datapoints with versicolor label
ax_4.scatter(X['sepal length'][50:100], X['sepal width'][50:100], c='#0fffa7', label='versicolor')
# ax_4.scatter(X['sepal length'][50:100], X['sepal width'][50:100], c='#0fffa7', label='versicolor')

# draw initial line
sample = np.linspace(-50, 50, 100)
init_equation = lambda x: -(3 * x) + 4

# when 0.25 is placed instead of theta_0 the decision boundary
# separates the data points properly but when it is the latter
# because of the -6.362366033996434
final_equation = lambda x: (-(theta_1) / theta_2) * x - 0.25 / theta_2

# get the minimum and maximum of x1 which is along the x axis
xmax, xmin = max(X['sepal length'][:100]), min(X['sepal length'][:100])

# draw final line
ax_4.plot([xmin, xmax], final_equation(np.array([xmin, xmax])), c='#c40a83', label='final decision boundary')

# labeling axes with X_1, X_2, X_3
ax_4.set_xlabel("X\u2081")
ax_4.set_ylabel("X\u2082")

plt.legend()
plt.show()



    





