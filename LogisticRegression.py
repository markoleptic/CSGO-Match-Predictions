import matplotlib.pyplot as plt
from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
import pandas as pd

data = pd.read_csv('roundMoneyWinners.csv',header=0, index_col=False)
features = ['t1_money', 't2_money', 't1_rank', 't2_rank', 'map']
X = data.loc[:, features]
y = data.loc[:,['winner']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)

model = LogisticRegression().fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
svc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()
#plt.plot(predictions, y_test)
#plt.show()