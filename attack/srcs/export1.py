import pandas
import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
import lightgbm as lgb
from sklearn.model_selection import train_test_split



train_df=pd.read_csv("train_fix1.csv")
test_df = pd.read_csv("test_fix1.csv")
print(train_df.info())

y_train = train_df['loan_status']
x_train = train_df.drop(['loan_status','id'], axis=1)
x_test = test_df.drop(['id'], axis=1)


model6 = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='gain', learning_rate=0.01, max_depth=-1,
        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
        n_estimators=1000, n_jobs=-1, num_leaves=31, objective=None,
        random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

result = model6.fit(x_train, y_train, eval_metric= "logloss", verbose = 50, early_stopping_rounds = 200)
y_pred1 = model6.predict(x_test, num_iteration = result.best_iteration_)
y_pred_proba = model6.predict_proba(x_test, num_iteration=result.best_iteration_)[:,1]
importance = pd.DataFrame(model6.feature_importances_, index = x_train.columns, columns=["importance"])
display(importance.sort_values("importance", ascending= False))

#混同行列の作成
matrix6 = confusion_matrix(y_test, y_pred1)
print("混同行列(LightGBM):\n{}".format(matrix6))

#適合率、再現率、F1値の計算
from sklearn.metrics import precision_score, recall_score, f1_score
print("LightRBMの適合率、再現率、F1値")
print("適合率：{:.3f}".format(precision_score(y_test, y_pred1)))
print("再現率：{:.3f}".format(recall_score(y_test, y_pred1)))
print("F1値：{:.3f}".format(f1_score(y_test, y_pred1)))

submit = pd.read_csv("submit.csv",header = None)
submit.iloc[:,1] = np.where(y_pred1 > 0.5, 1 ,0)
submit.to_csv("StackingSubmission.csv", index=False, header = None)

