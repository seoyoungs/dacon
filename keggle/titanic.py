import numpy as np
import pandas as pd
import timeit
import tensorflow as tf

import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.simplefilter('ignore')

from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

from mlxtend.classifier import StackingCVClassifier
import shap

RANDOM_SEED = 2021
PROBAS = True
FOLDS = 5
N_ESTIMATORS = 1000

TARGET = 'Survived'

train = pd.read_csv('C:/data/keggle/taita/train.csv')
test = pd.read_csv('C:/data/keggle/taita/test.csv')
submission = pd.read_csv('C:/data/keggle/taita/sample_submission.csv')

# Pseudo labels taken from great BIZEN notebook: https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble
pseudo_labels = pd.read_csv("C:/data/keggle/taita/sub_top.csv")
test[TARGET] = pseudo_labels[TARGET]

train['FirstName'] = train['Name'].apply(lambda x:x.split(', ')[0])
train['n'] = 1
gb = train.groupby('FirstName')
df_names = gb['n'].sum()
train['SameFirstName'] = train['FirstName'].apply(lambda x:df_names[x])

test['FirstName'] = test['Name'].apply(lambda x:x.split(', ')[0])
test['n'] = 1
gb = test.groupby('FirstName')
df_names = gb['n'].sum()
test['SameFirstName'] = test['FirstName'].apply(lambda x:df_names[x])


all_df = pd.concat([train, test], axis=0)

all_df['AnyMissing'] = np.where(all_df.isnull().any(axis=1) == True, 1, 0)


all_df['FamilySize'] = all_df['SibSp'] + all_df['Parch'] + 1
all_df['IsAlone'] = np.where(all_df['FamilySize'] <= 1, 1, 0)


all_df['Has_Cabin'] = all_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())
cabin_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
             'F': 6, 'G': 7, 'T': 1, 'X': 8}
all_df['Cabin'] = all_df['Cabin'].str[0].fillna('X').replace(cabin_map)

all_df['Embarked'] = all_df['Embarked'].fillna("No")
conditions = [
    (all_df['Embarked']=="S"),
    (all_df['Embarked']=="Q"),
    (all_df['Embarked']=="C"),
    (all_df['Embarked']=="No")
]
choices = [0, 1, 2, -1]
all_df["Embarked"] = np.select(conditions, choices)
all_df['Embarked'] = all_df['Embarked'].astype(int)


all_df['SecondName'] = all_df.Name.str.split(', ', 1, expand=True)[1] # to try
all_df['IsFirstNameDublicated'] = np.where(all_df.FirstName.duplicated(), 1, 0)


all_df['Fare'] = all_df['Fare'].fillna(train['Fare'].median())

conditions = [
    (all_df['Fare'] <= 7.91),
    ((all_df['Fare'] > 7.91) & (all_df['Fare'] <= 14.454)),
    ((all_df['Fare'] > 14.454) & (all_df['Fare'] <= 31)),
    (all_df['Fare'] > 31)
]

choices = [0, 1, 2, 3]
all_df["Fare"] = np.select(conditions, choices)
all_df['Fare'] = all_df['Fare'].astype(int)


all_df['Ticket'] = all_df.Ticket.str.replace('\.','', regex=True).\
                    str.replace('(\d+)', '', regex=True).\
                    str.replace(' ', '', regex=True).\
                    replace(r'^\s*$', 'X', regex=True).\
                    fillna('X')

 
conditions = [
    ((all_df.Sex=="female")&(all_df.Pclass==1)&(all_df.Age.isnull())),
    ((all_df.Sex=="male")&(all_df.Pclass==1)&(all_df.Age.isnull())),
    ((all_df.Sex=="female")&(all_df.Pclass==2)&(all_df.Age.isnull())),
    ((all_df.Sex=="male")&(all_df.Pclass==2)&(all_df.Age.isnull())),
    ((all_df.Sex=="female")&(all_df.Pclass==3)&(all_df.Age.isnull())),
    ((all_df.Sex=="male")&(all_df.Pclass==3)&(all_df.Age.isnull()))]


choices = all_df[['Age', 'Pclass', 'Sex']].\
            dropna().\
            groupby(['Pclass', 'Sex']).\
            mean()['Age']

all_df["Age"] = np.select(conditions, choices)

conditions = [
    (all_df['Age'].le(16)),
    (all_df['Age'].gt(16) & all_df['Age'].le(32)),
    (all_df['Age'].gt(32) & all_df['Age'].le(48)),
    (all_df['Age'].gt(48) & all_df['Age'].le(64)),
    (all_df['Age'].gt(64))
]
choices = [0, 1, 2, 3, 4]

all_df["Age"] = np.select(conditions, choices)
all_df['Sex'] = np.where(all_df['Sex']=='male', 1, 0)
all_df = all_df.drop(['Name', 'n'], axis = 1)

# ========== 4B. Encoding data ==================
label_cols = ['Ticket']

def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)

scaler = StandardScaler()
all_df.Ticket = all_df[['Ticket']].apply(label_encoder)

features_selected = ['Pclass', 'Sex', 'Age','Embarked','Parch','SibSp','Fare','Cabin','Ticket','SameFirstName', 'Survived']
all_df = all_df[features_selected]

# =========== 4C. Create train and test datasets =================
X = all_df.drop([TARGET], axis = 1)
y = all_df[TARGET]

print (f'X:{X.shape} y: {y.shape} \n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = RANDOM_SEED)
print (f'X_train:{X_train.shape} y_train: {y_train.shape}')
print (f'X_test:{X_test.shape} y_test: {y_test.shape}')

test = all_df[len(train):].drop([TARGET], axis = 1)
print (f'test:{test.shape}')

# ============== 5. Declare list of classifiers (level 1) for testing =========
lgb_params = {
    'metric': 'binary_logloss',
    'n_estimators': 10000,
    'objective': 'binary',
    'learning_rate': 0.02,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
    'device': 'gpu'
}

cb_params = {
    'max_depth':6,
    'max_ctr_complexity': 5,
    'num_trees': 50000,
    'od_wait': 500,
    'od_type':'Iter', 
    'learning_rate': 0.04,
    'min_data_in_leaf': 3,
    'task_type': 'GPU'
}


rf_params = {
    'max_depth': 15,
    'min_samples_leaf': 8,
    'random_state': RANDOM_SEED
}

cl1 = KNeighborsClassifier(n_neighbors = 1)
cl2 = RandomForestClassifier(**rf_params)
cl3 = GaussianNB()
cl4 = DecisionTreeClassifier(max_depth = 5)
cl5 = CatBoostClassifier(task_type = 'GPU', verbose = None, logging_level = 'Silent')
cl6 = LGBMClassifier(device = 'gpu')


# I used some hyperparameter search (ExtraTrees - Genetic search)
cl7 = ExtraTreesClassifier(bootstrap=False, criterion='entropy', max_features=0.55, min_samples_leaf=8, min_samples_split=4, n_estimators=100) # Optimized using TPOT
cl8 = MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = RANDOM_SEED)

# =============== 6. Declare classifiers (leval 1 and level 2) ============
# Read next sections - I took only 5 most promising classifier to speed up learning process 
classifiers = {
    "RandomForest": cl2,
    "DecisionTree": cl4,
    "CatBoost": cl5,
    "LGBM": cl6,
    "ExtraTrees": cl7,
}

mlr = LogisticRegression()

# ================== 7. Train classifiers ================
models_scores_results, models_names = list(), list() 

# This step could take some time .... it depends on classifiers you use .... So make a coffe or meditate ... 

print(">>>> Training started <<<<")
for key in classifiers:
    classifier = classifiers[key]
    scores = model_selection.cross_val_score(classifier, X_train, y_train, cv = FOLDS, scoring='accuracy')
    models_scores_results.append(scores)
    models_names.append(key)
    print("[%s] - accuracy: %0.5f " % (key, scores.mean()))
    classifier.fit(X_train, y_train)
    
    # Save classifier for prediction 
    classifiers[key] = classifier

plt.boxplot(models_scores_results, labels=models_names, showmeans=True)
plt.show()
taken_classifiers = ["RandomForest", "DecisionTree", "CatBoost", "LGBM", "ExtraTrees"]

# This function searches best stacking configuration
def best_stacking_search():
    cls_list = []
    best_auc = -1
    i=0

    best_cls_experiment = list()

    print(">>>> Training started <<<<")

    for cls_comb in range(2, len(taken_classifiers)+1):
        for subset in itertools.combinations(taken_classifiers, cls_comb):
            cls_list.append(subset)

    print(f"Total number of model combination: {len(cls_list)}")


    for cls_exp in cls_list:
        cls_labels = list(cls_exp)

        classifier_exp = []
        for ii in range(len(cls_labels)):
            label = taken_classifiers[ii]
            classifier = classifiers[label]
            classifier_exp.append(classifier)


        sclf = StackingCVClassifier(classifiers = classifier_exp,
                                    shuffle = False,
                                    use_probas = True,
                                    cv = FOLDS,
                                    meta_classifier = mlr,
                                    n_jobs = -1)

        scores = model_selection.cross_val_score(sclf, X_train, y_train, cv = FOLDS, scoring='accuracy')

        if scores.mean() > best_auc:
            best_cls_experiment = list(cls_exp)
        i += 1
        print(f"  {i} - Stacked combination - Acc {cls_exp}: {scores.mean():.5f}")
        
    return best_cls_experiment

best_cls_experiment = ['RandomForest', 'DecisionTree', 'ExtraTrees', 'LGBM'] 

print(f'The best models configuration: {best_cls_experiment}')

classifier_exp = []
for label in best_cls_experiment:
        classifier = classifiers[label]
        classifier_exp.append(classifier)

classifier_exp

scl = StackingCVClassifier(classifiers= classifier_exp,
                            meta_classifier = mlr, # use meta-classifier
                            use_probas = PROBAS,   # use_probas = True/False
                            random_state = RANDOM_SEED)

scores = model_selection.cross_val_score(scl, X_train, y_train, cv = FOLDS, scoring='accuracy')
models_scores_results.append(scores)
models_names.append('scl')
print("Meta model (slc) - accuracy: %0.5f " % (scores.mean()))
scl.fit(X_train, y_train)

top_meta_model = scl
base_acc = scores.mean()

scl.get_params().keys()

def meta_best_params_search():
    
    scl_params = {'meta_classifier__C': [0.001, 0.01, 0.1, 1, 10]}

    print(">>>> Searching for best parameters started <<<<")

    grid = GridSearchCV(estimator=scl, 
                        param_grid= scl_params, 
                        cv=5,
                        refit=True)
    grid.fit(X_train, y_train)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r" % (grid.cv_results_[cv_keys[0]][r], grid.cv_results_[cv_keys[1]][r] / 2.0, grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.5f' % grid.best_score_)
    return grid, grid.best_score_

scl = StackingCVClassifier(classifiers= classifier_exp,
                            meta_classifier = LogisticRegression(C = 0.1), # use meta-classifier
                            use_probas = PROBAS,   # use_probas = True/False
                            random_state = RANDOM_SEED)

scores = model_selection.cross_val_score(scl, X_train, y_train, cv = FOLDS, scoring='accuracy')
print("Meta model (slc) - accuracy: %0.5f " % (scores.mean()))
scl.fit(X_train, y_train)
top_meta_model = scl
# SCENARIO 2.

classifiers["scl"] = top_meta_model

preds = pd.DataFrame()

for key in classifiers:
    y_pred = classifiers[key].predict_proba(X_test)[:,1]
    preds[f"{key}"] = y_pred
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(f"{key} -> AUC: {auc:.5f}")

preds[TARGET] = pd.DataFrame(y_test).reset_index(drop=True)

# ============== 9. Understand models ================
def plot_model_AUC(cls_models):
    NUM_CLASS = len(cls_models)
    sns.set(font_scale = 1)
    sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                   "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                   'ytick.color': '0.4'})

    f, ax = plt.subplots(figsize=(20, 5), nrows=1, ncols = NUM_CLASS)

    for key, counter in zip(cls_models, range(NUM_CLASS)):

        y_pred = preds[key]

        auc = metrics.roc_auc_score(y_test, y_pred)
        textstr = f"AUC: {auc:.3f}"


        false_pred = preds[preds[TARGET] == 0]
        sns.distplot(false_pred[key], hist=True, kde=True, 
                     bins=int(50), color = 'red', 
                     hist_kws={'edgecolor':'black'}, ax = ax[counter])


        true_pred = preds[preds[TARGET] == 1]
        sns.distplot(true_pred[key], hist=True, kde=True, 
                     bins=int(50), color = 'green', 
                     hist_kws={'edgecolor':'black'}, ax = ax[counter])


        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        ax[counter].text(0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14,
                        verticalalignment = "top", bbox=props)

        ax[counter].set_title(f"{key}")
        ax[counter].set_xlim(0,1)
        ax[counter].set_xlabel("Probability")

    plt.tight_layout()
plot_model_AUC(classifiers)
shap.initjs()

explainer = shap.TreeExplainer(classifiers["LGBM"])
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values[1], X_train)

# ================== 10. Final prediction (ensemble-learning with meta-classifier) ======
test_preds = classifiers['scl'].predict_proba(test)[:,1]

threshold = pd.Series(test_preds).sort_values(ascending = False).head(34911).values[-1]
print(f"Current threshold is: {threshold}")

submission['submit_1'] = (test_preds > threshold).astype(int)
submission['submit_1'].mean()

# 11. Ensemble prediction
submission['submit_2'] = pd.read_csv("C:/data/keggle/taita/dae.csv")[TARGET]
submission['submit_3'] = pseudo_labels[TARGET]

submission[[col for col in submission.columns if col.startswith('submit_')]].sum(axis = 1).value_counts()

submission[TARGET] = (submission[[col for col in submission.columns if col.startswith('submit_')]].sum(axis=1) >= 2).astype(int)
submission[TARGET].mean()

submission[['PassengerId', TARGET]].to_csv("C:/data/keggle/taita/answer/sub_0429.csv", index = False)


