import pandas as pd
import numpy as np
import seaborn as sns
import csv
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report,  roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


train_file_path = 'rec_train_spectra0519.csv'
test_file_path = 'rec_test_spectra0519.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]


param_grid = {
    'n_estimators': [25, 50, 100, 150, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

y_pred = best_rf.predict(X_test)


importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

selected_features = [X_train.columns[i] for i in indices[:int(len(indices) * 0.5)]]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(X_train_selected.shape)
print(X_test_selected.shape)

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train_selected.shape[1]), importances[indices[:int(len(indices) * 0.5)]], color="r", align="center")
plt.xticks(range(X_train_selected.shape[1]), selected_features, rotation=90)
plt.xlim([-1, X_train_selected.shape[1]])
plt.show()


best_rf_selected = RandomForestClassifier(n_estimators=best_rf.get_params()['n_estimators'], 
                                          max_depth=best_rf.get_params()['max_depth'],
                                          min_samples_split=best_rf.get_params()['min_samples_split'],
                                          random_state=32,
                                          criterion="gini",
                                          class_weight="balanced"
                                          )
best_rf_selected.fit(X_train_selected, y_train)

cv_scores_selected = cross_val_score(best_rf_selected, X_train_selected, y_train, cv=5)
print("Cross-validation scores (selected features):", cv_scores_selected)
print("Mean CV Score (selected features):", np.mean(cv_scores_selected))

y_pred_selected = best_rf_selected.predict(X_test_selected)
report = classification_report(y_test, y_pred_selected, output_dict=True)
print(classification_report(y_test, y_pred_selected))

with open('classification_report_rf.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Precision', 'Recall', 'F1-score', 'Support']) 
    for metric, scores in report.items():
        if isinstance(scores, dict):
            writer.writerow([metric] + list(scores.values()))
        else:
            writer.writerow([metric, scores, '', ''])


joblib.dump(selected_features, 'selected_features_rf2_240519.joblib')
joblib.dump(best_rf_selected, 'rf_selected_model2_240519.joblib')


y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
classifier = OneVsRestClassifier(best_rf_selected)
y_score = classifier.fit(X_train_selected, y_train).predict_proba(X_test_selected)


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(np.unique(y_test))):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))
for i in range(len(np.unique(y_test))):
    plt.plot(fpr[i], tpr[i], label=f'type {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

mean_tpr = np.zeros_like(all_fpr)
for i in range(len(np.unique(y_test))):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(np.unique(y_test))
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

output_file = "roc_curve_rf.csv"
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['False Positive Rate', 'True Positive Rate']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(fpr["macro"])):
        writer.writerow({'False Positive Rate': fpr["macro"][i], 'True Positive Rate': tpr["macro"][i]})

y_pred_selected = classifier.predict(X_test_selected)
conf_matrix = confusion_matrix(y_test, y_pred_selected)

import itertools

accuracy_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

plt.figure(figsize=(12, 10))
plt.imshow(accuracy_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['Armid', 'Cellulse Acatate', 'Cotton', 'Feather', 'Flax',
           'PA', 'PE', 'PET', 'PLA', 'PP', 'PU', 'PVA', 'Syn-Cellulose',
           'Silk', 'Wool', 
           ] 
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, fontsize = 18, fontproperties = 'Times New Roman',  ha='right')
plt.yticks(tick_marks, classes, fontproperties = 'Times New Roman', fontsize = 18 )

thresh = accuracy_matrix.max() / 2.
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, f'{conf_matrix[i, j]}\n{accuracy_matrix[i, j]:.2%}',
             horizontalalignment="center",
             verticalalignment="center",
             color="white" if accuracy_matrix[i, j] > thresh else "black",
             fontsize=9,
             fontproperties = 'Times New Roman'
             )

plt.tight_layout()
plt.ylabel('True label', fontproperties = 'Times New Roman', fontsize=18)
plt.xlabel('Predicted label', fontproperties = 'Times New Roman', fontsize=18)
plt.savefig('confusion_matrix_rf.svg', format='svg')
plt.show()

# %%