
import pandas as pd
import numpy as np
import csv
import seaborn as sns
import joblib
import itertools
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
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

print(y_train.unique())
print(y_test.unique())



param_grid_svm_pca = {
    'pca__n_components': [50, 100, 200],
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': [1e-3, 1e-2, 1e-1, 1]
}

pipeline_svm_pca = make_pipeline(PCA(), SVC(random_state = 32))

grid_search_svm_pca = GridSearchCV(pipeline_svm_pca, param_grid_svm_pca, cv=3, n_jobs=-1, verbose=2)
grid_search_svm_pca.fit(X_train, y_train)
print("Best parameters:", grid_search_svm_pca.best_params_)

best_svm_pca = grid_search_svm_pca.best_estimator_
cv_scores_svm_pca = cross_val_score(best_svm_pca, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores_svm_pca)
print("Mean CV Score:", np.mean(cv_scores_svm_pca))

y_pred_svm_pca = best_svm_pca.predict(X_test)
report = classification_report(y_test, y_pred_svm_pca, output_dict=True)
print(classification_report(y_test, y_pred_svm_pca))

with open('classification_report_svm.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Precision', 'Recall', 'F1-score', 'Support'])
    for metric, scores in report.items():
        if isinstance(scores, dict):
            writer.writerow([metric] + list(scores.values()))
        else:
            writer.writerow([metric, scores, '', ''])

joblib.dump(best_svm_pca, 'svm_pca_model2.joblib')


y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

y_score_svm_pca = best_svm_pca.decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_svm_pca[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(np.unique(y_test)))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(np.unique(y_test))):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(np.unique(y_test))
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

output_file = "roc_curve_svm.csv"
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['False Positive Rate', 'True Positive Rate']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(fpr["macro"])):
        writer.writerow({'False Positive Rate': fpr["macro"][i], 'True Positive Rate': tpr["macro"][i]})

# In[4]

y_pred_svm_pca = best_svm_pca.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred_svm_pca)
accuracy_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
plt.imshow(accuracy_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=12)
plt.colorbar()

#classes = np.unique(y_test)
classes = ['Armid', 'CA', 'Cotton', 'Feather', 'Flax',
           'PA', 'PE', 'PET', 'PLA', 'PP', 'PU', 'PVA', 'SC',
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
             fontsize = 9, 
             color="white" if accuracy_matrix[i, j] > thresh else "black",
             fontproperties = 'Times New Roman',
             )

plt.tight_layout()
plt.ylabel('True label', fontproperties = 'Times New Roman', fontsize=18)
plt.xlabel('Predicted label', fontproperties = 'Times New Roman', fontsize=18)
plt.savefig('confusion_matrix_svm.svg', format='svg')
plt.show()
