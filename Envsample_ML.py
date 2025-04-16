import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

SVM_Env_vali = False

# 加载新的光谱数据
env_data_path = r'D:\Anaconda3\pythonwork\ramanspec_file\data\rec250210_Envsample_nofluoRaman.csv'
env_data = pd.read_csv(env_data_path, header=0)

if SVM_Env_vali :
    
    model_path = r'D:\Anaconda3\pythonwork\ramanspec_file\data_training\machine_learning\training5_240519\svm_pca_model2.joblib'
    loaded_model = joblib.load(model_path)

    # 提取光谱数据
    X_new = env_data.iloc[:, 1:]  # 获取所有特征
    label_true = env_data.iloc[:, 0]

    label_pred = loaded_model.predict(X_new)

else :
    
    model_path = r'D:\Anaconda3\pythonwork\ramanspec_file\data_training\machine_learning\training5_240519\rf_selected_model2_240519.joblib'
    loaded_model = joblib.load(model_path)
    selected_features = joblib.load(r'D:\Anaconda3\pythonwork\ramanspec_file\data_training\machine_learning\training5_240519\selected_features_rf2_240519.joblib')

    # 使用加载的特征名称进行预测
    X_new = env_data.iloc[:, 1:]  # 获取所有特征
    X_new_selected = X_new[selected_features]
    label_true = env_data.iloc[:, 0]

    label_pred = loaded_model.predict(X_new_selected)


print(label_true)
print(label_pred)
print("Classification Report:")
print(classification_report(label_true, label_pred))

# 绘制混淆矩阵
conf_matrix = confusion_matrix(label_true, label_pred)
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar = True)
plt.xlabel('Predicted Label', fontsize = 16)
plt.ylabel('True Label', fontsize = 16)

if SVM_Env_vali :
    plt.title('Confusion Matrix (SVM)', fontsize = 18)
else:
    plt.title('Confusion Matrix (RF)', fontsize = 18)

plt.show()