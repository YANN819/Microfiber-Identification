
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv

os.chdir(r'D:\Anaconda3\pythonwork\ramanspec_file\data')
input_folder = 'Ramandata_eastchinasea_txt'
output_folder = 'Ramandata_eastchinasea_csv'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

for txt_file in txt_files:
    input_path = os.path.join(input_folder, txt_file)
    output_file = os.path.splitext(txt_file)[0] + '.csv'
    output_path = os.path.join(output_folder, output_file)

    with open(input_path, 'r') as txt, open(output_path, 'w', newline='') as csv_file:
        reader = csv.reader(txt, delimiter='\t') 
        
        for i, row in enumerate(reader):
            writer.writerow(row)

print("Finished transformation")

input_folder = output_folder  
output_folder = r'D:\Anaconda3\pythonwork\ramanspec_file\data' 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

result_df = pd.DataFrame()

csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    input_path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(input_path, header=None)
    df = df[1].T
    base_name, _ = os.path.splitext(csv_file)
    df['File Name'] = base_name
    result_df = result_df.append(df, ignore_index=True)


result_df = result_df[['File Name'] + [col for col in result_df.columns if col != 'File Name']]
output_file = os.path.join(output_folder, 'RawRamandata_eastchinasea.csv')
result_df.to_csv(output_file, index=False)
print("Finish transposing and saved in ", output_file)

data = pd.read_csv(output_file, index_col=0)

feature_names = data.columns
sample_names = data.index

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.T).T
normalized_df = pd.DataFrame(normalized_data, columns=feature_names, index=sample_names)
normalized_file_path = os.path.join(output_folder, 'Ramandata_eastchinasea.csv') 
normalized_df.to_csv(normalized_file_path)

print(normalized_df)
