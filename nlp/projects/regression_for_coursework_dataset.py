import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


import os
import pandas as pd
import seaborn as sns

data_folder = "./datasets/coursework_dataset/"
file_names = []


def get_data(data_folder):
    for file_path in os.listdir(data_folder):
        # file_path = file_path.replace(' ', '')

        file_names.append(os.path.join(data_folder, file_path))

    dfs = [pd.read_csv(file) for file in file_names]

    merged_df = pd.concat(dfs,ignore_index=True)

    return merged_df


def correlation_matrix(df):
    correlation_matrix = df.corr()
    print(correlation_matrix)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    # plt.title('Combined Correlation Matrix')
    # plt.show()

    return correlation_matrix


data = get_data(data_folder)
data.to_csv('df.csv', index=True)
select_data = data.select_dtypes(include=['float64', 'int64'])
select_data.to_csv('select_df.csv', index=True)
correlation_matrix = correlation_matrix(select_data)
correlation_matrix.to_csv('correlation_matrix.csv', index=True)


threshold = 0.2
high_corr = correlation_matrix.columns[(abs(correlation_matrix) > threshold).any()].tolist()


if not high_corr:
    new_select_data = select_data[high_corr]
    new_correlation_matrix = correlation_matrix(new_select_data)
    new_correlation_matrix.to_csv('new_correlation_matrix.csv', index=True)

else:
    print("Threshold değeri çok büyük")
    new_select_data = select_data
# # data = get_data(data_folder)
# columns = ['Area', 'Year', 'Value']
# df = pd.read_csv('./dataset/Food trade indicators - FAOSTAT_data_en_2-22-2024.csv')
#
# export_values = df['Element'] == 'Export Value'
# select_df = df.loc[export_values, columns]
#
# encoder = LabelEncoder()
# select_df['Country_Area'] = encoder.fit_transform(select_df['Area'])
#
# print(select_df)

target_column = 'Value'

X = new_select_data.drop(columns=[target_column], errors='ignore') # diğer colonları al
Y = data[target_column] # numarical olmalı

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Train Data Count = {X_train + y_train}")
print(f"Test Data Count = {X_test + y_test}")


# Veri ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

hidden_l_size = (100, 200)
epochs = 15
batch_size = 32
verbose=1
input_shape = (2,)
# model = MLPRegressor(
#     hidden_layer_sizes=hidden_l_size,
#     activation='relu',
#     solver='adam',
#     random_state=42,
#     max_iter=500)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(X_train_scaled, y_train,epochs=epochs, batch_size=batch_size, verbose=verbose)

# evulate
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
print("Ortalama Mutlak Hata (MAE):", mae)
# mse = model.evaluate(X_test, y_test)
# print("Mean Squared Error:", mse)

# results_df = pd.DataFrame({'y_test': y_test.values.flatten(), 'y_pred': y_pred.values.flatten()})
# results_df['Mean_Absolute_Error'] = mae
# results_df.to_csv('model_test_results.csv', index=False)

test_data = {
    'Year': [2023],
    'Fertilizers use': [1000],
    'Land temperature change': [1.5]}


new_data = pd.DataFrame(test_data)
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("2023 yılı için tahmin edilen ihracat değeri :", prediction[0])