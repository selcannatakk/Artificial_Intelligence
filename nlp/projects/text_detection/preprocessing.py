import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../processed_data/labeling_data.csv")
# LabelEncoder'ı oluştur
encoder = LabelEncoder()

# 'Meyve' sütununu encode et
data['Encoded_Label'] = encoder.fit_transform(data['Label'])

# Encode edilmiş DataFrame'i göster
print(data)

# Result data (model kullanıcak)
data.to_csv('../processed_data/preprocessing_data.csv', index=False)
