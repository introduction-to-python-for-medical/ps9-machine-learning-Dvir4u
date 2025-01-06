%load_ext autoreload
%autoreload 2
# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
import pandas as pd
df = pd.read_csv('parkinsons.csv')
df = df.dropna()

input_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']
output_feature = 'status'
input_features = ['PPE', 'DFA']
output_feature = 'status'
x = df[input_features]
y = df[output_feature]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import joblib
joblib.dump(model, 'my_model.joblib')
