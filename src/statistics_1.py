import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_path = './data/requests.csv'
output_image_path = './data/correlation_matrix.png'

data = pd.read_csv(file_path)

data_encoded = pd.get_dummies(data, columns=['status', 'order_type'], drop_first=False)

selected_columns = [
    'requested_items',
    'total_value',
    'price_per_item',
    'request_age',
    'risk_score'
]

selected_columns += [col for col in data_encoded.columns if col.startswith('status_')]
selected_columns += [col for col in data_encoded.columns if col.startswith('order_type_')]

correlation_data = data_encoded[selected_columns]
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 9})
plt.xticks(rotation=90, ha='center', fontsize=10)
plt.yticks(fontsize=10)
plt.title('Correlation Matrix (Cleaned)', fontsize=16)

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)

if os.path.exists(output_image_path):
    os.remove(output_image_path)

plt.savefig(output_image_path, bbox_inches='tight')
plt.show()

print(f"✅ Cleaned correlation matrix heatmap saved as {output_image_path}")
