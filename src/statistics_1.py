import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_path = './data/requests.csv'
output_image_path = './data/correlation_matrix.png'

data = pd.read_csv(file_path)

data_encoded = pd.get_dummies(data, columns=['status', 'order_type'], drop_first=False)

# Select relevant columns for correlation analysis
correlation_data = data_encoded[['requested_items', 'total_value', 'is_urgent', 'is_from_wholesaler',
                                 'price_per_item'] +
                                 [col for col in data_encoded.columns if 'status' in col] +
                                 [col for col in data_encoded.columns if 'order_type' in col]]

correlation_matrix = correlation_data.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 10})
plt.xticks(rotation=90, ha='center', fontsize=12)
plt.yticks(fontsize=12)
plt.title('Correlation Matrix', fontsize=16)


plt.tight_layout()
plt.subplots_adjust(bottom=0.3)

plt.title('Correlation Matrix')

os.remove(output_image_path)
plt.savefig(output_image_path, bbox_inches='tight')

plt.show()

print(f"Correlation matrix heatmap saved as {output_image_path}")