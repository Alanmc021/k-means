# ELT574 12700-0 Atividade 3 - K-means e Clusterização de Músicas do Spotify

# Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from google.colab import files
uploaded = files.upload()

# 1. Carregar os dados
file_path = 'musicas_spotify_limpo.csv'
df = pd.read_csv(file_path)
df.head()

# 2. Selecionar e normalizar os dados
features = ['danceability', 'energy', 'loudness', 'tempo', 'valence']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Calcular a inércia para diferentes valores de K
inertias = []
k_range = range(1, 11)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

# Plotar curva do cotovelo
plt.figure(figsize=(8,5))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.title('Curva do Cotovelo')
plt.grid(True)
plt.show()

# 4. Calcular e plotar o score de silhueta para diferentes K
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8,5))
plt.plot(k_range, silhouette_scores, 'go-')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Score de Silhueta')
plt.title('Análise da Silhueta')
plt.grid(True)
plt.show()

# 5. Aplicar o K-means com o melhor K identificado (exemplo: K=4)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Visualizar os clusters com PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Visualização dos Clusters (PCA)')
plt.show()

# 7. Análise Final
print("\nResumo dos clusters:\n")
print(df.groupby('Cluster')[features].mean())
