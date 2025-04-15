from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.cluster import KMeans

model_name = "sentence-transformers/all-MiniLM-L12-v2"
model_kwargs = {'device': 'cpu', 'trust_remote_code':True}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,

)
def return_closest_indices(docs):
    vectors = hf.embed_documents([x.page_content for x in docs])
    num_clusters = 5
    while (len(vectors) <= num_clusters):
        num_clusters -= 1
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    closest_indices = []
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
    selected_indices = sorted(closest_indices)
    return selected_indices