import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# 1. Configuración de tus talentos y precios ficticios
data = [
    {"id": "v1", "habilidad": "Análisis de Datos Power BI", "texto": "Especialista en tableros dinámicos Power BI y DevExpress con KPIs avanzados[cite: 6, 15, 35].", "precio": 150},
    {"id": "v2", "habilidad": "Ciberseguridad Ransomware", "texto": "Blindaje contra Ransomware y gestión de servidores de backup[cite: 6, 20].", "precio": 200},
    {"id": "v3", "habilidad": "Seguridad Electrónica", "texto": "Diseño e implementación de videovigilancia avanzada Cámaras IP y analógicas[cite: 8, 11].", "precio": 45},
    {"id": "v4", "habilidad": "Desarrollo Python/C#", "texto": "Desarrollo Full Stack, integración de WhatsApp API y migradores de bases de datos[cite: 31, 32, 36].", "precio": 300},
    {"id": "v5", "habilidad": "Consultoría Timonel", "texto": "Perfil metódico y analítico para traducir procesos complejos a soluciones simples[cite: 4, 5].", "precio": 50}
]

# 2. Inicializar Pinecone y el Modelo de IA gratuito
pc = Pinecone(api_key="pcsk_6Wi5fS_L2Gctev1RSLcAZQMq6M5wTfzAFZ2AnjGs2RfDtFaGTRWrzSBrBCNeDk4Knxr6in")
model = SentenceTransformer('all-MiniLM-L6-v2') # Modelo liviano y gratuito

# Crear el índice si no existe
index_name = "cv-juan-martin"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, # Dimensión del modelo MiniLM
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 3. Generar vectores y subir
for item in data:
    vector = model.encode(item['texto']).tolist()
    index.upsert(vectors=[(item['id'], vector, {"texto": item['texto'], "precio": item['precio']})])

print("¡Talentos cargados con éxito en la nube!")