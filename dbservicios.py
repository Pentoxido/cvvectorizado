import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# 1. Configuración de servicios optimizada para DRPC
# Incluye servicios técnicos tradicionales y las nuevas capacitaciones corporativas.
data = [
    {"id": "s1", "texto": "Soporte Técnico Integral: Reinstalación completa de sistema operativo Windows, incluyendo paquete Office y programas esenciales. Optimización para un funcionamiento rápido.", "precio": 40},
    {"id": "s2", "texto": "Limpieza de Software: Eliminación profesional de virus, malware y publicidad molesta. Limpieza de archivos temporales para recuperar la velocidad del equipo.", "precio": 25},
    {"id": "s3", "texto": "Mantenimiento de Notebooks: Reparación de bisagras dañadas, carcasas y limpieza física interna de componentes para evitar sobrecalentamiento.", "precio": 35},
    {"id": "s4", "texto": "Upgrade de Velocidad: Aceleración de computadoras mediante la instalación de discos de estado sólido (SSD), ideal para revivir equipos lentos.", "precio": 30},
    {"id": "s5", "texto": "Armado de PC Gamer: Asesoramiento técnico en la elección de componentes por internet y ensamblado profesional de computadoras de alto rendimiento.", "precio": 60},
    {"id": "s6", "texto": "Iluminación LED Inteligente: Instalación y configuración de tiras LED (RGB o monocolor) en bajomesadas, techos o muebles, controlables desde el celular.", "precio": 35},
    {"id": "s7", "texto": "Domótica y Luces Wi-Fi: Instalación de lámparas inteligentes y módulos de encendido remoto para automatizar la iluminación del hogar mediante Wi-Fi.", "precio": 25},
    {"id": "s8", "texto": "Electricidad Menor y USB: Arreglo de enchufes, instalación de tomas con puertos USB integrados, armado de alargues reforzados y reparación de lámparas.", "precio": 15},
    {"id": "s9", "texto": "Asesoramiento y Consultoría: Guía experta para la compra de productos tecnológicos. Explicación de componentes para que el cliente compre exactamente lo que necesita.", "precio": 20},
    {"id": "s10", "texto": "Desarrollo y Redes: Creación de programas a medida para automatización y optimización de redes de trabajo o conectividad a internet profesional.", "precio": 250},
    {"id": "s11", "texto": "Backups Automáticos: Configuración de sistemas de respaldo automático de archivos en la nube o discos locales para proteger información crítica.", "precio": 35},
    {"id": "s12", "texto": "Reparación Electrónica: Revisión y reparación técnica de equipos electrónicos domésticos, de jardín y dispositivos de tecnología avanzada.", "precio": 30},
    {"id": "s13", "texto": "Análisis de Datos Power BI: Especialista en tableros dinámicos Power BI y DevExpress con KPIs avanzados para seguimiento comercial.", "precio": 150},
    {"id": "s14", "texto": "Seguridad Electrónica: Diseño e implementación de videovigilancia avanzada con cámaras IP Hikvision y monitoreo remoto por celular.", "precio": 45},
    {"id": "s15", "texto": "Capacitación en Ciberseguridad Corporativa: Talleres para empleados sobre prevención de Phishing, detección de correos con virus y manejo seguro de archivos y contraseñas.", "precio": 120},
    {"id": "s16", "texto": "Consultoría en Optimización de Costos IT: Sesión estratégica para empresarios para reducir gastos en licencias, servicios de nube y contratación de tecnología innecesaria.", "precio": 100},
    {"id": "s17", "texto": "Modernización Tecnológica: Capacitación para empresarios y empleados en el uso de herramientas de Inteligencia Artificial y digitalización para aumentar la productividad.", "precio": 130}
]

# 2. Inicializar Pinecone y el Modelo de IA
pc = Pinecone(api_key="pcsk_6Wi5fS_L2Gctev1RSLcAZQMq6M5wTfzAFZ2AnjGs2RfDtFaGTRWrzSBrBCNeDk4Knxr6in")
model = SentenceTransformer('all-MiniLM-L6-v2') 

index_name = "cv-juan-martin"

# Verificar si el índice existe
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 3. Generar vectores y subir (Upsert)
print(f"Subiendo {len(data)} servicios (incluyendo nuevas capacitaciones) a Pinecone...")

for item in data:
    # Generamos el vector numérico del texto descriptivo
    vector = model.encode(item['texto']).tolist()
    
    # Subimos el vector con sus metadatos
    index.upsert(vectors=[(
        item['id'], 
        vector, 
        {"texto": item['texto'], "precio": item['precio']}
    )])

print("¡Base de datos de DRPC actualizada con éxito! Tu asistente ya puede ofrecer las capacitaciones corporativas.")