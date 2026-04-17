import streamlit as st
import base64
import urllib.parse
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Juan Martín Bidonde | Portfolio IT", page_icon="💼", layout="wide")

@st.cache_resource
def init_connections():
    # MODIFICACIÓN: Ahora usa st.secrets en lugar de las llaves escritas
    pc_api_key = st.secrets["PINECONE_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
    
    pc = Pinecone(api_key=pc_api_key)
    index = pc.Index("cv-juan-martin")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=groq_api_key)
    return index, model, groq_client

# Inicialización de servicios
try:
    index, model, groq_client = init_connections()
except Exception as e:
    st.error(f"Error de conexión: {e}")
    st.stop()

@st.cache_data
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except: return ""

foto_b64 = get_base64_image("micara.jpg")

if "op" not in st.session_state:
    st.session_state.op = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. CSS (TU DISEÑO ORIGINAL) ---
st.markdown(f"""
    <style>
    .block-container {{ padding-top: 0rem !important; padding-bottom: 0rem !important; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stApp {{ background-color: #ffffff; }}

    .hero-banner {{
        background: linear-gradient(135deg, #e6f0fa 0%, #d1e3f8 100%);
        padding: 45px 20px;
        text-align: center;
        border-bottom: 2px solid #cfe0f1;
        margin-bottom: 30px;
    }}

    .profile-photo {{
        width: 105px; height: 105px;
        background-image: url('data:image/jpeg;base64,{foto_b64}');
        background-size: cover; border-radius: 50%; 
        display: inline-block; border: 3px solid white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 12px;
    }}

    .hero-banner h1 {{ color: #1a3a5a !important; font-size: 2.3rem !important; margin: 10px 0 !important; }}
    .hero-banner p {{ color: #0056b3 !important; font-weight: 700; text-decoration: none !important; margin-bottom: 25px; font-size: 1.1rem; }}

    .support-container {{ display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; }}
    .btn-supp {{
        padding: 12px 24px; border-radius: 10px; text-decoration: none !important; 
        font-weight: bold; color: white !important; transition: 0.3s;
        display: inline-block; box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }}
    .btn-supp:hover {{ transform: translateY(-2px); box-shadow: 0 5px 12px rgba(0,0,0,0.2); text-decoration: none !important; }}

    .stButton > button {{
        background-color: white !important;
        color: #333 !important;
        border: 1px solid #eee !important;
        border-radius: 14px !important;
        height: 115px !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        border-color: #007bff !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 20px rgba(0,123,255,0.12) !important;
        color: #007bff !important;
    }}

    .assistant-box {{
        background-color: #fcfdfe; padding: 22px; border-radius: 15px;
        border-left: 6px solid #007bff; margin: 25px 0; border: 1px solid #eef2f6; border-left-width: 6px;
    }}
    
    .expander-text {{ font-size: 0.98rem; line-height: 1.7; color: #3a3a3a; }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. CABECERA ---
hero_html = f"""
    <div class="hero-banner">
        <div class="profile-photo"></div>
        <h1>Juan Martín Bidonde</h1>
        <p>Estratega Tecnológico & Consultor IT</p>
        <div class="support-container">
            <a href="https://anydesk.com/es/downloads/" target="_blank" class="btn-supp" style="background-color: #e63946;">
                🔴 Soporte AnyDesk
            </a>
            <a href="https://www.teamviewer.com/es/download/" target="_blank" class="btn-supp" style="background-color: #005bb7;">
                🔵 Soporte TeamViewer
            </a>
        </div>
    </div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# --- 4. TARJETAS DE SERVICIOS ---
st.write("### 🛠️ Áreas de Especialización")
cols = st.columns(5)
servicios_lista = [
    ("📊", "Datos &\nPower BI", "datos"),
    ("🌐", "Redes &\nWi-Fi", "redes"),
    ("🛡️", "Seguridad &\nCámaras", "seguridad"),
    ("💡", "IA &\nSistemas", "ia"),
    ("🎓", "Formación\nEmpresas", "capacitacion")
]

for i, (icon, label, key) in enumerate(servicios_lista):
    with cols[i]:
        if st.button(f"{icon}\n{label}", key=f"btn_{key}"):
            st.session_state.op = key
            st.components.v1.html(
                f"""<script>window.parent.document.getElementById('detalles').scrollIntoView({{behavior: 'smooth'}});</script>""",
                height=0,
            )

# --- 5. ASISTENTE ---
st.markdown('<div class="assistant-box">', unsafe_allow_html=True)
st.write("#### 🤖 Consulta con nuestra IA")

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with st.form(key="chat_form", clear_on_submit=True):
    c_in, c_bt = st.columns([4, 1])
    user_input = c_in.text_input("", placeholder="¿Cómo puedo optimizar tu tecnología hoy?", label_visibility="collapsed")
    submit_button = c_bt.form_submit_button(label="Consultar")

if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Analizando..."):
        query_vector = model.encode(user_input).tolist()
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        ctx = "".join([f"- {res['metadata']['texto']}\n" for res in results['matches']])
        prompt = f"Eres Juan Martín de DRPC Tandil. {ctx}"
        comp = groq_client.chat.completions.create(messages=[{"role": "system", "content": prompt}] + st.session_state.messages, model="llama-3.1-8b-instant")
        ans = comp.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()

if st.session_state.messages:
    historial_texto = "Resumen de consulta en la Web:\n\n"
    for m in st.session_state.messages:
        rol = "👤 Cliente" if m["role"] == "user" else "🤖 IA"
        historial_texto += f"{rol}: {m['content']}\n\n"
    
    resumen_encoded = urllib.parse.quote(historial_texto)
    
    st.markdown(f"""
        <div style="margin-top: 15px;">
            <a href="https://wa.me/5492494537028?text={resumen_encoded}" target="_blank">
                <button style="width:100%; height:45px; background-color:#25D366; color:white; border:none; border-radius:10px; cursor:pointer; font-weight:bold; font-size: 0.95rem; transition: 0.3s; box-shadow: 0 4px 10px rgba(37,211,102,0.2);">
                    ✅ Enviar historial completo a Juan Martín por WhatsApp
                </button>
            </a>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- 6. DETALLE DE SOLUCIONES ---
st.markdown('<div id="detalles" style="padding-top:20px;"></div>', unsafe_allow_html=True)
st.write("## 📝 Información Detallada de Servicios")

with st.expander("🎓 Capacitaciones Corporativas y Estrategia de Ahorro", expanded=(st.session_state.op == "capacitacion")):
    st.markdown("""<div class="expander-text">Transformamos la cultura tecnológica de tu empresa para prevenir incidentes y reducir costos:<ul><li><b>Higiene Digital:</b> Talleres prácticos sobre detección de Phishing, gestión de archivos y navegación segura.</li><li><b>Auditoría de Gastos IT:</b> Revisamos abonos de software para eliminar gastos innecesarios y optimizar licencias.</li><li><b>Gestión de Identidad:</b> Implementación de políticas de contraseñas robustas y autenticación de dos factores (2FA).</li><li><b>Consultoría en IA:</b> Asesoramiento para dueños de PyMEs sobre cómo delegar tareas repetitivas en herramientas de IA.</li></ul></div>""", unsafe_allow_html=True)

with st.expander("📊 Business Intelligence y Análisis de Datos", expanded=(st.session_state.op == "datos")):
    st.markdown("""<div class="expander-text">Deja de adivinar y empieza a decidir basándote en datos reales:<ul><li><b>Tableros en Power BI:</b> Visualización dinámica de ventas, stock y rendimiento en tiempo real.</li><li><b>Integración de Orígenes:</b> Unificamos datos de Excel, SQL o sistemas de gestión (ERP).</li><li><b>Automatización de Reportes:</b> El sistema se actualiza solo cada mañana, eliminando informes manuales.</li></ul></div>""", unsafe_allow_html=True)

with st.expander("🌐 Infraestructura de Redes y Conectividad", expanded=(st.session_state.op == "redes")):
    st.markdown("""<div class="expander-text">Soluciones de conectividad robustas para que internet nunca sea un problema:<ul><li><b>Wi-Fi de Alta Densidad:</b> Access Points profesionales con tecnología Mesh para cobertura total sin cortes.</li><li><b>Enlaces Rurales:</b> Llevamos internet a campos o depósitos alejados mediante antenas punto a punto.</li><li><b>Cableado Estructurado:</b> Organización de racks y certificación de puntos de red categoría 6.</li></ul></div>""", unsafe_allow_html=True)

with st.expander("🛡️ Seguridad Electrónica y Respaldo", expanded=(st.session_state.op == "seguridad")):
    st.markdown("""<div class="expander-text">Protegemos tus activos físicos e intelectuales:<ul><li><b>Videovigilancia Inteligente:</b> Cámaras con detección de movimiento y acceso remoto desde el celular.</li><li><b>Backups Automatizados:</b> Sistemas de respaldo redundantes (nube y físico) para prevenir pérdida de datos.</li><li><b>Sistemas de Alerta:</b> Integración de notificaciones al móvil ante eventos detectados por seguridad.</li></ul></div>""", unsafe_allow_html=True)

with st.expander("💡 Desarrollo de Sistemas e Inteligencia Artificial", expanded=(st.session_state.op == "ia")):
    st.markdown("""<div class="expander-text">Desarrollamos herramientas a medida que trabajan para vos las 24 horas:<ul><li><b>Agentes de IA Personalizados:</b> Asistentes entrenados con la información de tu empresa para atención al cliente.</li><li><b>Automatización de Procesos:</b> Scripts que realizan tareas repetitivas en web, ahorrando horas de carga manual.</li><li><b>Dashboards de Gestión:</b> Creación de paneles de control internos orientados a la conversión y eficiencia.</li></ul></div>""", unsafe_allow_html=True)

st.divider()

# --- 7. CONTACTO ---
c1, c2 = st.columns([1, 1.5])
with c1:
    st.write("### WhatsApp")
    st.markdown('<a href="https://wa.me/5492494537028" target="_blank"><button style="width:100%; height:50px; background-color:#25D366; color:white; border:none; border-radius:12px; font-weight:bold; cursor:pointer;">💬 Contacto Directo</button></a>', unsafe_allow_html=True)
with c2:
    st.write("### Email Corporativo")
    st.markdown("""<form action="https://formsubmit.co/juan_martin_bidonde@hotmail.com" method="POST"><input type="text" name="name" placeholder="Nombre" style="width:100%; margin-bottom:10px; padding:10px; border:1px solid #ccc; border-radius:5px;" required><input type="email" name="email" placeholder="Email" style="width:100%; margin-bottom:10px; padding:10px; border:1px solid #ccc; border-radius:5px;" required><textarea name="message" placeholder="¿En qué podemos ayudarte?" style="width:100%; margin-bottom:10px; padding:10px; border:1px solid #ccc; border-radius:5px; height:80px;" required></textarea><button type="submit" style="width:100%; background-color:#007bff; color:white; border:none; padding:10px; border-radius:5px; font-weight:bold; cursor:pointer;">Enviar Mensaje</button></form>""", unsafe_allow_html=True)

st.caption("© 2026 DRPC Tandil")