#Imports
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from geopy.geocoders import Nominatim
import openrouteservice
from pathlib import Path
import base64

#Bakgrundsbild och chattdesign 
def local_bg_image(file_path):
    ext = Path(file_path).suffix.replace('.', '')
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/{ext};base64,{encoded}"

bg_img = local_bg_image("images.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("{bg_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        max-width: 900px;
        margin: auto;
        box-shadow: 0 0 30px rgba(0, 0, 0, 0.3);
    }}
    .stChatMessage {{
        background-color: white !important;
        color: black !important;
        padding: 0.75rem 1rem;
        border-radius: 15px;
        margin-bottom: 0.75rem;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }}
    .stChatMessage.user {{
        text-align: right;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Modell och index (cachas)
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_index():
    return faiss.read_index('Data/faiss_index.index')

model = load_model()
index = load_index()

#Läser in FAQ-texter
with open('Data/text_entries.txt', 'r', encoding='utf-8') as f:
    text_files = [entry.strip() for entry in f.read().split('<END>\n') if entry.strip()]

#Vitlistade frågor med individuella thresholds
ALLOWED_QUESTIONS = [
    ("Hur ställer man in effektbrytare?", 0.3, 0),
    ("Hur gör jag för att justera en effektbrytare?", 0.4, 1),
    ("Finns det en guide för inställning av effektbrytare?", 0.3, 2),
    ("Var hittar jag manual för effektbrytare?", 0.3, 3),
    ("Hur programmerar man Micrologic?", 0.3, 4),
    ("Hur funkar inställning av Micrologic X?", 0.3, 5),
    ("Hur konfigurerar jag en effektbrytare?", 0.3, 6),
    ("Hur programmerar jag en effektbrytare?", 0.3, 7),
    ("Hur ställer jag parametrarna på Micrologic?", 0.3, 8),
    ("Hur justerar man skyddsvärden på effektbrytare?", 0.3, 9),
    ("Hur ändrar jag inställningar på en effektbrytare?", 0.3, 10),
    ("Finns det instruktioner för Micrologic?", 0.3, 11),
    ("Var finns dokumentation för Micrologic X?", 0.3, 12),
    ("Hur använder man displayen på Micrologic?", 0.3, 13),
    ("Hur tolkar man larm i Micrologic X?", 0.3, 14),
    ("Vad kostar frakten?", 0.3, 15),
    ("Hur mycket kostar det att skicka något?", 0.3, 16),
    ("Vad är fraktpriset?", 0.3, 17),
    ("Vad tar ni i fraktavgift?", 0.3, 18),
    ("Vad kostar en leverans?", 0.3, 19),
    ("Hur dyrt är det att skicka en pall?", 0.3, 20),
    ("Har ni fraktkostnad?", 0.3, 21),
    ("Hur räknar ni ut frakt?", 0.3, 22),
    ("Vad debiterar ni för leverans?", 0.3, 23),
    ("Hur mycket blir det i frakt?", 0.3, 24),
    ("Kan jag få en fraktberäkning?", 0.3, 25),
    ("Kan man hämta skåpen själv hos er i Västberga?", 0.3, 26),
    ("Får jag hämta mitt skåp själv?", 0.3, 27),
    ("Har ni avhämtning i Västberga?", 0.3, 28),
    ("Kan jag komma förbi och hämta?", 0.3, 29),
    ("Kan jag slippa frakt och hämta på plats?", 0.3, 30),
    ("Går det att plocka upp skåpet hos er?", 0.3, 31),
    ("Har ni ett lager man kan hämta från?", 0.3, 32),
    ("Kan man hämta produkten på plats?", 0.3, 33),
    ("Får jag hämta beställningen direkt?", 0.3, 34),
    ("Erbjuder ni upphämtning i Hägersten?", 0.3, 35),
    ("Hur gör jag om jag vill köpa av er?", 0.3, 36),
    ("Hur köper man från er?", 0.3, 37),
    ("Kan jag beställa direkt av er?", 0.3, 38),
    ("Hur går ett köp till?", 0.3, 39),
    ("Vad är köpprocessen?", 0.3, 40),
    ("Säljer ni direkt till kunder?", 0.3, 41),
    ("Hur gör man en beställning?", 0.3, 42),
    ("Säljer ni till privatpersoner?", 0.3, 43),
    ("Kan man köpa direkt via hemsidan?", 0.3, 44),
    ("Hur beställer jag en elcentral?", 0.3, 45),
    ("Vad har ni för leveranstid?", 0.3, 46),
    ("Hur lång är leveranstiden?", 0.3, 47),
    ("När kan jag få min leverans?", 0.3, 48),
    ("Hur snabbt levererar ni?", 0.3, 49),
    ("Leveranstid efter beställning?", 0.3, 50),
    ("Vad gäller för leveranstider?", 0.3, 55)
]

#Hjälpfunktioner
def text_to_embedding(text):
    return model.encode(text)

def clean_result_text(text):
    lines = text.splitlines()
    return "\n".join([line for line in lines if not line.strip().upper().startswith("FRÅGA:")]).strip()

def search_index(query, index, k=5):
    query_embedding = text_to_embedding(query)
    query_embedding = normalize(query_embedding.reshape(1, -1), axis=1)

    for allowed_question, threshold, text_index in ALLOWED_QUESTIONS:
        allowed_embedding = text_to_embedding(allowed_question)
        allowed_embedding = normalize(allowed_embedding.reshape(1, -1), axis=1)
        distance = np.linalg.norm(query_embedding - allowed_embedding)

        if distance < threshold:
            st.sidebar.markdown(
                f"Matchad fråga: `{allowed_question}` (distans: {distance:.3f}, tröskel: {threshold})"
            )
            return clean_result_text(text_files[text_index])


    #Fall-back till FAISS
    D, I = index.search(query_embedding, k)
    if D[0][0] < 1.0:
        return clean_result_text(text_files[I[0][0]])

    return None

def berakna_fraktkostnad(km):
    if km <= 50:
        return 800
    elif km <= 500:
        return int(800 + (km - 50) / (500 - 50) * (1100 - 800))
    else:
        extra_km = km - 500
        extra_kostnad = int(extra_km / 100) * 300
        return int(1100 + extra_kostnad)

#Debugpanel
with st.sidebar.expander("Debug: Analysera fråga 🧪"):
    testfraga = st.text_input("Testa en fråga:")
    if testfraga:
        query_embedding = text_to_embedding(testfraga)
        query_embedding = normalize(query_embedding.reshape(1, -1), axis=1)
        D, I = index.search(query_embedding, 5)
        st.markdown("#### Liknande resultat:")
        for dist, idx in zip(D[0], I[0]):
            st.markdown(f"- **Dist:** {dist:.3f}")
            st.markdown(f"`{text_files[idx][:200]}...`")

#Chatgränssnitt
st.title("E3 Control Chattbot🤖")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if prompt := st.chat_input("Ställ en fråga om E3 Control..."):
    st.session_state.chat_history.append(("user", prompt))
    
    with st.spinner("Jag letar efter ett svar...🔍"):
        svar = search_index(prompt, index)
    
    st.session_state.chat_history.append(("assistant", svar or "Jag är osäker på vad du menar. Kan du omformulera frågan?"))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

    if role == "user" and "frakt" in message.lower():
        with st.chat_message("assistant"):
            st.markdown("Vill du att jag beräknar frakten från vårt lager i Stockholm till din adress?📦")
            kund_adress = st.text_input("Ange din adress (t.ex. Storgatan 10, Göteborg):", key=message)

            if kund_adress:
                try:
                    geolocator = Nominatim(user_agent="e3-control-bot")
                    start = geolocator.geocode("Västbergavägen 26, Hägersten, Stockholm, Sweden", timeout=5)
                    destination = geolocator.geocode(kund_adress + ", Sweden", timeout=5)

                    if not start or not destination:
                        st.error("Kunde inte hitta adressen. Kontrollera stavningen.")
                    else:
                        client = openrouteservice.Client(key="5b3ce3597851110001cf62484a241468714b4f06bf705648d66ce0e1")
                        coords = [(start.longitude, start.latitude), (destination.longitude, destination.latitude)]
                        route = client.directions(coords, profile='driving-car')
                        dist_km = route['routes'][0]['summary']['distance'] / 1000
                        kostnad = berakna_fraktkostnad(dist_km)
                        st.success(f"Avstånd från vårt lager: {int(dist_km)} km. Beräknad fraktkostnad: {kostnad} kr.")
                except Exception as e:
                    st.error(f"Ett fel uppstod vid adresshantering: {e}")