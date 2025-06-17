import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import asyncio
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from api_handler import OpenTripMapAPI
import logging
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Set page config
st.set_page_config(
    page_title="TravelSpotFinder - Travel Recommendations",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    try:
        st.session_state.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("✈️ TravelSpotFinder")
st.markdown("""
    Discover your perfect travel destinations with AI-powered recommendations.
    Tell us what you're looking for, and we'll help you find the perfect spot!
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a feature:", 
    ["Destination Recommendations", 
     "Location Explorer", 
     "Similar Places",
     "Topic Analysis"])

# Initialize data loading
@st.cache_resource
def get_api():
    try:
        return OpenTripMapAPI()
    except Exception as e:
        st.error(f"Error initializing API: {e}")
        st.stop()

def extract_places(text):
    doc = nlp(text)
    places = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    return places

# Expanded keyword mapping and synonyms
keyword_synonyms = {
    "beach": ["beach", "coast", "seaside", "ocean", "island", "tropical"],
    "nightlife": ["nightlife", "party", "clubs", "bars", "dancing", "music", "pubs"],
    "mountain": ["mountain", "alps", "hiking", "trek", "trekking", "summit", "peak", "hill"],
    "museum": ["museum", "gallery", "art", "exhibit", "history"],
    "food": ["food", "cuisine", "eat", "dining", "restaurant", "culinary", "street food", "gourmet"],
    "nature": ["nature", "forest", "park", "wildlife", "outdoors", "scenery", "landscape", "natural"],
    "romantic": ["romantic", "honeymoon", "couple", "love", "valentine"],
    "adventure": ["adventure", "extreme", "rafting", "bungee", "zipline", "paragliding", "surfing", "diving"],
    "history": ["history", "historic", "ancient", "ruins", "heritage", "archaeology"],
    "shopping": ["shopping", "mall", "market", "boutique", "fashion", "luxury"],
    "island": ["island", "archipelago", "atoll", "tropical"],
    "ski": ["ski", "snowboard", "snow", "alps", "winter sports"],
    "desert": ["desert", "dune", "sahara", "arid", "wadi"],
    "wildlife": ["wildlife", "safari", "animals", "zoo", "reserve", "sanctuary"],
    "art": ["art", "gallery", "museum", "exhibit", "painting", "sculpture"],
    "family": ["family", "kids", "children", "theme park", "zoo", "aquarium"],
    "spa": ["spa", "wellness", "relax", "hot spring", "thermal", "massage"],
    "music": ["music", "concert", "festival", "jazz", "opera", "band", "live"],
    "festivals": ["festival", "carnival", "event", "parade", "celebration", "oktoberfest", "mardi gras"],
}

keywords = {
    "beach": ["Maldives", "Bali", "Miami", "Barcelona", "Phuket", "Cancun", "Santorini"],
    "nightlife": ["Bangkok", "Berlin", "New York", "Ibiza", "Las Vegas", "Rio de Janeiro"],
    "mountain": ["Denver", "Kathmandu", "Chamonix", "Aspen", "Zermatt", "Queenstown"],
    "museum": ["Paris", "London", "New York", "Madrid", "Florence", "St. Petersburg"],
    "food": ["Tokyo", "Bangkok", "Rome", "Lyon", "Istanbul", "Mexico City"],
    "nature": ["Vancouver", "Cape Town", "Queenstown", "Banff", "Reykjavik", "Patagonia"],
    "romantic": ["Venice", "Paris", "Santorini", "Prague", "Kyoto", "Budapest"],
    "adventure": ["Queenstown", "Interlaken", "Cusco", "Cape Town", "Banff", "Moab"],
    "history": ["Rome", "Athens", "Cairo", "Jerusalem", "Istanbul", "Xi'an"],
    "shopping": ["Milan", "Dubai", "New York", "London", "Tokyo", "Paris"],
    "island": ["Maldives", "Bali", "Maui", "Seychelles", "Fiji", "Bora Bora"],
    "ski": ["Aspen", "Zermatt", "Whistler", "Chamonix", "St. Moritz", "Niseko"],
    "desert": ["Dubai", "Marrakech", "Las Vegas", "Cairo", "Phoenix", "Doha"],
    "wildlife": ["Nairobi", "Kruger", "Darwin", "Galapagos", "Yellowstone", "Serengeti"],
    "art": ["Florence", "Paris", "New York", "Berlin", "Barcelona", "Venice"],
    "family": ["Orlando", "San Diego", "Singapore", "Copenhagen", "Sydney", "Toronto"],
    "spa": ["Baden-Baden", "Budapest", "Ubud", "Sedona", "Bath", "Rotorua"],
    "music": ["Nashville", "New Orleans", "Vienna", "Liverpool", "Austin", "Memphis"],
    "festivals": ["Rio de Janeiro", "Munich", "New Orleans", "Venice", "Edinburgh", "Pamplona"],
}

# Theme descriptions for embedding-based fuzzy matching
theme_descriptions = {
    "beach": "Relaxing on beautiful sandy beaches and swimming in the ocean.",
    "nightlife": "Enjoying vibrant nightlife, clubs, and parties.",
    "mountain": "Exploring mountains, hiking, and breathtaking views.",
    "museum": "Visiting world-class museums and art galleries.",
    "food": "Experiencing delicious local cuisine and food markets.",
    "nature": "Exploring nature, forests, parks, and wildlife.",
    "romantic": "Romantic getaways and destinations for couples.",
    "adventure": "Adventure activities like rafting, bungee, and paragliding.",
    "history": "Exploring historical sites, ruins, and ancient cities.",
    "shopping": "Shopping in malls, markets, and boutiques.",
    "island": "Visiting tropical islands and archipelagos.",
    "ski": "Skiing and snowboarding in winter resorts.",
    "desert": "Exploring deserts, dunes, and arid landscapes.",
    "wildlife": "Wildlife safaris and animal encounters.",
    "art": "Experiencing art, galleries, and exhibitions.",
    "family": "Family-friendly destinations and activities.",
    "spa": "Relaxing in spas, hot springs, and wellness retreats.",
    "music": "Enjoying music, concerts, and festivals.",
    "festivals": "Attending festivals, carnivals, and celebrations.",
}

# Precompute theme embeddings
if 'theme_embeddings' not in st.session_state:
    st.session_state.theme_embeddings = {
        theme: st.session_state.model.encode(desc)
        for theme, desc in theme_descriptions.items()
    }

from sklearn.metrics.pairwise import cosine_similarity

def get_best_themes_by_embedding(user_input, top_n=3, threshold=0.5):
    user_emb = st.session_state.model.encode(user_input)
    theme_scores = []
    for theme, emb in st.session_state.theme_embeddings.items():
        score = cosine_similarity([user_emb], [emb])[0][0]
        theme_scores.append((theme, score))
    theme_scores.sort(key=lambda x: x[1], reverse=True)
    return [theme for theme, score in theme_scores if score > threshold][:top_n]

# Update get_places_from_description to use embedding-based fuzzy matching

MAX_PLACES = 20

def get_places_from_description(description):
    lemmatizer = WordNetLemmatizer()
    desc = description.lower()
    desc_tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(desc)]
    matched_themes = set()
    found = []
    for theme, synonyms in keyword_synonyms.items():
        if any(syn in desc_tokens or syn in desc for syn in synonyms):
            matched_themes.add(theme)
            found.extend(keywords[theme])
    best_themes = get_best_themes_by_embedding(description)
    for theme in best_themes:
        if theme not in matched_themes:
            matched_themes.add(theme)
            found.extend(keywords[theme])
    places = extract_places(description)
    if places:
        matched_themes.add("Named Entities")
        found.extend(places)
    if matched_themes:
        st.info(f"Matched themes: {', '.join(sorted(matched_themes))}")
    # Remove duplicates and limit to MAX_PLACES
    if found:
        return list(dict.fromkeys(found))[:MAX_PLACES]
    return [
        "Amsterdam", "Bangkok", "Barcelona", "Berlin", "Brasilia",
        "Cape Town", "Chicago", "Dubai", "Ho Chi Minh City", "Jakarta",
        "Los Angeles", "London", "Melbourne", "New York", "Paris",
        "The Maldives", "Vienna"
    ][:MAX_PLACES]

def load_data(places):
    """Load data from API and cache it in session state"""
    cache_key = "df_" + "_".join(sorted(places))
    if cache_key not in st.session_state:
        try:
            with st.spinner("Loading data from OpenTripMap API..."):
                api = get_api()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    df = loop.run_until_complete(api.fetch_data(places))
                    if df is None or df.empty:
                        raise ValueError("No data received from API")
                    st.session_state[cache_key] = df
                except Exception as e:
                    logger.error(f"Error fetching data: {e}")
                    st.error(f"Error fetching data: {e}")
                    return None
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            st.error(f"Error loading data: {e}")
            return None
    return st.session_state[cache_key]

# Load the data
df = load_data(get_places_from_description(""))

if df is None:
    st.error("Failed to load data. Please check your API key and try again.")
    st.stop()

# Add this line to convert the rating column to numeric
df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')

# Text preprocessing function
def preprocess_text(doc):
    try:
        # Remove html
        doc = re.sub(r'<[^>]+>', '', doc)
        # Remove non-alphanumeric characters
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        # Convert text to lowercase
        doc = doc.lower()
        # Remove extra whitespace
        doc = doc.strip()
        # Tokenize and remove stopwords, then Lemmatize
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(doc) if word not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return ""

def ensure_embeddings(df):
    if 'embeddings' not in df.columns:
        try:
            df['clean_text'] = df['description'].apply(preprocess_text)
            df['embeddings'] = df['clean_text'].apply(lambda x: st.session_state.model.encode(x))
        except Exception as e:
            logger.error(f"Error processing embeddings: {e}")
            st.error("Error processing text embeddings. Please try again.")
            st.stop()
    return df

# Process embeddings if not already done
if 'embeddings' not in df.columns:
    try:
        df['clean_text'] = df['description'].apply(preprocess_text)
        df['embeddings'] = df['clean_text'].apply(lambda x: st.session_state.model.encode(x))
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        st.error("Error processing text embeddings. Please try again.")
        st.stop()

# Recommendation function
def get_recommendations(user_input, df, top_n=5):
    try:
        user_embedding = st.session_state.model.encode(user_input)
        similarities = cosine_similarity([user_embedding], np.stack(df['embeddings']))[0]
        df['similarity'] = similarities
        return df.sort_values(by='similarity', ascending=False).head(top_n)
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        st.error("Error generating recommendations. Please try again.")
        return pd.DataFrame()

# Similar locations function
def find_similar_locations(location, df, top_n=5):
    try:
        location_embedding = st.session_state.model.encode(location)
        similarities = cosine_similarity([location_embedding], np.stack(df['embeddings']))[0]
        df['similarity'] = similarities
        filtered_df = df[df['location'] != location]
        return filtered_df.sort_values(by='similarity', ascending=False).head(top_n)
    except Exception as e:
        logger.error(f"Error in find_similar_locations: {e}")
        st.error("Error finding similar locations. Please try again.")
        return pd.DataFrame()

def parse_rating(raw_rating):
    if not raw_rating or raw_rating == "No rating":
        return None
    digits = ''.join(filter(str.isdigit, str(raw_rating)))
    if digits:
        return int(digits)
    return None

# Main content based on selected page
if page == "Destination Recommendations":
    st.header("Find Your Perfect Destination")
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.form("recommendation_form"):
            user_input = st.text_input(
                "Describe your dream vacation",
                placeholder="e.g., I want to experience beautiful beaches, vibrant nightlife, and rich cultural heritage..."
            )
            submitted = st.form_submit_button("Get Recommendations")
            manual_place = None
            if submitted:
                if user_input:
                    places = get_places_from_description(user_input)
                    default_list = [
                        "Amsterdam", "Bangkok", "Barcelona", "Berlin", "Brasilia",
                        "Cape Town", "Chicago", "Dubai", "Ho Chi Minh City", "Jakarta",
                        "Los Angeles", "London", "Melbourne", "New York", "Paris",
                        "The Maldives", "Vienna"
                    ][:MAX_PLACES]
                    if set(places) == set(default_list):
                        manual_place = st.text_input(
                            "No places matched your description. Enter a city or country manually:",
                            key="manual_destination"
                        )
                        if manual_place:
                            places = [manual_place] + default_list
                            places = list(dict.fromkeys(places))[:MAX_PLACES]
                    df = load_data(places)
                    if df is not None:
                        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
                        df = ensure_embeddings(df)
                        with st.spinner("Finding the perfect destinations for you..."):
                            recommendations = get_recommendations(user_input, df)
                            for idx, row in recommendations.iterrows():
                                with st.expander(f"📍 {row['location']} (Similarity: {row['similarity']:.2f})"):
                                    st.write(row['description'])
                                    m = folium.Map(location=[row['lat'], row['lon']], zoom_start=12)
                                    folium.Marker(
                                        [row['lat'], row['lon']],
                                        popup=row['location'],
                                        icon=folium.Icon(color='red', icon='info-sign')
                                    ).add_to(m)
                                    folium_static(m)
                    else:
                        st.error("No recommendations found. Try a different description.")
                else:
                    st.warning("Please enter your travel preferences to get recommendations.")

elif page == "Location Explorer":
    st.header("Explore Destinations")
    explorer_desc = st.text_input(
        "Describe what you want to explore (e.g. historic cities, mountain towns, foodie capitals):",
        ""
    )
    if explorer_desc.strip():
        explorer_places = get_places_from_description(explorer_desc)
        explorer_df = load_data(explorer_places)
        if explorer_df is not None and not explorer_df.empty:
            selected_location = st.selectbox(
                "Choose a location to explore",
                options=sorted(explorer_df['location'].unique())
            )
            if selected_location:
                location_data = explorer_df[explorer_df['location'] == selected_location].iloc[0]
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Location Details")
                    st.write(f"**Description:** {location_data['description']}")
                    st.write(f"**Country:** {location_data['countries']}")
                    st.write(f"**Place Type:** {location_data['place_type']}")
                    st.write(f"**Rating:** {location_data['rating']}")
                with col2:
                    m = folium.Map(location=[location_data['lat'], location_data['lon']], zoom_start=12)
                    folium.Marker(
                        [location_data['lat'], location_data['lon']],
                        popup=location_data['location'],
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
                    folium_static(m)
        else:
            st.warning("No locations found for your description.")

elif page == "Similar Places":
    st.header("Find Similar Destinations")
    similar_desc = st.text_input(
        "Describe the type of place you want to compare (e.g. beach cities, art capitals, adventure spots):",
        ""
    )
    if similar_desc.strip():
        similar_places = get_places_from_description(similar_desc)
        similar_df = load_data(similar_places)
        if similar_df is not None and not similar_df.empty:
            selected_location = st.selectbox(
                "Choose a location to find similar places",
                options=sorted(similar_df['location'].unique())
            )
            if selected_location:
                similar_df = ensure_embeddings(similar_df)
                similar_locations = find_similar_locations(selected_location, similar_df)
                st.subheader(f"Places Similar to {selected_location}")
                for idx, row in similar_locations.iterrows():
                    with st.expander(f"📍 {row['location']} (Similarity: {row['similarity']:.2f})"):
                        st.write(row['description'])
                        m = folium.Map(location=[row['lat'], row['lon']], zoom_start=12)
                        folium.Marker(
                            [row['lat'], row['lon']],
                            popup=row['location'],
                            icon=folium.Icon(color='red', icon='info-sign')
                        ).add_to(m)
                        folium_static(m)
        else:
            st.warning("No similar places found for your description.")

elif page == "Topic Analysis":
    st.header("Destination Topics")
    topic_desc = st.text_input(
        "Describe the types of destinations you want to analyze (e.g. foodie cities, ski resorts, romantic getaways):",
        ""
    )
    if topic_desc.strip():
        topic_places = get_places_from_description(topic_desc)
        topic_df = load_data(topic_places)
    else:
        topic_df = load_data(get_places_from_description(""))
    if topic_df is not None and not topic_df.empty:
        # Limit to MAX_PLACES
        topic_df = topic_df.head(MAX_PLACES)
        topic_df['rating_numeric'] = pd.to_numeric(topic_df['rating'], errors='coerce')
        st.subheader("Popular Destination Categories")
        place_types = topic_df['place_type'].value_counts().head(10)
        fig = px.bar(
            x=place_types.index,
            y=place_types.values,
            title="Top 10 Most Common Place Types",
            labels={'x': 'Place Type', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Destination Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Destinations", len(topic_df))
        with col2:
            st.metric("Unique Countries", topic_df['countries'].nunique())
        with col3:
            avg_rating = topic_df['rating_numeric'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
    else:
        st.warning("No destinations found for your topic description.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built using Streamlit | Powered by OpenTripMap API and Wikipedia</p>
    </div>
""", unsafe_allow_html=True) 