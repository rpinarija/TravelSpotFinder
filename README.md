# TravelSpotFinder - AI-Powered Travel Recommendations

TravelSpotFinder is an intelligent travel recommendation system that helps users discover perfect destinations based on their preferences. Using advanced NLP techniques and the OpenTripMap API, it provides personalized travel suggestions and detailed information about various locations worldwide.

## Features

- **Destination Recommendations**: Get personalized travel suggestions based on your preferences
- **Location Explorer**: Explore detailed information about specific destinations
- **Similar Places**: Find destinations similar to your favorite locations
- **Topic Analysis**: Visualize popular destination categories and statistics
- **Interactive Maps**: View locations on interactive maps
- **AI-Powered Matching**: Uses advanced NLP and semantic similarity to match your preferences

## Installation

1. Clone this repository:
```bash
git clone https://github.com/rpinarija/TravelSpotFinder.git
cd TravelSpotFinder
```

2. Create a virtual environment (recommended):
```bash
python -m venv myvenv
source myvenv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

5. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

6. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenTripMap API key:
   ```
   OPENTRIPMAP_API_KEY=your_api_key_here
   ```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to navigate between different features:
   - **Destination Recommendations**: Enter your travel preferences to get personalized suggestions
   - **Location Explorer**: Browse and explore specific destinations
   - **Similar Places**: Find destinations similar to your favorite locations
   - **Topic Analysis**: View statistics and visualizations about destinations

## Data

The application uses data from:
- OpenTripMap API for location information
- Wikipedia for detailed descriptions
- Pre-processed embeddings for semantic similarity

## Technologies Used

- Streamlit for the web interface
- Sentence Transformers for semantic similarity
- Folium for interactive maps
- Plotly for data visualization
- NLTK for text processing
- scikit-learn for machine learning algorithms