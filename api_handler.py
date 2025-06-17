import asyncio
import requests
import pandas as pd
import pycountry
import wikipedia
from geopy.distance import geodesic
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenTripMapAPI:
    def __init__(self):
        self.api_key = os.getenv('OPENTRIPMAP_API_KEY')
        if not self.api_key:
            raise ValueError("OpenTripMap API key not found. Please set OPENTRIPMAP_API_KEY in your .env file")
        
        self.base_url = "https://api.opentripmap.com/0.1/en/places"
        self.page_length = 5
        self.min_description_length = 50

    async def api_get(self, method: str, query: Optional[str] = None) -> Dict:
        """Make API request to OpenTripMap"""
        try:
            url = f"{self.base_url}/{method}?apikey={self.api_key}"
            if query:
                url += f"&{query}"
            response = requests.get(url)
            return response.json()
        except requests.exceptions.RequestException as err:
            print(f"Fetch Error: {err}")
            return None

    def get_country_name(self, country_code: str) -> str:
        """Convert country code to full country name"""
        try:
            country = pycountry.countries.get(alpha_2=country_code)
            return country.name if country else "Unknown Country"
        except KeyError:
            return "Unknown Country"

    def are_coordinates_too_close(self, coord1: tuple, coord2: tuple, threshold: int = 10) -> bool:
        """Check if two coordinates are too close to each other"""
        return geodesic(coord1, coord2).meters < threshold

    async def get_place_details(self, xid: str, country: str, distance: float, 
                              kinds: str, city_name: str, 
                              prev_coordinates: Optional[tuple] = None) -> Dict[str, Any]:
        """Fetch detailed information about a place"""
        data = await self.api_get(f"xid/{xid}")
        
        if not data:
            return {"xid": xid, "error": "Details not found"}

        description = data.get("wikipedia_extracts", {}).get("text") or data.get("info", {}).get("descr", "")
        new_coordinates = (data.get("point", {}).get("lat"), data.get("point", {}).get("lon"))

        # Skip if coordinates are too close to previous ones
        if prev_coordinates and self.are_coordinates_too_close(prev_coordinates, new_coordinates):
            return None

        # If description is too short, try to get more information
        if len(description) < self.min_description_length:
            try:
                wiki_summary = wikipedia.summary(city_name, sentences=2)
                description = wiki_summary if wiki_summary else description
            except wikipedia.exceptions.PageError:
                pass

        return {
            "xid": xid,
            "location": city_name,
            "countries": country,
            "lon": data.get("point", {}).get("lon"),
            "lat": data.get("point", {}).get("lat"),
            "distance": distance,
            "kinds": kinds.split(","),
            "place_type": data.get("name", "Unknown"),
            "rating": data.get("rate", "No rating"),
            "wikidata_id": data.get("wikidata", "N/A"),
            "description": description
        }

    async def search_place(self, places: List[str]) -> List[Dict]:
        """Search for multiple places and get their details"""
        results = []
        for name in places:
            print(f"\nSearching for: {name}")
            data = await self.api_get("geoname", f"name={name}")

            if data and data.get('status') == "OK":
                lon = data['lon']
                lat = data['lat']
                country = self.get_country_name(data['country'])
                print(f"Location found: {name}, {country}")
                
                # Get places within radius
                radius = 4000  # 4km radius
                places_data = await self.api_get(
                    "radius",
                    f"radius={radius}&limit={self.page_length}&lon={lon}&lat={lat}&rate=2&format=json"
                )

                if places_data:
                    for place in places_data:
                        place_details = await self.get_place_details(
                            place['xid'], country, place['dist'], 
                            place['kinds'], name
                        )
                        if place_details:
                            results.append(place_details)
            else:
                print(f"Name not found for: {name}")
                results.append({"location": name, "error": "Name not found"})
        
        return results

    async def fetch_data(self, places_list: List[str]) -> pd.DataFrame:
        """Fetch data for a list of places and return as DataFrame"""
        results = await self.search_place(places_list)
        df = pd.DataFrame(results)
        df.drop_duplicates(subset=['description'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

# Example usage
async def main():
    # Example places to search
    places = [
        "Amsterdam", "Bangkok", "Barcelona", "Berlin", "Brasilia",
        "Cape Town", "Chicago", "Dubai", "Ho Chi Minh City", "Jakarta",
        "Los Angeles", "London", "Melbourne", "New York", "Paris",
        "The Maldives", "Vienna"
    ]
    
    api = OpenTripMapAPI()
    df = await api.fetch_data(places)
    return df

if __name__ == "__main__":
    df = asyncio.run(main())
    print(f"Fetched data for {len(df)} locations") 