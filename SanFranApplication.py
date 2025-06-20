#!/usr/bin/env python3
"""
San Francisco Home Comparables Finder
A tool to find comparable properties ("comps") in San Francisco based on various property characteristics.
"""

import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

@dataclass
class Property:
    """Data class to represent a property with its characteristics"""
    address: str
    lat: float
    lon: float
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    square_feet: Optional[int] = None
    year_built: Optional[int] = None
    property_type: Optional[str] = None
    assessed_value: Optional[float] = None
    lot_size: Optional[float] = None
    zoning: Optional[str] = None


def haversine_np(lat1, lon1, lat2, lon2):
    # vectorized haversine formula for distance in kilometers
    R = 6371  # Earth radius in km
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def plot_comparables(comps_df):
    """
    Visualizes comparable real estate properties using three different plots:
    
    1. Scatter plot of Square Feet vs Price, with points colored by number of bedrooms.
    2. Histogram showing how far each comparable is from the target property (in miles).
    3. Scatter plot of Price per SqFt vs Distance, where point size represents square footage and color shows bathrooms.

    This helps visually explore how nearby listings compare in size, price, and proximity.
    
    Parameters:
        comps_df (pd.DataFrame): DataFrame containing comparable properties, including
                                 fields like 'sqft', 'price', 'bedrooms', 'bathrooms', 'distance'.
    """
    if comps_df.empty:
        print("No comparables to plot.")
        return

    # Clean data for plotting
    comps_df = comps_df.dropna(subset=['sqft', 'price', 'bedrooms', 'bathrooms', 'distance'])

    # Scatter plot: Square Feet vs Price, colored by bedrooms
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        comps_df['sqft'], comps_df['price'],
        c=comps_df['bedrooms'], cmap='viridis', alpha=0.7, edgecolors='k'
    )
    plt.colorbar(scatter, label='Bedrooms')
    plt.xlabel('Square Feet')
    plt.ylabel('Price ($)')
    plt.title('Comparable Properties: Square Feet vs Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Histogram of distances
    plt.figure(figsize=(8, 4))
    plt.hist(comps_df['distance'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Number of Properties')
    plt.title('Distance Distribution of Comparable Properties')
    plt.tight_layout()
    plt.show()

    # Scatter: Price per SqFt vs Distance, size by sqft, color by bathrooms
    comps_df['price_per_sqft'] = comps_df['price'] / comps_df['sqft']

    plt.figure(figsize=(10, 6))
    scatter2 = plt.scatter(
        comps_df['distance'], comps_df['price_per_sqft'],
        s=comps_df['sqft'] / 10,  # scale size for visualization
        c=comps_df['bathrooms'], cmap='plasma', alpha=0.7, edgecolors='k'
    )
    plt.colorbar(scatter2, label='Bathrooms')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Price per Square Foot ($/sqft)')
    plt.title('Price per SqFt vs Distance (marker size = sqft)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class SFComparableFinder:
    """
    Main class for finding comparable properties in San Francisco
    """

    def __init__(self):
        self.geolocator = Nominatim(user_agent="sf_comps_finder")
        self.sf_data = None
        self.load_sf_property_data()

    def load_sf_property_data(self):
        """
        Load San Francisco property data from DataSF API
        """
        print("Loading San Francisco property data...")

        # DataSF API endpoint for property tax data
        # https://data.sfgov.org/Housing-and-Buildings/Assessor-Historical-Secured-Property-Tax-Rolls/wv5m-vpq2/about_data
        # They gave API endpoints, similar to what my city also gave for my geospatial project.
        url = "https://data.sfgov.org/resource/wv5m-vpq2.json"

        params = {
            '$limit': 10000,
            '$where': "closed_roll_year >= '2017' AND property_location IS NOT NULL"
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        self.sf_data = pd.DataFrame(data)

        # Clean and prepare the data
        self._clean_property_data()

        print(f"Loaded {len(self.sf_data)} properties from DataSF")

    def _clean_property_data(self):
        """Clean and standardize the property data"""
        if self.sf_data is None or self.sf_data.empty:
            return

        # Convert numeric columns
        numeric_cols = ['number_of_bedrooms', 'number_of_bathrooms', 'number_of_rooms',
                       'year_property_built', 'assessed_fixtures_value', 'assessed_land_value',
                       'assessed_improvement_value']

        for col in numeric_cols:
            if col in self.sf_data.columns:
                self.sf_data[col] = pd.to_numeric(self.sf_data[col], errors='coerce')

        # Create total assessed value
        if 'assessed_fixtures_value' in self.sf_data.columns and 'assessed_land_value' in self.sf_data.columns:
            self.sf_data['total_assessed_value'] = (
                self.sf_data['assessed_fixtures_value'].fillna(0) +
                self.sf_data['assessed_land_value'].fillna(0) +
                self.sf_data['assessed_improvement_value'].fillna(0)
            )

        # Clean property location for geocoding
        if 'property_location' in self.sf_data.columns:
            self.sf_data['clean_address'] = self.sf_data['property_location'].str.replace(r'\([^)]*\)', '', regex=True).str.strip()

        # Remove properties with insufficient data
        self.sf_data = self.sf_data.dropna(subset=['property_location'])
        # print(self.sf_data.head())


    def geocode_address(self, address: str) -> Tuple[float, float]:
        """
        Geocode an address to get latitude and longitude
        """
        try:
            # Ensure address includes San Francisco, CA
            if "San Francisco" not in address:
                address += ", San Francisco, CA"

            location = self.geolocator.geocode(address, timeout=10)
            if location:
                return location.latitude, location.longitude
            else:
                raise ValueError(f"Could not geocode address: {address}")
        except Exception as e:
            print(f"Geocoding error: {e}")
            # Return approximate SF coordinates as fallback
            return 37.7749, -122.4194

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in miles"""
        return geodesic((lat1, lon1), (lat2, lon2)).miles

    def calculate_similarity_score(self, target_property: pd.Series, candidate_property: pd.Series) -> float:
        """
        Calculate a similarity score between target and candidate properties.
        Scores normalized to 0-100, higher = more similar.
        """
        score = 0.0
        max_score = 0.0

        # Distance (miles)
        distance = self.calculate_distance(
            target_property['parsed_lat'], target_property['parsed_long'],
            candidate_property['parsed_lat'], candidate_property['parsed_long']
        )
        max_score += 20
        if distance <= 2.0:
            score += max(0, 1 - distance / 2.0) * 20

        # Bedrooms
        max_score += 15
        if pd.notnull(target_property['bedrooms']) and pd.notnull(candidate_property['bedrooms']):
            bed_diff = abs(target_property['bedrooms'] - candidate_property['bedrooms'])
            score += max(0, 1 - bed_diff / 3.0) * 15

        # Bathrooms
        max_score += 10
        if pd.notnull(target_property['bathrooms']) and pd.notnull(candidate_property['bathrooms']):
            bath_diff = abs(target_property['bathrooms'] - candidate_property['bathrooms'])
            score += max(0, 1 - bath_diff / 2.0) * 10

        # Square footage
        max_score += 20
        if pd.notnull(target_property['sqft']) and pd.notnull(candidate_property['sqft']) and target_property['sqft'] > 0:
            sqft_diff = abs(target_property['sqft'] - candidate_property['sqft'])
            sqft_ratio = sqft_diff / target_property['sqft']
            score += max(0, 1 - sqft_ratio) * 20

        # Price per sqft
        max_score += 25
        if pd.notnull(target_property['price']) and pd.notnull(target_property['sqft']) and target_property['sqft'] > 0 \
        and pd.notnull(candidate_property['price']) and pd.notnull(candidate_property['sqft']) and candidate_property['sqft'] > 0:
            target_ppsqft = target_property['price'] / target_property['sqft']
            candidate_ppsqft = candidate_property['price'] / candidate_property['sqft']
            price_diff_ratio = abs(target_ppsqft - candidate_ppsqft) / target_ppsqft
            score += max(0, 1 - price_diff_ratio) * 25

        return (score / max_score) * 100 if max_score > 0 else 0

   
    def find_property_by_long_lat(self, long, lat):
        """
        Finds the property in the dataset that's geographically closest to the given longitude and latitude.

        Parameters:
            long (float): Longitude of the location we're trying to match.
            lat (float): Latitude of the location we're trying to match.

        Returns:
            pd.Series: The row from the dataset that represents the closest property to the given coordinates.
        """

        def extract_coords(geom):
            """
            Tries to extract longitude and latitude from a GeoJSON-like geometry dictionary.

            Parameters:
                geom (dict): A dictionary expected to contain 'coordinates' in the format [longitude, latitude].

            Returns:
                tuple: (longitude, latitude) if successful, otherwise (None, None).
            """
            try:
                if isinstance(geom, dict) and 'coordinates' in geom:
                    return geom['coordinates'][0], geom['coordinates'][1]
            except Exception as e:
                print(f"Failed to extract coords from: {geom} → {e}")
            return None, None

        # Parse coordinates if not already parsed
        if 'parsed_long' not in self.sf_data.columns or 'parsed_lat' not in self.sf_data.columns:
            self.sf_data[['parsed_long', 'parsed_lat']] = self.sf_data['the_geom'].apply(
                lambda g: pd.Series(extract_coords(g))
            )

        # Drop any rows missing coordinates (litterally no point in trying to find a property without coordinates)
        df = self.sf_data.dropna(subset=['parsed_lat', 'parsed_long']).copy()

        # Compute Haversine distance in kilometers
        df['distance_km'] = haversine_np(lat, long, df['parsed_lat'], df['parsed_long'])

        # Find the closest row
        closest = df.loc[df['distance_km'].idxmin()]

        print("Found closest matching property:") # This is something I had to add, because finding the OG property may have limitions in the dataset.
        print(f"Address: {closest['clean_address']}")
        print(f"Distance: {closest['distance_km']:.4f} km")
        return closest
    
    def format_results(self, comparables: List[Dict]) -> str:
        """Format the results for display"""
        if not comparables:
            return "No comparable properties found."

        result = "\n" + "="*80 + "\n"
        result += "COMPARABLE PROPERTIES FOUND\n"
        result += "="*80 + "\n"

        for i, comp in enumerate(comparables, 1):
            result += f"\n{i}. {comp['address']}\n"
            result += f"   Distance: {comp['distance_miles']} miles\n"
            result += f"   Similarity Score: {comp['similarity_score']}/100\n"

            if comp['bedrooms']:
                result += f"   Bedrooms: {comp['bedrooms']}\n"
            if comp['bathrooms']:
                result += f"   Bathrooms: {comp['bathrooms']}\n"
            if comp['square_feet']:
                result += f"   Square Feet: {comp['square_feet']:,}\n"
            if comp['year_built']:
                result += f"   Year Built: {int(comp['year_built'])}\n"
            if comp['assessed_value']:
                result += f"   Assessed Value: ${comp['assessed_value']:,.0f}\n"

            result += "-" * 50 + "\n"

        return result
    
    def find_comparables(self, target_property: pd.Series, df_nearby: pd.DataFrame) -> pd.DataFrame:
        """
        Find and score comparables without heavy filters, rank by similarity, return top 5.
        """
        # Calculate similarity score for each candidate
        df_nearby = df_nearby.copy()

        # Ensure parsed_lat, parsed_long exist
        if 'parsed_lat' not in df_nearby.columns or 'parsed_long' not in df_nearby.columns:
            df_nearby[['parsed_long', 'parsed_lat']] = df_nearby['the_geom'].apply(
                lambda g: pd.Series(self._extract_coords(g))
            )
        df_nearby = df_nearby.dropna(subset=['parsed_lat', 'parsed_long', 'price', 'sqft', 'bedrooms', 'bathrooms'])

        # Calculate similarity scores
        df_nearby['similarity_score'] = df_nearby.apply(lambda row: self.calculate_similarity_score(target_property, row), axis=1)

        # Remove the target property itself if present
        df_nearby = df_nearby[df_nearby.index != target_property.name]

        # Sort by similarity score descending, pick top 5
        top_comps = df_nearby.sort_values(by='similarity_score', ascending=False).head(5)

        return top_comps
    


def main():
    """Main function to run the comparable finder with user input and retry on failure."""
    finder = SFComparableFinder()

    while True:
        address = input("Enter a San Francisco address to find comparables: ").strip()
        if not address:
            print("Please enter a non-empty address.")
            continue

        # Step 1: Geocode
        lat, lon = finder.geocode_address(address)
        print(f"Geocoded address '{address}' to coordinates: ({lat}, {lon})")

        # Step 2: Find the matching property in the dataset
        property_row = finder.find_property_by_long_lat(lon, lat)

        if property_row is None:
            print("No matching property found for that address. Please try another address.\n")
            continue

        print("Found exact property in dataset.")

        # Step 3: Create a dataframe of nearby properties (within ~100 miles here)
        nearby_df = finder.sf_data.copy()
        nearby_df = nearby_df.dropna(subset=['parsed_lat', 'parsed_long'])
        nearby_df['distance'] = haversine_np(lat, lon, nearby_df['parsed_lat'], nearby_df['parsed_long'])
        nearby_df = nearby_df[nearby_df['distance'] <= 100.0]

        # Step 4: Create synthetic fields for price and sqft
        nearby_df['sqft'] = pd.to_numeric(nearby_df.get('property_area', np.nan), errors='coerce')
        nearby_df['price'] = nearby_df.get('total_assessed_value', np.nan)
        nearby_df['bedrooms'] = pd.to_numeric(nearby_df['number_of_bedrooms'], errors='coerce')
        nearby_df['bathrooms'] = pd.to_numeric(nearby_df['number_of_bathrooms'], errors='coerce')

        print(f"Nearby dataset size: {len(nearby_df)}")

        # Step 5: Prepare target property dict
        target_property = {
            'property_type': property_row.get('property_class_code_definition'),
            'bedrooms': pd.to_numeric(property_row.get('number_of_bedrooms'), errors='coerce'),
            'bathrooms': pd.to_numeric(property_row.get('number_of_bathrooms'), errors='coerce'),
            'sqft': pd.to_numeric(property_row.get('property_area'), errors='coerce'),
            'price': pd.to_numeric(property_row.get('total_assessed_value'), errors='coerce'),
            'parsed_lat': property_row.get('parsed_lat'),
            'parsed_long': property_row.get('parsed_long')
        }

        # Check if critical data is missing
        if not all(pd.notnull(val) for val in target_property.values()):
            print("Target property has missing critical data, cannot find comparables. Please try another address.\n")
            continue

        target_property = pd.Series(target_property, name=property_row.name)

        # Step 6: Run comparables finder
        comps = finder.find_comparables(target_property, nearby_df)

        # Step 7: Output results or retry
        if comps.empty:
            print("No comparables found with current filters. Please try another address.\n")
            continue

        print("\nFound Comparable Properties:")
        for i, row in comps.iterrows():
            print(f"\nAddress: {row.get('clean_address', 'Unknown')}")
            print(f"Bedrooms: {row.get('bedrooms')}, Bathrooms: {row.get('bathrooms')}")
            print(f"SqFt: {row.get('sqft')}, Price: ${row.get('price'):,.0f}")
            print(f"Distance: {row.get('distance'):.2f} miles")
            print("-" * 40)

        plot_comparables(comps)
        break  # Exit the loop once valid results are found
    

if __name__ == "__main__":
    main()
