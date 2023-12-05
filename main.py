import folium
from folium.features import GeoJsonTooltip
import json
import os
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import folium_static

# # For WaterTracker
# # 2023

def main():
    #############
    # STREAMLIT #
    #############
    st.markdown("""
        <style>
        iframe {
            width: 100%;
            min-height: 400px;
            height: 100%:
        }
        </style>
        """, unsafe_allow_html=True)
        
    st.title('Work in progress')

if __name__ == "__main__":
    main()