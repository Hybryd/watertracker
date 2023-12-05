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
        
    st.title('Sécheresse en France')

if __name__ == "__main__":
    main()



# # For WaterTracker
# # 2023

# def api_key():
#     return os.environ['API_KEY']

# def state_types():
#     headers = {
#         'accept': 'application/json',
#         'Authorization': f'Bearer {api_key()}'
#     }
    
#     params = {
#         'with':'states;locationTypes'
#     }
    
#     try:
#         response = requests.get(f'https://api.emi.imageau.eu/app/state-types/', params=params, headers=headers)
#     except Exception as e:
#         print(e)
#         return
#     else:
#         if response.status_code == 200:
#             return dict(response.json())
#         else:
#             return None

# def departements():
#     headers = {
#         'accept': 'application/json',
#         'Authorization': f'Bearer {api_key()}'
#     }
    
#     params = {
#         'with':'geometry;indicators.state.type'
#     }
    
#     try:
#         response = requests.get(f'https://api.emi.imageau.eu/app/departments', params=params, headers=headers)
#     except Exception as e:
#         print(e)
#         return
#     else:
#         if response.status_code == 200:
#             return dict(response.json())
#         else:
#             return None

# def update_map(field, geometrie_departements, df_indicators):
#     dict_field_name = {"Nappes phréatiques": "dryness_groundwater_severity",
#                        "Nappes phréatiques profondes": "dryness_groundwater_deep_severity",
#                        "Cours d'eau": "dryness_stream_flow_severity",
#                        "Ruisseaux": "dryness_small_stream_severity",
#                        "Pluviométrie 30 jours": "dryness_meteo_severity",
#                        "Pluviométrie 3 mois": "dryness_agricultural_severity",
#                        "Pluviométrie 6 mois": "dryness_resources_severity",
#                        }
    
#     m = folium.Map(location=[47, 2.2], zoom_start=5.5)
#     c = folium.Choropleth(
#         geo_data=geometrie_departements,
#         name="Sévérité sécheresse",
#         data=df_indicators,
#         line_weight=1,
#         fill_opacity=0.7,
#         line_opacity=0.2,
#         highlight=True,
#         columns=["name", dict_field_name[field]],
#         key_on="feature.properties.name",
#         fill_color='RdYlGn_r',#"RdYlGn_r",
#         bins=[-5,-4,-3,-2,2,3,4,5],
#         legend_name="Sévérité"
#     ).add_to(m)
    
#     # https://gist.github.com/insightsbees/e563de67f2a3fd1ebdf383288fc5dce4
#     #Add Customized Tooltips to the map
#     feature = folium.features.GeoJson(
#     data=geometrie_departements,
#     name='bulles',
#     smooth_factor=2,
#     style_function=lambda x: {'color':'black','fillColor':'transparent','weight':0},
#     tooltip=folium.features.GeoJsonTooltip(
#         fields=['name', "code"],
#         aliases=["",""], 
#         localize=True,
#         sticky=False,
#         labels=True,
#         style="""
#             background-color: #F0EFEF;
#             border: 1px solid black;
#             border-radius: 1px;
#             box-shadow: 3px;
#         """,
#         max_width=800,),
#             highlight_function=lambda x: {'weight': 0,'fillColor':'grey'},
#         ).add_to(m) 
        
#     #folium.GeoJsonTooltip(['name']).add_to(c.geojson)
#     folium.LayerControl().add_to(m)
#     folium_static(m, width=500, height=500)
#     #folium_static(m)


# def main():
#     # Download data from info-secheresse
#     data_departements = departements()

#     geometrie_departements = {"type": "FeatureCollection", 'features': []}
#     for dep in data_departements["data"]:
#         d = {'type': 'Feature',
#              "geometry": dep["geometry"],
#              "properties": {"name": dep["name"],
#                             "code": dep["code"]
#                            }
#             }
#         geometrie_departements["features"].append(d)


#     data_indicators = []
#     for dep in data_departements["data"]:
#         dryness_groundwater_name = "dryness-groundwater"
#         dryness_groundwater_severity = None
#         dryness_groundwater_color = None
        
#         dryness_groundwater_deep_name = "dryness-groundwater-deep"
#         dryness_groundwater_deep_severity = None
#         dryness_groundwater_deep_color = None
        
#         dryness_stream_flow_name = "dryness-stream-flow"
#         dryness_stream_flow_severity = None
#         dryness_stream_flow_color = None
        
#         dryness_small_stream_name = "dryness-small-stream"
#         dryness_small_stream_severity = None
#         dryness_small_stream_color = None
        
#         dryness_meteo_name = "dryness-meteo"
#         dryness_meteo_severity = None
#         dryness_meteo_color = None
        
#         dryness_agricultural_name = "dryness-agricultural"
#         dryness_agricultural_severity = None
#         dryness_agricultural_color = None
        
#         dryness_resources_name = "dryness-resources"
#         dryness_resources_severity = None
#         dryness_resources_color = None
        
#         for indic in dep['indicators']:
#             if indic["state"]["type"]["name"] == dryness_groundwater_name:
#                 dryness_groundwater_severity = indic["state"]["severity"]
#                 dryness_groundwater_color = indic["state"]["color"]
#             elif indic["state"]["type"]["name"] == dryness_groundwater_deep_name:
#                 dryness_groundwater_deep_severity = indic["state"]["severity"]
#                 dryness_groundwater_deep_color = indic["state"]["color"]
#             elif indic["state"]["type"]["name"] == dryness_stream_flow_name:
#                 dryness_stream_flow_severity = indic["state"]["severity"]
#                 dryness_stream_flow_color = indic["state"]["color"]
#             elif indic["state"]["type"]["name"] == dryness_small_stream_name:
#                 dryness_small_stream_severity = indic["state"]["severity"]
#                 dryness_small_stream_color = indic["state"]["color"]
#             elif indic["state"]["type"]["name"] == dryness_meteo_name:
#                 dryness_meteo_severity = indic["state"]["severity"]
#                 dryness_meteo_color = indic["state"]["color"]
#             elif indic["state"]["type"]["name"] == dryness_agricultural_name:
#                 dryness_agricultural_severity = indic["state"]["severity"]
#                 dryness_agricultural_color = indic["state"]["color"]
#             elif indic["state"]["type"]["name"] == dryness_resources_name:
#                 dryness_resources_severity = indic["state"]["severity"]
#                 dryness_resources_color = indic["state"]["color"]
        
#         d = {"name": dep["name"],
#              "code": dep["code"],
#              "dryness_groundwater_severity": dryness_groundwater_severity,
#              "dryness_groundwater_color": dryness_groundwater_color,
#              "dryness_groundwater_deep_severity": dryness_groundwater_deep_severity,
#              "dryness_groundwater_deep_color": dryness_groundwater_deep_color,
#              "dryness_stream_flow_severity": dryness_stream_flow_severity,
#              "dryness_stream_flow_color": dryness_stream_flow_color,
#              "dryness_small_stream_severity": dryness_small_stream_severity,
#              "dryness_small_stream_color": dryness_small_stream_color,
#              "dryness_meteo_severity": dryness_meteo_severity,
#              "dryness_meteo_color": dryness_meteo_color,
#              "dryness_agricultural_severity": dryness_agricultural_severity,
#              "dryness_agricultural_color": dryness_agricultural_color,
#              "dryness_resources_severity": dryness_resources_severity,
#              "dryness_resources_color": dryness_resources_color         
#             }
#         data_indicators.append(d)
#     df_indicators = pd.DataFrame(data_indicators)

#     #############
#     # STREAMLIT #
#     #############
#     st.markdown("""
#         <style>
#         iframe {
#             width: 100%;
#             min-height: 400px;
#             height: 100%:
#         }
#         </style>
#         """, unsafe_allow_html=True)
        
#     st.title('Sécheresse en France')

#     tab_1, tab_2 = st.sidebar.tabs(["Indicateurs", "Budget eau"])
    
#     tab_1.markdown("## Eaux de surface")
#     eau_surface = tab_1.divider()
#     eau_surface_1, eau_surface_2 = eau_surface.columns([1,1])
#     if eau_surface_1.button("Cours d'eau", type="secondary"):
#         st.markdown("Cours d'eau")
#         update_map("Cours d'eau", geometrie_departements, df_indicators)
        
#     if eau_surface_2.button("Ruisseaux", type="secondary"):
#         st.markdown("Ruisseaux")
#         update_map("Ruisseaux", geometrie_departements, df_indicators)
        
#     tab_1.markdown("## Eaux souterraines")
#     eau_souterraine = tab_1.divider()
#     eau_souterraine_1, eau_souterraine_2 = eau_souterraine.columns([1,1])
#     if eau_souterraine_1.button("Nappes phréatiques", type="secondary"):
#         st.markdown("Nappes phréatiques")
#         update_map("Nappes phréatiques", geometrie_departements, df_indicators)
#     if eau_souterraine_2.button("Nappes profondes", type="secondary"):
#         st.markdown("Nappes profondes")
#         update_map("Nappes phréatiques profondes", geometrie_departements, df_indicators)
        
#     tab_1.markdown("## Pluviométrie")
#     pluviometrie = tab_1.divider()
#     pluviometrie_1, pluviometrie_2, pluviometrie_3 = pluviometrie.columns([1,1,1])
#     if pluviometrie_1.button("30 jours", type="secondary"):
#         st.markdown("Pluviométrie 30 jours")
#         update_map("Pluviométrie 30 jours", geometrie_departements, df_indicators)
#     if pluviometrie_2.button("3 mois", type="secondary"):
#         st.markdown("Pluviométrie 3 mois")
#         update_map("Pluviométrie 3 mois", geometrie_departements, df_indicators)
#     if pluviometrie_3.button("6 mois", type="secondary"):
#         st.markdown("Pluviométrie 6 mois")
#         update_map("Pluviométrie 6 mois", geometrie_departements, df_indicators)
    
#     tab_2.markdown("## TODO")

# if __name__ == "__main__":
#     main()
