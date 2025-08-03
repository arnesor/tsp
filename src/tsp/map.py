import webbrowser
from pathlib import Path

import folium
import pandas as pd
from node_schema import NodeInputModel


def show_map(tsp_map: folium.Map) -> None:
    """Show the map in a browser window."""
    filename = "map.html"
    tsp_map.save(filename)
    webbrowser.open(filename, new=2)


pd.set_option("display.precision", 2)

filename = Path(__file__).parent.parent.parent / "data" / "Paris.csv"
df_sites = pd.read_csv(filename, skipinitialspace=True)
validated = NodeInputModel.validate(df_sites)
print(df_sites)

avg_location = df_sites[["lat", "lon"]].mean()
map_paris = folium.Map(location=avg_location.tolist(), zoom_start=13)

for site in df_sites.itertuples():
    marker = folium.Marker(location=(site.lat, site.lon), tooltip=site.name)
    marker.add_to(map_paris)

show_map(map_paris)
