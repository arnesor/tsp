import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import folium
import pandas as pd

from tsp.tsp_data import TspData


def show_map(tsp_map: folium.Map) -> None:
    """Show the map in a browser window."""
    filename = Path("map.html")
    tsp_map.save(filename)
    webbrowser.open(filename, new=2)


pd.set_option("display.precision", 2)

filename = Path(__file__).parent.parent.parent / "data" / "Paris.csv"
sites = TspData.from_csv(filename)
print(sites)

avg_location = sites.df()[["lat", "lon"]].mean()
map_paris = folium.Map(location=avg_location.tolist(), zoom_start=13)

for site in sites.df().itertuples():
    site_loc = cast(Sequence[float], (site.lat, site.lon))
    site_name = cast(str, site.name)
    marker = folium.Marker(location=site_loc, tooltip=site_name)
    marker.add_to(map_paris)

show_map(map_paris)
