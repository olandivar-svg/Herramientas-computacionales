# %%
#Librerías
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import folium
from folium.plugins import HeatMap, MarkerCluster
from folium import FeatureGroup, LayerControl
import warnings
from IPython.display import display
from tqdm import tqdm
warnings.filterwarnings("ignore")

# %%
#Datos DENUE
df = pd.read_csv('data/DENUE_CDMX.csv', encoding='latin1', low_memory=False)

df_cdmx = df[df['entidad'] == 'Ciudad de México'].copy().reset_index(drop=True)

df_cdmx['latitud'] = pd.to_numeric(df_cdmx['latitud'], errors='coerce')
df_cdmx['longitud'] = pd.to_numeric(df_cdmx['longitud'], errors='coerce')

df_cdmx = df_cdmx.dropna(subset=['latitud', 'longitud'])

df_cdmx['cve_ent'] = df_cdmx['cve_ent'].astype(str).str.zfill(2)
df_cdmx['cve_mun'] = df_cdmx['cve_mun'].astype(str).str.zfill(3)
df_cdmx['cve_loc'] = df_cdmx['cve_loc'].astype(str).str.zfill(4)
df_cdmx['ageb']    = df_cdmx['ageb'].astype(str)
df_cdmx['CVEGEO']  = df_cdmx['cve_ent'] + df_cdmx['cve_mun'] + df_cdmx['cve_loc'] + df_cdmx['ageb']

CODIGO_CAFE = [722515] 
CODIGOS_CULTURA = [711110, 711120, 711130, 711190, 712110, 712120, 713940, 713990, 813110, 813120, 813210, 814110]

# %%
#GeoDataFrame
gdf_pts = gpd.GeoDataFrame(
    df_cdmx,
    geometry=gpd.points_from_xy(df_cdmx['longitud'], df_cdmx['latitud']),
    crs="EPSG:4326"
)

# %%
#AGEBS
df_ageb = pd.read_csv("data/RESAGEBURB_09CSV20.csv", encoding="latin1")

if 'ï»¿ENTIDAD' in df_ageb.columns:
    df_ageb.rename(columns={'ï»¿ENTIDAD':'ENTIDAD'}, inplace=True)

df_ageb['CVEGEO'] = (
    df_ageb['ENTIDAD'].astype(str).str.zfill(2) +
    df_ageb['MUN'].astype(str).str.zfill(3) +
    df_ageb['LOC'].astype(str).str.zfill(4) +
    df_ageb['AGEB'].astype(str)
)

for col in ['POBTOT','TOTHOG','VIVTOT','VPH_INTER']:
    df_ageb[col] = pd.to_numeric(df_ageb[col], errors='coerce')

# %%
#Score
df_ageb['prop_internet'] = df_ageb['VPH_INTER'] / df_ageb['VIVTOT']
df_ageb['atractivo'] = (
    0.5 * df_ageb['POBTOT'] +
    0.3 * df_ageb['TOTHOG'] +
    0.2 * df_ageb['prop_internet'].fillna(0) * 1000
)

df_ageb['atractivo_norm'] = (
    (df_ageb['atractivo'] - df_ageb['atractivo'].min()) /
    (df_ageb['atractivo'].max() - df_ageb['atractivo'].min())
)

# %%
#Competencias Cafeterías
df_cafe = df_cdmx[df_cdmx['codigo_act'].isin(CODIGO_CAFE)].copy()

comp_ageb = df_cafe.groupby('CVEGEO', as_index=False).agg(rest_count=('id','count'))

ageb_comp = df_ageb[['CVEGEO','POBTOT','atractivo_norm']].merge(comp_ageb, on='CVEGEO', how='left')
ageb_comp['rest_count'] = ageb_comp['rest_count'].fillna(0)

ageb_comp['rest_x_1000hab'] = (ageb_comp['rest_count'] / ageb_comp['POBTOT'].replace(0, np.nan)) * 1000
ageb_comp['rest_x_1000hab'] = ageb_comp['rest_x_1000hab'].fillna(0)

mn, mx = ageb_comp['rest_x_1000hab'].min(), ageb_comp['rest_x_1000hab'].max()
ageb_comp['competencia_norm'] = (ageb_comp['rest_x_1000hab'] - mn) / (mx - mn + 1e-9)

# %%
#Centro Culturales
df_cultura = df_cdmx[df_cdmx['codigo_act'].isin(CODIGOS_CULTURA)].copy()
cultura_por_ageb = df_cultura.groupby('CVEGEO', as_index=False).size().rename(columns={'size':'centros_culturales'})

ageb_comp = ageb_comp.merge(cultura_por_ageb, on='CVEGEO', how='left')
ageb_comp['centros_culturales'] = ageb_comp['centros_culturales'].fillna(0)

mn_c, mx_c = ageb_comp['centros_culturales'].min(), ageb_comp['centros_culturales'].max()
ageb_comp['centros_culturales_norm'] = (ageb_comp['centros_culturales'] - mn_c) / (mx_c - mn_c + 1e-9)

# %%
#Score Final
w_pop, w_cultura, w_comp = 0.5, 0.5, 1.0

ageb_comp['score_oportunidad'] = (
    w_pop * ageb_comp['atractivo_norm'] +
    w_cultura * ageb_comp['centros_culturales_norm'] -
    w_comp * ageb_comp['competencia_norm']
)

# %%
#Centros AGEB
centros = (
    gdf_pts.groupby('CVEGEO', as_index=False)
           .agg(lat_mean=('latitud','mean'),
                lon_mean=('longitud','mean'))
)
ageb_comp_geo = ageb_comp.merge(centros, on='CVEGEO', how='left')
# %%
#Candidatos Optimizados
pts_comp = df_cafe[['latitud','longitud']].dropna().values
if pts_comp.shape[0] == 0:
    print("No hay puntos de cafeterías válidos para calcular competencia.")
else:
    tree = BallTree(np.radians(pts_comp), metric='haversine')

    top_agebs = ageb_comp_geo.dropna(subset=['lat_mean','lon_mean']).sort_values('score_oportunidad', ascending=False).head(15)

    def mejor_punto_score(c_lat, c_lon, score, r_m=600, n=500):
        R = 6371000.0
        rs = r_m * np.sqrt(np.random.rand(n))
        thetas = 2*np.pi*np.random.rand(n)
        dlat = (rs / R) * (180/np.pi)
        dlon = (rs / (R*np.cos(np.deg2rad(c_lat)))) * (180/np.pi)
        lats = c_lat + dlat * np.sin(thetas)
        lons = c_lon + dlon * np.cos(thetas)
        cand = np.vstack([lats, lons]).T

        dist_rad, _ = tree.query(np.radians(cand), k=1)
        dist_m = dist_rad[:,0] * R

        score_bonus = score

        final_score = dist_m + 5000*score_bonus
        i = np.argmax(final_score)
        return cand[i,0], cand[i,1], dist_m[i]

    candidatos = []
    for _, r in tqdm(top_agebs.iterrows(), total=len(top_agebs)):
        lat_best, lon_best, dmin = mejor_punto_score(r['lat_mean'], r['lon_mean'], r['score_oportunidad'])
        candidatos.append({
            'CVEGEO': r['CVEGEO'],
            'lat_centro': r['lat_mean'],
            'lon_centro': r['lon_mean'],
            'lat_candidato': lat_best,
            'lon_candidato': lon_best,
            'dist_comp_mas_cercano_m': float(dmin),
            'score_oportunidad': r['score_oportunidad']
        })

    df_candidatos = pd.DataFrame(candidatos)
    display(df_candidatos.head())
    print(f"{len(df_candidatos)} candidatos generados y guardados.")

# %%
#Coordenadas Centros Culturales
df_cultura = df_cdmx[df_cdmx['codigo_act'].isin(CODIGOS_CULTURA)].copy()

df_cultura['cve_ent'] = df_cultura['cve_ent'].astype(str).str.zfill(2)
df_cultura['cve_mun'] = df_cultura['cve_mun'].astype(str).str.zfill(3)
df_cultura['cve_loc'] = df_cultura['cve_loc'].astype(str).str.zfill(4)
df_cultura['ageb'] = df_cultura['ageb'].astype(str).str.zfill(4)
df_cultura['CVEGEO'] = (
    df_cultura['cve_ent'] + df_cultura['cve_mun'] + df_cultura['cve_loc'] + df_cultura['ageb']
)

centros_cultura = (
    df_cultura.groupby('CVEGEO', as_index=False)
              .agg(lat_mean_cultura=('latitud', 'mean'),
                   lon_mean_cultura=('longitud', 'mean'),
                   total_centros=('codigo_act', 'count'))
)

ageb_cultura_geo = ageb_comp_geo.merge(centros_cultura, on='CVEGEO', how='left')
ageb_cultura_geo['total_centros'] = ageb_cultura_geo['total_centros'].fillna(0)


# %% 
#Mapa
m = folium.Map(location=[19.4326,-99.1332], zoom_start=12, control_scale=True)

heat_data = ageb_comp_geo[['lat_mean','lon_mean','score_oportunidad']].dropna().values.tolist()
weights = np.array([w for _,_,w in heat_data])
mn, mx = weights.min(), weights.max()
heat_data_norm = [[lat, lon, (w-mn)/(mx-mn+1e-9)] for lat, lon, w in heat_data]
heat_layer = FeatureGroup(name="Score de oportunidad")
HeatMap(heat_data_norm, radius=22, blur=26, min_opacity=0.3).add_to(heat_layer)
heat_layer.add_to(m)

cafe_layer = FeatureGroup(name="Candidatos Café", show=True)
for _, row in df_candidatos.iterrows():
    folium.Marker(
        location=[row['lat_candidato'], row['lon_candidato']],
        popup=(f"AGEB: {row['CVEGEO']}<br>"
               f"Score: {row['score_oportunidad']:.3f}<br>"
               f"Dist. competidor más cercano: {int(row['dist_comp_mas_cercano_m'])} m"),
        icon=folium.Icon(color="darkgreen", icon="star")
    ).add_to(cafe_layer)
cafe_layer.add_to(m)

cultura_layer = FeatureGroup(name="Centros culturales", show=True)
for _, r in ageb_cultura_geo.iterrows():
    if r['total_centros'] > 0 and pd.notnull(r['lat_mean_cultura']):
        folium.CircleMarker(
            location=[r['lat_mean_cultura'], r['lon_mean_cultura']],
            radius=3 + int(r['total_centros']/2),
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=f"AGEB: {r['CVEGEO']}<br>Centros culturales: {int(r['total_centros'])}"
        ).add_to(cultura_layer)
cultura_layer.add_to(m)

LayerControl(collapsed=False).add_to(m)
m