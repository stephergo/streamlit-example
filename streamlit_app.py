"""
Application Streamlit de géocodage d'adresses avec visualisation sur carte
"""
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
import io
from datetime import timedelta
from enum import Enum, auto
from geopy.geocoders import Nominatim, BANFrance
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
from typing import Optional, Tuple
from dataclasses import dataclass
import random
import time
import uuid  # Pour générer des clés uniques mais stables

# Configuration de la page
st.set_page_config(
    page_title="Géocodage d'adresses",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class PinStyle(Enum):
    """Available pin styles for markers"""
    CLASSIC = "classic"
    MODERN = "modern"
    ROUNDED = "rounded"
    TARGET = "target"
    MAN = "man"
    CIRCLE = "circle"

@dataclass
class MapConfig:
    """Configuration pour la création de carte"""
    title: str
    center: List[float] = (45.7355138, 4.8941066)
    zoom: int = 9
    tile_style: str = 'cartodbpositron'
    cluster_markers: bool = True
    
@dataclass
class PinConfig:
    """Configuration pour les pins"""
    style: PinStyle
    primary_color: str = "#ffffff"
    secondary_color: str = "#FF0000"
    third_color: str = "#FF0000"
    stroke_color: str = "#000000"
    opacity: float = 0.9
    scale: float = 0.75

# Classes de base pour le géocodage
class GeocodingMode(Enum):
    NOMINATIM_ONLY = auto()
    BANFRANCE_ONLY = auto()
    NOMINATIM_THEN_BANFRANCE = auto()
    FALLBACK_TO_CITY = auto()

@dataclass
class GeocodingResult:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    service: str = "non_trouvé"
    error: str = ""
    address_type: str = "complète"

class GeocodingManager:
    def __init__(self, app_name: str, mode: GeocodingMode = GeocodingMode.FALLBACK_TO_CITY):
        self.mode = mode
        self.app_name = app_name
        self.nominatim = None
        self.banfrance = None
        self._initialize_services()

    def _initialize_services(self):
        if self.mode in [GeocodingMode.NOMINATIM_ONLY, GeocodingMode.NOMINATIM_THEN_BANFRANCE, GeocodingMode.FALLBACK_TO_CITY]:
            unique_id = f"{random.randint(1000, 9999)}"
            user_agent = f"{self.app_name}-nominatim-{unique_id}"
            geolocator = Nominatim(user_agent=user_agent, timeout=10)
            self.nominatim = RateLimiter(geolocator.geocode, min_delay_seconds=1)

        if self.mode in [GeocodingMode.BANFRANCE_ONLY, GeocodingMode.NOMINATIM_THEN_BANFRANCE, GeocodingMode.FALLBACK_TO_CITY]:
            unique_id = f"{random.randint(1000, 9999)}"
            user_agent = f"{self.app_name}-ban-{unique_id}"
            geolocator = BANFrance(user_agent=user_agent, timeout=10)
            self.banfrance = RateLimiter(geolocator.geocode, min_delay_seconds=1.5)

    def geocode_address(self, street: str, postal_code: str, city: str) -> GeocodingResult:
        full_address = f"{street}, {postal_code}, {city}".strip().replace('\n', ' ')
        
        if self.mode == GeocodingMode.FALLBACK_TO_CITY:
            return self._fallback_geocoding(street, postal_code, city, full_address)
        elif self.mode == GeocodingMode.NOMINATIM_ONLY:
            return self._geocode_with_nominatim(full_address)
        elif self.mode == GeocodingMode.BANFRANCE_ONLY:
            return self._geocode_with_banfrance(full_address)
        elif self.mode == GeocodingMode.NOMINATIM_THEN_BANFRANCE:
            result = self._geocode_with_nominatim(full_address)
            if result.latitude is None:
                time.sleep(1)
                result = self._geocode_with_banfrance(full_address)
            return result
        
        return GeocodingResult(error="Mode de géocodage non reconnu")

    def _geocode_with_nominatim(self, address: str, service_name: str = "nominatim") -> GeocodingResult:
        try:
            location = self.nominatim(address)
            if location:
                return GeocodingResult(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    service=service_name
                )
        except Exception as e:
            return GeocodingResult(error=f"Erreur avec {service_name}: {str(e)}")
        return GeocodingResult(error=f"Aucun résultat trouvé avec {service_name}")

    def _geocode_with_banfrance(self, address: str) -> GeocodingResult:
        try:
            location = self.banfrance(address)
            if location:
                return GeocodingResult(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    service="banfrance"
                )
        except Exception as e:
            return GeocodingResult(error=f"Erreur avec BAN France: {str(e)}")
        return GeocodingResult(error="Aucun résultat trouvé avec BAN France")

    def _fallback_geocoding(self, street: str, postal_code: str, city: str, full_address: str) -> GeocodingResult:
        # Tentative avec adresse complète via Nominatim
        result = self._geocode_with_nominatim(full_address)
        if result.latitude is not None:
            result.address_type = "complète_nominatim"
            return result
        
        time.sleep(1)
        
        # Tentative avec adresse complète via BAN France
        result = self._geocode_with_banfrance(full_address)
        if result.latitude is not None:
            result.address_type = "complète_banfrance"
            return result
        
        time.sleep(1)
        
        # Fallback sur ville/CP
        city_only_address = f"{postal_code}, {city}".strip()
        result = self._geocode_with_nominatim(city_only_address, "nominatim_city")
        if result.latitude is not None:
            result.address_type = "ville_cp"
        return result

def init_session_state(map:MapConfig, pin:PinConfig)->None:
    """Initialise les variables de session"""
    defaults = {
        'map_data': None,
        'geocoded_df': None,
        'file_uploaded': False,
        'geocoding_complete': False,
        'df': None,
        'geocoding_stats': None,
        'map': None,
        # Options de personnalisation de la carte
        'use_clusters': map.cluster_markers,
        'pin_scale': f'{pin.scale}',
        'opacity': f'{pin.opacity}',
        'marker_color': f'{pin.primary_color}',
        'marker_color1': f'{pin.secondary_color}',
        'secondary_color': f'{pin.secondary_color}',# Rouge en hexadécimal au lieu de 'red'
        'map_title': f'{map.title}',
        'custom_popup_fields': ['adresse_complete', 'service_geocodage', 'type_adresse'],
        'available_columns': [],  # Stockera les colonnes disponibles du fichier Excel importé
        'field_labels': {},  # Stockera les libellés personnalisés pour chaque champ
        'tooltip_field': 'adresse_complete',  # Champ à utiliser pour le tooltip
        'tooltip_max_length': 50,  # Longueur maximale du tooltip
        'rerun_in_progress': False,  # Flag pour éviter les reruns en cascade
        'map_stabilized': False,    # Flag pour stabiliser l'affichage de la carte
        'last_map_update': None,    # Timestamp de la dernière mise à jour de carte
        'map_key': str(uuid.uuid4())  # Clé unique mais stable pour la carte
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def apply_custom_styles():
    """Applique les styles CSS personnalisés"""
    st.markdown('''
        <style>
        .stApp {
            max-width: 1920px;
            margin: 0 auto;
        }
        .element-container {
            margin-bottom: 0.5rem !important;
        }
        .stButton {
            margin-top: 0.5rem !important;
        }
        div[data-testid="stVerticalBlock"] > div {
            padding-bottom: 0.5rem !important;
        }
        .element-container iframe {
            margin-bottom: 0.5rem !important;
        }        
        .upload-box {
            border: 2px dashed #4a90e2;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: #f8f9fa;
        }
        .success-message {
            padding: 10px;
            border-radius: 5px;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            margin: 10px 0;
        }
        .error-message {
            padding: 10px;
            border-radius: 5px;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            margin: 10px 0;
        }
        .info-box {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .map-container {
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
        }
        </style>
    ''', unsafe_allow_html=True)

def show_sidebar_options():
    """Affiche les options de personnalisation dans le panneau latéral"""
    with st.sidebar:
        st.title("⚙️ Options d'affichage")
        # Section Titre
        st.header("📝 Carte")
        map_title = st.text_input(
            "Titre de la carte",
            value=st.session_state.map_title,
            help="Ce titre apparaîtra sur la carte"
        )
                
        # Section Marqueurs
        st.header("🎯 Marqueurs")
        with st.expander("Personnalisation des marqueurs"):
            use_clusters = st.toggle(
                "Marqueurs regroupés",
                value=st.session_state.use_clusters,
                help="Active/désactive le regroupement des marqueurs proches"
            )
            
            col1,col2,col3=st.columns([1,1,2])
            # Sélection de la couleur
            with col1:
                marker_color = st.color_picker(
                    "Couleur1",
                    value=st.session_state.marker_color,
                    #help="Choisissez la couleur des marqueurs sur la carte"
            )
            with col2:
                marker_color1 = st.color_picker(
                    "Couleur2",
                    value=st.session_state.marker_color1,
                    #help="Choisissez la couleur des marqueurs sur la carte"
            )

            
            # selection de l'opacité
            opacity = st.slider(
                "Opacité des marqueurs",
                min_value=0.1,
                max_value=1.0,
                value=float(st.session_state.opacity),
                help="Réglez l'opacité des marqueurs sur la carte"
            )
            
            # Échelle de l'icône de marqueur
            pin_scale = st.slider(
                "Échelle de l'icône de marqueur",
                min_value=0.1,
                max_value=2.0,
                value=float(st.session_state.pin_scale),
                help="Réglez la taille de l'icône de marqueur sur la carte"
            )
        

        # Section Popup et Tooltip
        with st.expander("Personnalisation des infobulles   💭 "):
                
            st.header("💭 Popups & Tooltips")
            
            # Utiliser les colonnes disponibles du DataFrame si elles existent
            available_fields = st.session_state.available_columns
            
            if st.session_state.file_uploaded and len(available_fields) > 0:
                st.subheader("📌 Popup (clic)")
                # Sélection des champs à afficher dans les popups
                custom_popup_fields = st.multiselect(
                    "Informations à afficher dans les popups",
                    options=available_fields,
                    default=st.session_state.custom_popup_fields if set(st.session_state.custom_popup_fields).issubset(set(available_fields)) else available_fields[:min(3, len(available_fields))],
                    help="Sélectionnez les informations à afficher dans les popups des marqueurs"
                )
                
                # Interface pour personnaliser les labels des champs
                st.markdown("##### Labels personnalisés pour les champs")
                field_labels = {}
                
                # Si la sélection a changé, réinitialiser les labels
                if set(custom_popup_fields) != set(st.session_state.custom_popup_fields):
                    st.session_state.field_labels = {field: field.replace('_', ' ').title() for field in custom_popup_fields}
                
                # Création des champs de texte pour personnaliser les labels
                for field in custom_popup_fields:
                    default_label = st.session_state.field_labels.get(field, field.replace('_', ' ').title())
                    field_labels[field] = st.text_input(
                        f"Label pour {field}",
                        value=default_label,
                        key=f"label_{field}"
                    )
                
                # Mise à jour des labels personnalisés
                st.session_state.field_labels = field_labels
                
                # Section Tooltip (survol)
                st.subheader("🔍 Tooltip (survol)")
                tooltip_field = st.selectbox(
                    "Champ à afficher au survol",
                    options=available_fields,
                    index=available_fields.index(st.session_state.tooltip_field) if st.session_state.tooltip_field in available_fields else 0,
                    help="Ce champ sera affiché lorsque l'utilisateur survole un marqueur"
                )
                
                tooltip_max_length = st.slider(
                    "Longueur maximale du tooltip",
                    min_value=10,
                    max_value=200,
                    value=st.session_state.tooltip_max_length,
                    help="Nombre maximal de caractères à afficher dans le tooltip"
                )
                
                # Mise à jour des valeurs de session pour le tooltip
                if tooltip_field != st.session_state.tooltip_field:
                    st.session_state.tooltip_field = tooltip_field
                    options_changed = True
                    
                if tooltip_max_length != st.session_state.tooltip_max_length:
                    st.session_state.tooltip_max_length = tooltip_max_length
                    options_changed = True
            else:
                # Texte par défaut si aucun fichier n'a été importé
                st.info("Importez un fichier Excel pour personnaliser les champs à afficher")
                custom_popup_fields = st.session_state.custom_popup_fields
        

        st.markdown(
            """
            ---
            Follow me on:
            
            LinkedIn → [Stephane DENIS](http://www.linkedin.com/in/stephane-denis-07344527)
            
            Copyright(c) 2024 - Stephane DENIS

            """
        )        
        # Détecter les changements
        options_changed = False
        if use_clusters != st.session_state.use_clusters:
            st.session_state.use_clusters = use_clusters
            options_changed = True
        if marker_color != st.session_state.marker_color:
            st.session_state.marker_color = marker_color
            options_changed = True
        if marker_color1 != st.session_state.marker_color1:
            st.session_state.marker_color1 = marker_color1
            options_changed = True
        if map_title != st.session_state.map_title:
            st.session_state.map_title = map_title
            options_changed = True
        if st.session_state.file_uploaded and len(available_fields) > 0 and custom_popup_fields != st.session_state.custom_popup_fields:
            st.session_state.custom_popup_fields = custom_popup_fields
            options_changed = True
        if opacity != float(st.session_state.opacity):
            st.session_state.opacity = str(opacity)
            options_changed = True
        if pin_scale != float(st.session_state.pin_scale):
            st.session_state.pin_scale = str(pin_scale)
            options_changed = True
            
        # Mettre à jour la carte si nécessaire
        if options_changed and st.session_state.geocoding_complete:
            # Réinitialiser le flag de stabilisation pour permettre une mise à jour intentionnelle
            st.session_state.map_stabilized = False
            st.session_state.map = create_folium_map(st.session_state.geocoded_df)
            
            # Éviter les reruns en cascade avec un flag et ne pas recharger si la carte est déjà stable
            if not st.session_state.rerun_in_progress and not st.session_state.map_stabilized:
                st.session_state.last_map_update = time.time()
                st.session_state.rerun_in_progress = True
                st.rerun()
        
        # Bouton d'actualisation manuel
        if st.session_state.geocoding_complete:
            if st.button("🔄 Actualiser la carte"):
                # Forcer la mise à jour même si la carte est stabilisée
                st.session_state.map_stabilized = False
                st.session_state.map = create_folium_map(st.session_state.geocoded_df)
                
                # Éviter les reruns en cascade avec un flag
                if not st.session_state.rerun_in_progress:
                    st.session_state.last_map_update = time.time()
                    st.session_state.rerun_in_progress = True
                    st.rerun()

def create_custom_popup_content(row: pd.Series) -> str:
    """Crée un contenu de popup personnalisé basé sur les champs sélectionnés"""
    content = '<div style="width:200px;padding:10px;background-color:#fff;border-radius:5px;">'
    
    # Utiliser les labels personnalisés s'ils existent, sinon utiliser le nom du champ formaté
    for field in st.session_state.custom_popup_fields:
        # Pour les séries pandas, nous utilisons .get() avec un second argument pour éviter les erreurs
        # quand une clé n'existe pas, et pd.notna() pour vérifier si la valeur est valide
        field_value = row.get(field, None)
        if field_value is not None and pd.notna(field_value):
            # Utiliser le label personnalisé ou formater le nom du champ
            field_label = st.session_state.field_labels.get(field, field.replace('_', ' ').title())
            content += f'<strong>{field_label}:</strong> {field_value}<br>'
    
    content += '</div>'
    return content

def create_pin_icon() -> DivIcon:
    """Crée une icône de marqueur personnalisée"""
    pin_html = f'''
    <div>
        <svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" style="enable-background:new 0 0 128 128;" transform="scale({st.session_state.pin_scale})">
            <style type="text/css">
                .pin-main{{fill:{st.session_state.marker_color1};fill-opacity:{st.session_state.opacity};stroke:#000000;stroke-width:2}}
                .pin-circle{{fill:{st.session_state.marker_color};stroke:#000000;stroke-width:2}}
            </style>
            <g>
                <path class="pin-main" d="M64,0C35.8,0,12.8,23,12.8,51.2s51.2,76.8,51.2,76.8s51.2-48.6,51.2-76.8S92.2,0,64,0z"/>
                <circle class="pin-circle" cx="64" cy="45" r="25"/>
            </g>
        </svg>
    </div>
    '''
    return DivIcon(
        icon_size=(32, 32),
        icon_anchor=(16, 32),
        html=pin_html
    )

def create_folium_map(df):
    """Crée une carte Folium avec les options personnalisées"""
    # Vérification et conversion des coordonnées
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Filtrer les coordonnées valides
    valid_coords = df[df['latitude'].notna() & df['longitude'].notna()]
    
    if len(valid_coords) > 0:
        center_lat = valid_coords['latitude'].mean()
        center_lon = valid_coords['longitude'].mean()
        zoom_start = 6
    else:
        center_lat, center_lon = 46.227638, 2.213749  # Centre de la France
        zoom_start = 5

    # Création de la carte
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='cartodbpositron'
    )
    
    # Ajout du titre sur la carte
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 500px; 
                height: 40px; 
                z-index:9999; 
                font-size:20px;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                padding: 10px;
                text-align: center;">
        {st.session_state.map_title}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Création du cluster si l'option est activée
    if st.session_state.use_clusters:
        marker_cluster = MarkerCluster().add_to(m)
    
    # Conversion de la couleur hexadécimale en nom de couleur pour Folium
    color_mapping = {
        '#FF0000': 'red',
        '#0000FF': 'blue',
        '#00FF00': 'green',
        '#FFA500': 'orange',
        '#800080': 'purple',
        '#000000': 'black',
    }
    
    #marker_color = color_mapping.get(st.session_state.marker_color, 'red')  # 'red' comme couleur par défaut
    
    # Pour debugging
    if len(st.session_state.custom_popup_fields) == 0:
        st.warning("Aucun champ sélectionné pour les popups. Utilisez le panneau latéral pour personnaliser les popups.")
    
    # Ajout des marqueurs avec la couleur convertie
    for idx, row in valid_coords.iterrows():
        try:
            popup_content = create_custom_popup_content(row)
            
            # Paramétrer le tooltip en fonction du champ sélectionné
            tooltip_field = st.session_state.tooltip_field
            tooltip_max_length = st.session_state.tooltip_max_length
            
            # Valeur par défaut pour le tooltip si le champ sélectionné n'est pas disponible
            tooltip_value = "..."
            
            if tooltip_field in row and pd.notna(row[tooltip_field]):
                tooltip_text = str(row[tooltip_field])
                # Tronquer si nécessaire et ajouter "..." pour indiquer la troncature
                if len(tooltip_text) > tooltip_max_length:
                    tooltip_value = tooltip_text[:tooltip_max_length] + "..."
                else:
                    tooltip_value = tooltip_text
            
            marker = folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_content,
                tooltip=tooltip_value,
                icon=create_pin_icon(),
            )
            
            if st.session_state.use_clusters:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(m)
            
        except Exception as e:
            st.error(f"Erreur pour la ligne {idx}: {str(e)}")
            continue

    return m

def display_results(geocoded_df, stats):
    """Affiche les résultats du géocodage"""
    # Affichage des statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total d'adresses", stats['total_rows'])
    with col2:
        st.metric("Adresses géocodées", stats['success_count'])
    with col3:
        st.metric("Taux de réussite", f"{stats['success_rate']:.1%}")

    # Création et affichage de la carte
    st.markdown("### 🗺️ Résultats du géocodage")
    
    # Éviter d'invoquer st_folium d'une manière qui déclencherait un rerendering
    # Utiliser une clé unique mais stable pour le composant folium
    if 'map_key' not in st.session_state:
        st.session_state.map_key = str(uuid.uuid4())
    
    # Stabiliser l'affichage de la carte pour éviter les rerenders constants
    folium_static_result = st_folium(
        st.session_state.map, 
        width=1200, 
        height=600,
        returned_objects=[],  # Ne retourne AUCUN objet pour éviter des rerenders
        key=st.session_state.map_key  # Utiliser une clé stable
    )
    
    # Marquer la carte comme stabilisée pour éviter les rafraîchissements inutiles
    if not st.session_state.map_stabilized:
        st.session_state.map_stabilized = True

    # Génération des noms de fichiers basés sur le titre de la carte
    # Remplacer les espaces par des underscores et nettoyer le titre pour un nom de fichier valide
    safe_filename = st.session_state.map_title.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in '_-.')
    
    # Si après nettoyage le nom est vide, utiliser un nom par défaut
    if not safe_filename:
        safe_filename = "carte_geocodage"
    
    html_filename = f"{safe_filename}.html"
    excel_filename = f"{safe_filename}.xlsx"

    # Boutons de téléchargement
    col1, col2 = st.columns(2)
    with col1:
        # Carte HTML
        html_buffer = io.BytesIO()
        st.session_state.map.save(html_buffer, close_file=False)
        st.download_button(
            "📥 Télécharger la carte (HTML)",
            data=html_buffer,
            file_name=html_filename,
            mime="text/html"
        )

    with col2:
        # Fichier Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            geocoded_df.to_excel(writer, sheet_name='Resultats', index=False)
        st.download_button(
            "📥 Télécharger les résultats (Excel)",
            data=excel_buffer,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.success(f"✅ Géocodage terminé en {stats['processing_time']}")

def process_uploaded_file(uploaded_file):
    """Traite le fichier uploadé et retourne un DataFrame"""
    try:
        df = pd.read_excel(uploaded_file, dtype=str)
        required_columns = {'street', 'postalcode', 'city'}
        
        if not all(col in df.columns for col in required_columns):
            st.error("❌ Le fichier doit contenir les colonnes: street, postalcode, city")
            return None
            
        # Mettre à jour les colonnes disponibles pour les popups
        st.session_state.available_columns = list(df.columns)
        
        # Ajouter les colonnes qui seront créées lors du géocodage
        geocoding_columns = ['adresse_complete', 'service_geocodage', 'type_adresse', 'latitude', 'longitude', 'erreur_geocodage']
        for col in geocoding_columns:
            if col not in st.session_state.available_columns:
                st.session_state.available_columns.append(col)
        
        # Mettre à jour les champs par défaut pour les popups si nécessaire
        default_fields = ['adresse_complete', 'service_geocodage', 'type_adresse']
        st.session_state.custom_popup_fields = [f for f in default_fields if f in st.session_state.available_columns]
        
        # Initialiser les labels personnalisés pour les champs par défaut
        st.session_state.field_labels = {
            'adresse_complete': 'Adresse',
            'service_geocodage': 'Service',
            'type_adresse': 'Type'
        }
            
        st.success("✅ Fichier chargé avec succès!")
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
        return None

def perform_geocoding(df):
    """Effectue le géocodage des adresses"""
    geocoder = GeocodingManager(app_name="streamlit_geocoding")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()

    total_rows = len(df)
    results = []
    
    for idx, row in df.iterrows():
        result = geocoder.geocode_address(
            row['street'],
            row['postalcode'],
            row['city']
        )
        
        results.append({
            'latitude': result.latitude,
            'longitude': result.longitude,
            'service_geocodage': result.service,
            'type_adresse': result.address_type,
            'erreur_geocodage': result.error,
            'adresse_complete': f"{row['street']}, {row['postalcode']}, {row['city']}"
        })
        
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        
        if idx > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / (idx + 1)
            remaining_rows = total_rows - (idx + 1)
            eta_seconds = avg_time_per_row * remaining_rows
            eta = timedelta(seconds=int(eta_seconds))
            status_text.text(f"Traitement en cours... {idx + 1}/{total_rows} (Temps restant estimé: {eta})")
    
    progress_bar.empty()
    status_text.empty()
    
    results_df = pd.DataFrame(results)
    geocoded_df = pd.concat([df, results_df], axis=1)
    
    success_count = sum(pd.notna(results_df['latitude']))
    success_rate = success_count / total_rows
    
    processing_time = timedelta(seconds=int(time.time() - start_time))
    
    return geocoded_df, {
        'total_rows': total_rows,
        'success_count': success_count,
        'success_rate': success_rate,
        'processing_time': processing_time
    }

def main():
    # Application des styles
    apply_custom_styles()
    
    # Réinitialiser le flag de rerun au début de chaque cycle
    if 'rerun_in_progress' in st.session_state and st.session_state.rerun_in_progress:
        st.session_state.rerun_in_progress = False
    
    # Gestion de la stabilité de la carte
    current_time = time.time()
    if 'last_map_update' in st.session_state and st.session_state.last_map_update:
        # Empêcher les mises à jour trop fréquentes (moins de 2 secondes d'écart)
        if current_time - st.session_state.last_map_update < 2:
            st.session_state.map_stabilized = True
        elif current_time - st.session_state.last_map_update > 10:
            # Réinitialiser après un certain temps pour permettre des mises à jour intentionnelles
            st.session_state.map_stabilized = False
    
    map = MapConfig(
    title="Ma carte",
    center=[45.7355138, 4.8941066],
    cluster_markers=False
    )
    
    pin = PinConfig(
    style=PinStyle.MODERN,
    primary_color="#00FF00",
    secondary_color="#FFFFFF",
    third_color="#004400",
    opacity=0.9,
    scale=0.8
)
    
    # Initialisation des états
    init_session_state(map,pin)
    
    # Affichage du panneau latéral avec les options
    show_sidebar_options()
    
    # Contenu principal
    st.title("🌍 Application de Géocodage d'Adresses")
    st.markdown("""
        Cette application permet de géocoder des adresses à partir d'un fichier Excel
        et de visualiser les résultats sur une carte interactive.
    """)
    with st.expander("**📖 Comment utiliser cette application**"):
        st.markdown("""
            1. **Chargement du fichier**: Cliquez sur le bouton ci-dessous pour charger un fichier Excel (.xlsx) contenant les colonnes: `street`, `postalcode`, `city`.
            2. **Géocodage**: Cliquez sur le bouton pour lancer le géocodage des adresses.
            3. **Résultats**: Les adresses géocodées seront affichées sur la carte et vous pourrez télécharger les résultats.
        """)
        st.info("💡 **Astuce**: Utilisez les options dans le panneau latéral pour personnaliser l'affichage de la carte.")
    
    # Section de chargement de fichier
    st.markdown("<h2>📁 Chargement du fichier</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier Excel (.xlsx)",
        type=['xlsx'],
        help="Le fichier doit contenir les colonnes: street, postalcode, city"
    )
    
    # Traitement du fichier uploadé
    if uploaded_file is not None and not st.session_state.file_uploaded:
        df = process_uploaded_file(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.file_uploaded = True
    
    # Section de géocodage
    if st.session_state.file_uploaded and not st.session_state.geocoding_complete:
        st.markdown("<h2>🔍 Géocodage</h2>", unsafe_allow_html=True)
        
        if st.button("Lancer le géocodage", type="primary"):
            geocoded_df, stats = perform_geocoding(st.session_state.df)
            st.session_state.geocoded_df = geocoded_df
            st.session_state.geocoding_stats = stats
            st.session_state.geocoding_complete = True
            st.session_state.map = create_folium_map(geocoded_df)
    
    # Affichage des résultats
    if st.session_state.geocoding_complete:
        display_results(st.session_state.geocoded_df, st.session_state.geocoding_stats)

    # Éviter les reruns en cascade avec un flag
    if not st.session_state.rerun_in_progress and not st.session_state.map_stabilized:
        st.session_state.last_map_update = time.time()
        st.session_state.rerun_in_progress = True
        st.rerun()
        
    # Éviter les reruns en cascade avec un flag
    if not st.session_state.rerun_in_progress and not st.session_state.map_stabilized:
        st.session_state.last_map_update = time.time()
        st.session_state.rerun_in_progress = True
        st.rerun()

if __name__ == "__main__":
    main()
