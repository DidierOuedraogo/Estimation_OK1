import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io
import zipfile
import math
import base64
from scipy.linalg import solve
from stqdm import stqdm
import seaborn as sns
from PIL import Image
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="MineEstim - Krigeage",
    page_icon="📊",
    layout="wide"
)

# Fonction pour créer un lien de téléchargement
def get_download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        file_type = 'text/csv'
    elif isinstance(object_to_download, plt.Figure):
        buf = BytesIO()
        object_to_download.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        file_type = 'image/png'
    elif isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
        file_type = 'application/zip'
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()
        file_type = 'text/plain'
    
    return f'<a href="data:{file_type};base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Fonctions pour l'estimation par krigeage
def calculate_variogram_value(distance, model_type, nugget, sill, range_value):
    if distance == 0:
        return 0
    
    normalized_distance = distance / range_value
    
    if normalized_distance >= 1:
        return nugget + sill
    
    if model_type == 'spherical':
        structural_component = sill * (1.5 * normalized_distance - 0.5 * normalized_distance**3)
    elif model_type == 'exponential':
        structural_component = sill * (1 - math.exp(-3 * normalized_distance))
    elif model_type == 'gaussian':
        structural_component = sill * (1 - math.exp(-3 * normalized_distance**2))
    else:
        structural_component = sill * normalized_distance  # Linear
        
    return nugget + structural_component

def ordinary_kriging(point, samples, model_type, nugget, sill, range_value, anisotropy):
    if len(samples) == 0:
        return {'estimate': 0, 'variance': 0}
    
    # Si un seul échantillon, retourner sa valeur
    if len(samples) == 1:
        return {'estimate': samples[0]['value'], 'variance': nugget + sill}
    
    n = len(samples)
    
    # Matrice de covariance entre échantillons
    K = np.zeros((n + 1, n + 1))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                K[i, j] = nugget
            else:
                # Distance entre échantillons avec anisotropie
                dx = (samples[i]['x'] - samples[j]['x']) / anisotropy['x']
                dy = (samples[i]['y'] - samples[j]['y']) / anisotropy['y']
                dz = (samples[i]['z'] - samples[j]['z']) / anisotropy['z']
                
                h = math.sqrt(dx**2 + dy**2 + dz**2)
                
                # Covariance = palier - semivariance
                semivariance = calculate_variogram_value(h, model_type, nugget, sill, range_value)
                cov = nugget + sill - semivariance
                
                K[i, j] = cov
        
        # Contrainte de biais nul
        K[i, n] = 1
        K[n, i] = 1
    
    # Aucun biais sur le multiplicateur de Lagrange
    K[n, n] = 0
    
    # Vecteur de covariance entre le point à estimer et les échantillons
    k = np.zeros(n + 1)
    
    for i in range(n):
        # Distance entre le point à estimer et l'échantillon avec anisotropie
        dx = (point['x'] - samples[i]['x']) / anisotropy['x']
        dy = (point['y'] - samples[i]['y']) / anisotropy['y']
        dz = (point['z'] - samples[i]['z']) / anisotropy['z']
        
        h = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Covariance = palier - semivariance
        semivariance = calculate_variogram_value(h, model_type, nugget, sill, range_value)
        cov = nugget + sill - semivariance
        
        k[i] = cov
    
    # Contrainte de biais nul
    k[n] = 1
    
    try:
        # Résolution du système linéaire K.w = k
        w = solve(K, k)
        
        # Calcul de l'estimation
        estimate = sum(w[i] * samples[i]['value'] for i in range(n))
        
        # Calcul de la variance de krigeage
        variance = nugget + sill
        for i in range(n):
            variance -= w[i] * k[i]
        variance -= w[n] * k[n]
        
        return {'estimate': estimate, 'variance': max(0, variance)}
    except Exception as e:
        # En cas de problème numérique, utiliser IDW comme fallback
        weighted_sum = 0
        weight_sum = 0
        
        for sample in samples:
            if sample['distance'] == 0:
                return {'estimate': sample['value'], 'variance': nugget}
            
            weight = 1 / sample['distance']**2
            weighted_sum += weight * sample['value']
            weight_sum += weight
        
        estimate = weighted_sum / weight_sum if weight_sum > 0 else 0
        return {'estimate': estimate, 'variance': nugget + sill/2}

def calculate_stats(values):
    if len(values) == 0:
        return {}
    
    values = np.array(values)
    return {
        'count': len(values),
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'stddev': np.std(values),
        'variance': np.var(values),
        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
    }

def is_point_inside_envelope(point, min_bounds, max_bounds):
    # Simple bounding box check for envelope
    if not min_bounds or not max_bounds:
        return True
    
    return (point['x'] >= min_bounds['x'] and point['x'] <= max_bounds['x'] and
            point['y'] >= min_bounds['y'] and point['y'] <= max_bounds['y'] and
            point['z'] >= min_bounds['z'] and point['z'] <= max_bounds['z'])

def create_block_model(composites, block_sizes, envelope_bounds=None, use_envelope=True):
    # Déterminer les limites du modèle
    if use_envelope and envelope_bounds:
        min_bounds = envelope_bounds['min']
        max_bounds = envelope_bounds['max']
    else:
        x_values = [comp['X'] for comp in composites]
        y_values = [comp['Y'] for comp in composites]
        z_values = [comp['Z'] for comp in composites]
        
        min_bounds = {
            'x': math.floor(min(x_values) / block_sizes['x']) * block_sizes['x'],
            'y': math.floor(min(y_values) / block_sizes['y']) * block_sizes['y'],
            'z': math.floor(min(z_values) / block_sizes['z']) * block_sizes['z']
        }
        
        max_bounds = {
            'x': math.ceil(max(x_values) / block_sizes['x']) * block_sizes['x'],
            'y': math.ceil(max(y_values) / block_sizes['y']) * block_sizes['y'],
            'z': math.ceil(max(z_values) / block_sizes['z']) * block_sizes['z']
        }
    
    # Créer les blocs
    blocks = []
    
    x_range = np.arange(min_bounds['x'] + block_sizes['x']/2, max_bounds['x'] + block_sizes['x']/2, block_sizes['x'])
    y_range = np.arange(min_bounds['y'] + block_sizes['y']/2, max_bounds['y'] + block_sizes['y']/2, block_sizes['y'])
    z_range = np.arange(min_bounds['z'] + block_sizes['z']/2, max_bounds['z'] + block_sizes['z']/2, block_sizes['z'])
    
    for x in x_range:
        for y in y_range:
            for z in z_range:
                block = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'size_x': block_sizes['x'],
                    'size_y': block_sizes['y'],
                    'size_z': block_sizes['z']
                }
                
                # Vérifier si le bloc est dans l'enveloppe
                if not use_envelope or is_point_inside_envelope(block, min_bounds, max_bounds):
                    blocks.append(block)
    
    return blocks, {'min': min_bounds, 'max': max_bounds}

def estimate_block_model(empty_blocks, composites, kriging_params, search_params):
    estimated_blocks = []
    kriging_variances = []
    
    with st.spinner('Estimation en cours...'):
        progress_bar = st.progress(0)
        
        for idx, block in enumerate(stqdm(empty_blocks)):
            progress = idx / len(empty_blocks)
            progress_bar.progress(progress)
            
            # Chercher les échantillons pour le krigeage
            samples = []
            
            for composite in composites:
                # Appliquer l'anisotropie
                dx = (composite['X'] - block['x']) / kriging_params['anisotropy']['x']
                dy = (composite['Y'] - block['y']) / kriging_params['anisotropy']['y']
                dz = (composite['Z'] - block['z']) / kriging_params['anisotropy']['z']
                
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dx <= search_params['x'] and dy <= search_params['y'] and dz <= search_params['z']:
                    samples.append({
                        'x': composite['X'],
                        'y': composite['Y'],
                        'z': composite['Z'],
                        'value': composite['VALUE'],
                        'distance': distance
                    })
            
            samples.sort(key=lambda x: x['distance'])
            
            if len(samples) >= search_params['min_samples']:
                used_samples = samples[:min(len(samples), search_params['max_samples'])]
                
                # Krigeage ordinaire
                result = ordinary_kriging(
                    block, 
                    used_samples, 
                    kriging_params['model_type'], 
                    kriging_params['nugget'], 
                    kriging_params['sill'], 
                    kriging_params['range'],
                    kriging_params['anisotropy']
                )
                
                estimated_block = block.copy()
                estimated_block['value'] = result['estimate']
                estimated_block['variance'] = result['variance']
                
                estimated_blocks.append(estimated_block)
                kriging_variances.append(result['variance'])
        
        progress_bar.progress(1.0)
    
    return estimated_blocks, kriging_variances

def calculate_tonnage_grade(blocks, density, method, cutoff_value=None, cutoff_min=None, cutoff_max=None):
    if not blocks:
        return {}, {}
    
    # Extraire les valeurs
    values = [block['value'] for block in blocks]
    min_grade = min(values)
    max_grade = max(values)
    
    # Calculer le volume d'un bloc
    block_volume = blocks[0]['size_x'] * blocks[0]['size_y'] * blocks[0]['size_z']
    
    # Générer les coupures
    step = (max_grade - min_grade) / 20
    cutoffs = np.arange(min_grade, max_grade + step, step)
    
    tonnages = []
    grades = []
    metals = []
    cutoff_labels = []
    
    for cutoff in cutoffs:
        cutoff_labels.append(f"{cutoff:.2f}")
        
        if method == 'above':
            filtered_blocks = [block for block in blocks if block['value'] >= cutoff]
        elif method == 'below':
            filtered_blocks = [block for block in blocks if block['value'] <= cutoff]
        elif method == 'between':
            filtered_blocks = [block for block in blocks if cutoff_min <= block['value'] <= cutoff_max]
            
            # Pour la méthode between, on n'a besoin que d'un seul résultat
            if cutoff > min_grade:
                continue
        
        if not filtered_blocks:
            tonnages.append(0)
            grades.append(0)
            metals.append(0)
            continue
        
        tonnage = len(filtered_blocks) * block_volume * density
        
        # Calculer la teneur moyenne pondérée
        weighted_sum = sum(block['value'] * block_volume for block in filtered_blocks)
        grade = weighted_sum / (len(filtered_blocks) * block_volume)
        
        # Calculer le métal contenu
        metal = tonnage * grade
        
        tonnages.append(tonnage)
        grades.append(grade)
        metals.append(metal)
    
    return {
        'cutoffs': cutoff_labels,
        'tonnages': tonnages,
        'grades': grades,
        'metals': metals
    }, {
        'method': method,
        'min_grade': min_grade,
        'max_grade': max_grade
    }

# Fonctions de visualisation
def plot_3d_model(blocks, composites, envelope_bounds=None):
    fig = go.Figure()
    
    # Ajouter les composites
    if composites:
        x = [comp['X'] for comp in composites]
        y = [comp['Y'] for comp in composites]
        z = [comp['Z'] for comp in composites]
        values = [comp['VALUE'] for comp in composites]
        
        composite_scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=values,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Teneur")
            ),
            text=[f"Teneur: {v:.3f}" for v in values],
            name='Composites'
        )
        fig.add_trace(composite_scatter)
    
    # Ajouter les blocs
    if blocks:
        x = [block['x'] for block in blocks]
        y = [block['y'] for block in blocks]
        z = [block['z'] for block in blocks]
        values = [block['value'] for block in blocks]
        variances = [block.get('variance', 0) for block in blocks]
        
        block_scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=values,
                colorscale='Viridis',
                opacity=0.7,
                symbol='square'
            ),
            text=[f"Teneur: {v:.3f}<br>Variance: {var:.3f}" for v, var in zip(values, variances)],
            name='Blocs estimés'
        )
        fig.add_trace(block_scatter)
    
    # Ajouter l'enveloppe (bounding box)
    if envelope_bounds:
        min_x = envelope_bounds['min']['x']
        max_x = envelope_bounds['max']['x']
        min_y = envelope_bounds['min']['y']
        max_y = envelope_bounds['max']['y']
        min_z = envelope_bounds['min']['z']
        max_z = envelope_bounds['max']['z']
        
        # Créer les lignes pour la bounding box
        x_lines = [min_x, max_x, max_x, min_x, min_x, min_x, max_x, max_x, min_x, min_x, min_x, max_x, max_x, max_x, max_x, max_x]
        y_lines = [min_y, min_y, max_y, max_y, min_y, min_y, min_y, min_y, min_y, max_y, max_y, max_y, max_y, min_y, min_y, max_y]
        z_lines = [min_z, min_z, min_z, min_z, min_z, max_z, max_z, min_z, min_z, min_z, max_z, max_z, min_z, min_z, max_z, max_z]
        
        envelope_scatter = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color='rgba(76, 175, 80, 0.5)', width=2),
            name='Enveloppe'
        )
        fig.add_trace(envelope_scatter)
    
    # Mise en page
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0, y=0.9)
    )
    
    return fig

def plot_histogram(values, title, color='steelblue'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculer le nombre de bins
    n_bins = int(1 + 3.322 * math.log10(len(values)))
    
    sns.histplot(values, bins=n_bins, kde=True, color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Fréquence')
    
    return fig

def plot_tonnage_grade(tonnage_data, plot_info=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if plot_info and plot_info['method'] == 'between':
        # Pour la méthode 'between', on utilise un graphique à barres
        fig.add_trace(
            go.Bar(
                x=['Résultat'],
                y=[tonnage_data['tonnages'][0]],
                name='Tonnage',
                marker_color='rgb(63, 81, 181)'
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=['Résultat'],
                y=[tonnage_data['grades'][0]],
                name='Teneur moyenne',
                marker_color='rgb(0, 188, 212)'
            ),
            secondary_y=True
        )
    else:
        # Pour les méthodes 'above' et 'below', on utilise un graphique en ligne
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['tonnages'],
                name='Tonnage',
                fill='tozeroy',
                mode='lines',
                line=dict(color='rgb(63, 81, 181)')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['grades'],
                name='Teneur moyenne',
                mode='lines',
                line=dict(color='rgb(0, 188, 212)')
            ),
            secondary_y=True
        )
    
    fig.update_layout(
        title_text='Courbe Tonnage-Teneur',
        xaxis_title='Teneur de coupure',
        legend=dict(x=0, y=1.1, orientation='h')
    )
    
    fig.update_yaxes(title_text='Tonnage (t)', secondary_y=False)
    fig.update_yaxes(title_text='Teneur moyenne', secondary_y=True)
    
    return fig

def plot_metal_content(tonnage_data, plot_info=None):
    fig = go.Figure()
    
    if plot_info and plot_info['method'] == 'between':
        # Pour la méthode 'between', on utilise un graphique à barres
        fig.add_trace(
            go.Bar(
                x=['Résultat'],
                y=[tonnage_data['metals'][0]],
                name='Métal contenu',
                marker_color='rgb(76, 175, 80)'
            )
        )
    else:
        # Pour les méthodes 'above' et 'below', on utilise un graphique en ligne
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['metals'],
                name='Métal contenu',
                fill='tozeroy',
                mode='lines',
                line=dict(color='rgb(76, 175, 80)')
            )
        )
    
    fig.update_layout(
        title_text='Métal contenu',
        xaxis_title='Teneur de coupure',
        yaxis_title='Métal contenu'
    )
    
    return fig

def plot_kriging_variance(variances):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculer le nombre de bins
    n_bins = min(20, int(1 + 3.322 * math.log10(len(variances))))
    
    sns.histplot(variances, bins=n_bins, kde=True, color='forestgreen', ax=ax)
    ax.set_title('Distribution de la variance de krigeage')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Fréquence')
    
    return fig

# Interface utilisateur Streamlit
st.title("MineEstim - Estimation par krigeage ordinaire")
st.caption("Développé par Didier Ouedraogo, P.Geo")

# Sidebar - Chargement des données et paramètres
with st.sidebar:
    st.header("Données")
    
    uploaded_file = st.file_uploader("Fichier CSV des composites", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"{len(df)} lignes chargées")
        
        # Mappage des colonnes
        st.subheader("Mappage des colonnes")
        
        col_x = st.selectbox("Colonne X", options=df.columns, index=df.columns.get_loc('X') if 'X' in df.columns else 0)
        col_y = st.selectbox("Colonne Y", options=df.columns, index=df.columns.get_loc('Y') if 'Y' in df.columns else 0)
        col_z = st.selectbox("Colonne Z", options=df.columns, index=df.columns.get_loc('Z') if 'Z' in df.columns else 0)
        col_value = st.selectbox("Colonne Teneur", options=df.columns, index=df.columns.get_loc('VALUE') if 'VALUE' in df.columns else 0)
        
        # Filtres optionnels
        st.subheader("Filtre (facultatif)")
        
        domain_options = ['-- Aucun --'] + list(df.columns)
        col_domain = st.selectbox("Colonne de domaine", options=domain_options)
        
        # Si un domaine est sélectionné
        if col_domain != '-- Aucun --':
            domain_filter_type = st.selectbox("Type de filtre", options=["=", "!=", "IN", "NOT IN"])
            
            if domain_filter_type in ["=", "!="]:
                domain_filter_value = st.text_input("Valeur")
            else:
                domain_values = df[col_domain].unique()
                domain_filter_value = st.multiselect("Valeurs", options=domain_values)
        
        # Enveloppe
        st.subheader("Enveloppe (facultatif)")
        
        use_envelope = st.checkbox("Définir une enveloppe manuelle", value=False)
        
        envelope_bounds = None
        
        if use_envelope:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("Minimum")
                min_x = st.number_input("Min X", value=float(df[col_x].min()), format="%.2f")
                min_y = st.number_input("Min Y", value=float(df[col_y].min()), format="%.2f")
                min_z = st.number_input("Min Z", value=float(df[col_z].min()), format="%.2f")
            
            with col2:
                st.markdown("Maximum")
                max_x = st.number_input("Max X", value=float(df[col_x].max()), format="%.2f")
                max_y = st.number_input("Max Y", value=float(df[col_y].max()), format="%.2f")
                max_z = st.number_input("Max Z", value=float(df[col_z].max()), format="%.2f")
            
            envelope_bounds = {
                'min': {'x': min_x, 'y': min_y, 'z': min_z},
                'max': {'x': max_x, 'y': max_y, 'z': max_z}
            }
    
    # Paramètres de krigeage
    st.header("Paramètres de krigeage")
    
    variogram_model = st.selectbox(
        "Modèle de variogramme",
        options=["spherical", "exponential", "gaussian"],
        index=0
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        nugget = st.number_input("Effet pépite", min_value=0.0, value=0.0, step=0.01)
        sill = st.number_input("Palier", min_value=0.01, value=1.0, step=0.01)
    
    with col2:
        range_value = st.number_input("Portée", min_value=1, value=50, step=1)
    
    st.subheader("Anisotropie (ratio des portées)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        anisotropy_x = st.number_input("X", min_value=0.1, value=1.0, step=0.1)
    
    with col2:
        anisotropy_y = st.number_input("Y", min_value=0.1, value=1.0, step=0.1)
    
    with col3:
        anisotropy_z = st.number_input("Z", min_value=0.1, value=0.5, step=0.1)
    
    # Paramètres du modèle de blocs
    st.header("Paramètres du modèle")
    
    st.subheader("Taille des blocs (m)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        block_size_x = st.number_input("X", min_value=1, value=5, step=1)
    
    with col2:
        block_size_y = st.number_input("Y", min_value=1, value=5, step=1)
    
    with col3:
        block_size_z = st.number_input("Z", min_value=1, value=5, step=1)
    
    density = st.number_input("Densité (t/m³)", min_value=0.1, value=2.7, step=0.1)
    
    st.subheader("Rayon de recherche (m)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_radius_x = st.number_input("X ", min_value=1, value=50, step=1)
    
    with col2:
        search_radius_y = st.number_input("Y ", min_value=1, value=50, step=1)
    
    with col3:
        search_radius_z = st.number_input("Z ", min_value=1, value=25, step=1)
    
    min_samples = st.number_input("Nombre min d'échantillons", min_value=1, value=3, step=1)
    max_samples = st.number_input("Nombre max d'échantillons", min_value=1, value=16, step=1)

# Traitement des données
if uploaded_file:
    # Préparation des composites avec mappage des colonnes
    composites_data = []
    
    # Appliquer le filtre de domaine si nécessaire
    filtered_df = df.copy()
    
    if col_domain != '-- Aucun --' and 'domain_filter_value' in locals():
        if domain_filter_type == "=":
            filtered_df = filtered_df[filtered_df[col_domain] == domain_filter_value]
        elif domain_filter_type == "!=":
            filtered_df = filtered_df[filtered_df[col_domain] != domain_filter_value]
        elif domain_filter_type == "IN":
            filtered_df = filtered_df[filtered_df[col_domain].isin(domain_filter_value)]
        elif domain_filter_type == "NOT IN":
            filtered_df = filtered_df[~filtered_df[col_domain].isin(domain_filter_value)]
    
    # Créer la liste des composites
    for _, row in filtered_df.iterrows():
        if pd.notnull(row[col_x]) and pd.notnull(row[col_y]) and pd.notnull(row[col_z]) and pd.notnull(row[col_value]):
            composites_data.append({
                'X': float(row[col_x]),
                'Y': float(row[col_y]),
                'Z': float(row[col_z]),
                'VALUE': float(row[col_value]),
                'DOMAIN': row[col_domain] if col_domain != '-- Aucun --' else None
            })
    
    # Afficher les statistiques des composites
    composite_values = [comp['VALUE'] for comp in composites_data]
    composite_stats = calculate_stats(composite_values)
    
    # Onglets principaux
    tabs = st.tabs(["Modèle 3D", "Statistiques", "Tonnage-Teneur", "Erreurs"])
    
    with tabs[0]:  # Modèle 3D
        st.subheader("Modèle de blocs 3D")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            create_model_button = st.button("Créer le modèle de blocs", type="primary")
            
            if "empty_blocks" in st.session_state and st.session_state.empty_blocks:
                estimate_button = st.button("Estimer par krigeage", type="primary")
            
            # Options d'affichage
            st.subheader("Options d'affichage")
            show_composites = st.checkbox("Afficher les composites", value=True)
            show_blocks = st.checkbox("Afficher les blocs", value=True)
            show_envelope = st.checkbox("Afficher l'enveloppe", value=True if envelope_bounds else False)
        
        with col1:
            if create_model_button:
                # Créer le modèle de blocs vide
                block_sizes = {'x': block_size_x, 'y': block_size_y, 'z': block_size_z}
                empty_blocks, model_bounds = create_block_model(composites_data, block_sizes, envelope_bounds, use_envelope)
                
                st.session_state.empty_blocks = empty_blocks
                st.session_state.model_bounds = model_bounds
                
                st.success(f"Modèle créé avec {len(empty_blocks)} blocs")
                
                # Afficher le modèle 3D
                fig = plot_3d_model(
                    [],
                    composites_data if show_composites else [],
                    model_bounds if show_envelope else None
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif "empty_blocks" in st.session_state and estimate_button:
                # Paramètres pour le krigeage
                kriging_params = {
                    'model_type': variogram_model,
                    'nugget': nugget,
                    'sill': sill,
                    'range': range_value,
                    'anisotropy': {'x': anisotropy_x, 'y': anisotropy_y, 'z': anisotropy_z}
                }
                
                search_params = {
                    'x': search_radius_x,
                    'y': search_radius_y,
                    'z': search_radius_z,
                    'min_samples': min_samples,
                    'max_samples': max_samples
                }
                
                # Estimer le modèle
                estimated_blocks, kriging_variances = estimate_block_model(
                    st.session_state.empty_blocks, 
                    composites_data, 
                    kriging_params, 
                    search_params
                )
                
                st.session_state.estimated_blocks = estimated_blocks
                st.session_state.kriging_variances = kriging_variances
                
                st.success(f"Estimation terminée, {len(estimated_blocks)} blocs estimés")
                
                # Afficher le modèle estimé
                fig = plot_3d_model(
                    estimated_blocks if show_blocks else [],
                    composites_data if show_composites else [],
                    st.session_state.model_bounds if show_envelope else None
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Section d'export
                st.subheader("Exporter")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Exporter en CSV"):
                        # Créer un DataFrame pour l'export
                        export_df = pd.DataFrame(estimated_blocks)
                        
                        # Renommer les colonnes pour correspondre au format d'origine
                        export_df = export_df.rename(columns={
                            'x': 'X', 'y': 'Y', 'z': 'Z', 'value': 'VALUE', 'variance': 'VARIANCE',
                            'size_x': 'SIZE_X', 'size_y': 'SIZE_Y', 'size_z': 'SIZE_Z'
                        })
                        
                        # Créer le lien de téléchargement
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Télécharger CSV",
                            data=csv,
                            file_name="modele_blocs_krigeage.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("Exporter image 3D"):
                        # Créer une image de haute qualité
                        fig = plot_3d_model(
                            estimated_blocks if show_blocks else [],
                            composites_data if show_composites else [],
                            st.session_state.model_bounds if show_envelope else None
                        )
                        fig.update_layout(width=1200, height=800)
                        
                        # Convertir en image PNG
                        img_bytes = fig.to_image(format="png", scale=2)
                        
                        # Créer le lien de téléchargement
                        st.download_button(
                            label="Télécharger PNG",
                            data=img_bytes,
                            file_name="modele_3d.png",
                            mime="image/png"
                        )
            
            elif "estimated_blocks" in st.session_state:
                # Afficher le modèle estimé déjà calculé
                fig = plot_3d_model(
                    st.session_state.estimated_blocks if show_blocks else [],
                    composites_data if show_composites else [],
                    st.session_state.model_bounds if show_envelope else None
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Section d'export
                st.subheader("Exporter")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Exporter en CSV"):
                        # Créer un DataFrame pour l'export
                        export_df = pd.DataFrame(st.session_state.estimated_blocks)
                        
                        # Renommer les colonnes pour correspondre au format d'origine
                        export_df = export_df.rename(columns={
                            'x': 'X', 'y': 'Y', 'z': 'Z', 'value': 'VALUE', 'variance': 'VARIANCE',
                            'size_x': 'SIZE_X', 'size_y': 'SIZE_Y', 'size_z': 'SIZE_Z'
                        })
                        
                        # Créer le lien de téléchargement
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Télécharger CSV",
                            data=csv,
                            file_name="modele_blocs_krigeage.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("Exporter image 3D"):
                        # Créer une image de haute qualité
                        fig = plot_3d_model(
                            st.session_state.estimated_blocks if show_blocks else [],
                            composites_data if show_composites else [],
                            st.session_state.model_bounds if show_envelope else None
                        )
                        fig.update_layout(width=1200, height=800)
                        
                        # Convertir en image PNG
                        img_bytes = fig.to_image(format="png", scale=2)
                        
                        # Créer le lien de téléchargement
                        st.download_button(
                            label="Télécharger PNG",
                            data=img_bytes,
                            file_name="modele_3d.png",
                            mime="image/png"
                        )
            
            else:
                # Afficher seulement les composites si aucun modèle n'est créé
                fig = plot_3d_model(
                    [],
                    composites_data if show_composites else [],
                    envelope_bounds if show_envelope and envelope_bounds else None
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # Statistiques
        st.subheader("Statistiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Statistiques des composites")
            
            stats_df = pd.DataFrame({
                'Paramètre': ['Nombre d\'échantillons', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'CV'],
                'Valeur': [
                    composite_stats['count'],
                    f"{composite_stats['min']:.3f}",
                    f"{composite_stats['max']:.3f}",
                    f"{composite_stats['mean']:.3f}",
                    f"{composite_stats['median']:.3f}",
                    f"{composite_stats['stddev']:.3f}",
                    f"{composite_stats['cv']:.3f}"
                ]
            })
            
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            st.markdown("### Histogramme des composites")
            fig = plot_histogram(composite_values, "Distribution des teneurs des composites", "darkblue")
            st.pyplot(fig)
        
        with col2:
            if "estimated_blocks" in st.session_state:
                block_values = [block['value'] for block in st.session_state.estimated_blocks]
                block_stats = calculate_stats(block_values)
                
                st.markdown("### Statistiques du modèle de blocs")
                
                stats_df = pd.DataFrame({
                    'Paramètre': ['Nombre de blocs', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'CV'],
                    'Valeur': [
                        block_stats['count'],
                        f"{block_stats['min']:.3f}",
                        f"{block_stats['max']:.3f}",
                        f"{block_stats['mean']:.3f}",
                        f"{block_stats['median']:.3f}",
                        f"{block_stats['stddev']:.3f}",
                        f"{block_stats['cv']:.3f}"
                    ]
                })
                
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                st.markdown("### Histogramme du modèle de blocs")
                fig = plot_histogram(block_values, "Distribution des teneurs du modèle de blocs", "teal")
                st.pyplot(fig)
                
                # Résumé des statistiques globales
                st.markdown("### Résumé global")
                
                block_volume = st.session_state.estimated_blocks[0]['size_x'] * st.session_state.estimated_blocks[0]['size_y'] * st.session_state.estimated_blocks[0]['size_z']
                total_volume = len(st.session_state.estimated_blocks) * block_volume
                total_tonnage = total_volume * density
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Nombre de blocs", f"{len(st.session_state.estimated_blocks)}")
                
                with col2:
                    st.metric("Teneur moyenne", f"{block_stats['mean']:.3f}")
                
                with col3:
                    st.metric("Volume total (m³)", f"{total_volume:,.0f}")
                
                with col4:
                    st.metric("Tonnage total (t)", f"{total_tonnage:,.0f}")
            else:
                st.info("Veuillez d'abord créer et estimer le modèle de blocs pour afficher les statistiques.")
    
    with tabs[2]:  # Tonnage-Teneur
        st.subheader("Analyse Tonnage-Teneur")
        
        if "estimated_blocks" in st.session_state:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                cutoff_method = st.selectbox(
                    "Méthode de coupure",
                    options=["above", "below", "between"],
                    format_func=lambda x: "Teneur ≥ Coupure" if x == "above" else "Teneur ≤ Coupure" if x == "below" else "Entre deux teneurs"
                )
            
            cutoff_value = None
            cutoff_min = None
            cutoff_max = None
            
            if cutoff_method == "between":
                with col2:
                    cutoff_min = st.number_input("Teneur min", min_value=0.0, value=0.5, step=0.1)
                
                with col3:
                    cutoff_max = st.number_input("Teneur max", min_value=cutoff_min, value=1.0, step=0.1)
            else:
                with col2:
                    cutoff_value = st.number_input("Teneur de coupure", min_value=0.0, value=0.5, step=0.1)
            
            with col4:
                if st.button("Calculer", type="primary"):
                    # Calculer les données tonnage-teneur
                    tonnage_data, plot_info = calculate_tonnage_grade(
                        st.session_state.estimated_blocks,
                        density,
                        cutoff_method,
                        cutoff_value,
                        cutoff_min,
                        cutoff_max
                    )
                    
                    st.session_state.tonnage_data = tonnage_data
                    st.session_state.plot_info = plot_info
            
            if "tonnage_data" in st.session_state:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique Tonnage-Teneur
                    fig = plot_tonnage_grade(st.session_state.tonnage_data, st.session_state.plot_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Graphique Métal contenu
                    fig = plot_metal_content(st.session_state.tonnage_data, st.session_state.plot_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des résultats
                st.subheader("Résultats détaillés")
                
                if st.session_state.plot_info['method'] == 'between':
                    # Pour la méthode between, afficher un seul résultat
                    result_df = pd.DataFrame({
                        'Coupure': [f"{cutoff_min:.2f} - {cutoff_max:.2f}"],
                        'Tonnage (t)': [st.session_state.tonnage_data['tonnages'][0]],
                        'Teneur moyenne': [st.session_state.tonnage_data['grades'][0]],
                        'Métal contenu': [st.session_state.tonnage_data['metals'][0]]
                    })
                else:
                    # Pour les méthodes above et below, afficher la courbe complète
                    result_df = pd.DataFrame({
                        'Coupure': st.session_state.tonnage_data['cutoffs'],
                        'Tonnage (t)': st.session_state.tonnage_data['tonnages'],
                        'Teneur moyenne': st.session_state.tonnage_data['grades'],
                        'Métal contenu': st.session_state.tonnage_data['metals']
                    })
                
                st.dataframe(result_df, hide_index=True, use_container_width=True)
                
                # Export des résultats
                st.subheader("Exporter les résultats")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export Excel
                    if st.button("Exporter en Excel"):
                        # Créer un buffer pour le fichier Excel
                        output = io.BytesIO()
                        
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            # Écrire les données tonnage-teneur
                            result_df.to_excel(writer, sheet_name='Tonnage-Teneur', index=False)
                            
                            # Ajouter une feuille pour les paramètres
                            param_df = pd.DataFrame({
                                'Paramètre': [
                                    'Méthode de coupure', 
                                    'Méthode d\'estimation',
                                    'Taille des blocs (m)',
                                    'Densité (t/m³)',
                                    'Modèle de variogramme',
                                    'Effet pépite',
                                    'Palier',
                                    'Portée (m)',
                                    'Date d\'exportation'
                                ],
                                'Valeur': [
                                    "Teneur ≥ Coupure" if cutoff_method == "above" else "Teneur ≤ Coupure" if cutoff_method == "below" else f"Entre {cutoff_min} et {cutoff_max}",
                                    'Krigeage ordinaire',
                                    f"{block_size_x} × {block_size_y} × {block_size_z}",
                                    density,
                                    variogram_model,
                                    nugget,
                                    sill,
                                    range_value,
                                    pd.Timestamp.now().strftime('%Y-%m-%d')
                                ]
                            })
                            param_df.to_excel(writer, sheet_name='Paramètres', index=False)
                        
                        # Télécharger le fichier
                        output.seek(0)
                        st.download_button(
                            label="Télécharger Excel",
                            data=output,
                            file_name="tonnage_teneur_krigeage.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                with col2:
                    # Export graphiques PNG
                    if st.button("Exporter graphiques PNG"):
                        # Créer un buffer ZIP pour les graphiques
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                            # Ajouter le graphique Tonnage-Teneur
                            fig = plot_tonnage_grade(st.session_state.tonnage_data, st.session_state.plot_info)
                            fig_bytes = fig.to_image(format="png", scale=2)
                            zip_file.writestr("tonnage_teneur.png", fig_bytes)
                            
                            # Ajouter le graphique Métal contenu
                            fig = plot_metal_content(st.session_state.tonnage_data, st.session_state.plot_info)
                            fig_bytes = fig.to_image(format="png", scale=2)
                            zip_file.writestr("metal_contenu.png", fig_bytes)
                        
                        # Télécharger le fichier ZIP
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Télécharger graphiques PNG",
                            data=zip_buffer,
                            file_name="graphiques_tonnage_teneur.zip",
                            mime="application/zip"
                        )
        else:
            st.info("Veuillez d'abord créer et estimer le modèle de blocs pour effectuer l'analyse tonnage-teneur.")
    
    with tabs[3]:  # Erreurs
        st.subheader("Analyse des erreurs et de la variance de krigeage")
        
        if "estimated_blocks" in st.session_state and "kriging_variances" in st.session_state:
            # Statistiques de variance
            variance_stats = calculate_stats(st.session_state.kriging_variances)
            
            # Métriques sommaires
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Variance de krigeage moyenne", f"{variance_stats['mean']:.3f}")
            
            with col2:
                st.metric("Écart-type de la variance", f"{variance_stats['stddev']:.3f}")
            
            with col3:
                st.metric("CV de la variance", f"{variance_stats['cv']:.3f}")
            
            # Histogramme de la variance de krigeage
            st.subheader("Distribution de la variance de krigeage")
            fig = plot_kriging_variance(st.session_state.kriging_variances)
            st.pyplot(fig)
            
            # Carte de la variance
            st.subheader("Carte 3D de la variance de krigeage")
            
            # Créer un tracé 3D avec la variance comme couleur
            variances_3d = go.Scatter3d(
                x=[block['x'] for block in st.session_state.estimated_blocks],
                y=[block['y'] for block in st.session_state.estimated_blocks],
                z=[block['z'] for block in st.session_state.estimated_blocks],
                mode='markers',
                marker=dict(
                    size=5,
                    color=[block['variance'] for block in st.session_state.estimated_blocks],
                    colorscale='Viridis',
                    colorbar=dict(title="Variance"),
                    opacity=0.8,
                    symbol='square'
                ),
                text=[f"X: {block['x']:.1f}, Y: {block['y']:.1f}, Z: {block['z']:.1f}<br>Variance: {block['variance']:.3f}" for block in st.session_state.estimated_blocks],
                name='Variance de krigeage'
            )
            
            layout = go.Layout(
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectratio=dict(x=1, y=1, z=1)
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            
            fig = go.Figure(data=[variances_3d], layout=layout)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            st.subheader("Exporter l'analyse d'erreur")
            
            if st.button("Exporter graphique de variance"):
                # Créer un graphique haute résolution de la variance
                fig = plot_kriging_variance(st.session_state.kriging_variances)
                fig.set_size_inches(12, 8)
                fig.set_dpi(300)
                
                # Convertir en bytes
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                # Télécharger l'image
                st.download_button(
                    label="Télécharger PNG",
                    data=buf,
                    file_name="variance_krigeage.png",
                    mime="image/png"
                )
        else:
            st.info("Veuillez d'abord créer et estimer le modèle de blocs pour analyser les erreurs d'estimation.")

else:
    # Affichage par défaut lorsqu'aucun fichier n'est chargé
    st.info("Veuillez charger un fichier CSV de composites dans le panneau latéral pour commencer.")
    
    st.markdown("""
    ## Guide d'utilisation
    
    1. **Chargez un fichier CSV** contenant vos données de composites dans le panneau latéral
    2. **Mappez les colonnes** pour identifier les coordonnées X, Y, Z et les valeurs de teneur
    3. **Définissez les paramètres du variogramme** selon votre connaissance du gisement
    4. **Créez le modèle de blocs** en définissant la taille des blocs
    5. **Estimez par krigeage** pour obtenir les teneurs des blocs
    6. **Analysez les résultats** à travers les différents onglets
    
    ### Format du fichier CSV
    
    Le fichier CSV doit contenir au minimum les colonnes suivantes:
    - Coordonnées X, Y, Z des échantillons
    - Valeurs de teneur
    - Optionnellement, une colonne de domaine pour le filtrage
    
    ### À propos du krigeage ordinaire
    
    Le krigeage ordinaire est une méthode d'estimation géostatistique optimale qui tient compte de:
    - La **configuration spatiale** des échantillons
    - La **corrélation spatiale** des teneurs via le variogramme
    - La **minimisation** de la variance d'estimation
    
    Cette application permet d'effectuer une estimation par krigeage ordinaire et d'analyser les résultats de manière interactive.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 MineEstim - Développé par Didier Ouedraogo, P.Geo")