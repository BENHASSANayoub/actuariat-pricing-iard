#!/usr/bin/env python3
"""
================================================================================
PROJET ACTUARIAT PRICING IARD
================================================================================
Tarification Assurance Multirisque Habitation (MRH)
- Analyse de sinistralité
- Modélisation GLM (Poisson & Gamma)
- Machine Learning (Random Forest, XGBoost)
- Construction du zonier
- Analyse d'impact

Auteur: Ayoub BENHASSAN
MBA Finance Quantitative - ESLSCA Business School
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURATION
# ================================================================================

# Chemin vers le fichier de données
DATA_PATH = "Data.xlsx"  # Modifier si nécessaire

# ================================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ================================================================================

def charger_donnees(path):
    """Charge les données depuis le fichier Excel."""
    print("=" * 70)
    print(" CHARGEMENT DES DONNÉES")
    print("=" * 70)
    
    # Charger la feuille principale
    df = pd.read_excel(path, sheet_name='Sinistres_MRH')
    
    print(f"✓ Données chargées: {len(df):,} observations")
    print(f"✓ Colonnes: {list(df.columns)}")
    
    return df


def preparer_donnees(df):
    """Prépare les données pour la modélisation."""
    print("\n" + "=" * 70)
    print(" PRÉPARATION DES DONNÉES")
    print("=" * 70)
    
    df = df.copy()
    
    # 1. Calculer le log de l'exposition (pour l'offset du GLM)
    df['log_exposition'] = np.log(df['exposition'])
    
    # 2. Calculer les variables dérivées
    df['frequence'] = df['nb_sinistres'] / df['exposition']
    df['prime_pure'] = df['cout_sinistre'] / df['exposition']
    
    # 3. Calculer le coût moyen par sinistre (sévérité)
    df['cout_moyen'] = np.where(
        df['nb_sinistres'] > 0,
        df['cout_sinistre'] / df['nb_sinistres'],
        0
    )
    
    # 4. Traiter les outliers (winsorisation au percentile 99)
    if df['cout_sinistre'].max() > 0:
        upper_99 = df.loc[df['cout_sinistre'] > 0, 'cout_sinistre'].quantile(0.99)
        df['cout_sinistre_cap'] = df['cout_sinistre'].clip(upper=upper_99)
        print(f"✓ Winsorisation sévérité: plafond à {upper_99:,.0f}€")
    
    print(f"✓ Variables créées: log_exposition, frequence, prime_pure, cout_moyen")
    print(f"✓ Données prêtes: {len(df):,} observations")
    
    return df


# ================================================================================
# 2. ANALYSE EXPLORATOIRE
# ================================================================================

def analyse_exploratoire(df):
    """Analyse exploratoire des données."""
    print("\n" + "=" * 70)
    print(" ANALYSE EXPLORATOIRE")
    print("=" * 70)
    
    # Statistiques globales
    print("\n📊 STATISTIQUES GLOBALES:")
    print("-" * 50)
    
    exposition_totale = df['exposition'].sum()
    nb_sinistres_total = df['nb_sinistres'].sum()
    cout_total = df['cout_sinistre'].sum()
    
    frequence_globale = nb_sinistres_total / exposition_totale
    severite_globale = cout_total / nb_sinistres_total if nb_sinistres_total > 0 else 0
    prime_pure_globale = cout_total / exposition_totale
    
    print(f"  Exposition totale: {exposition_totale:,.1f} années")
    print(f"  Nombre de sinistres: {nb_sinistres_total:,}")
    print(f"  Coût total: {cout_total:,.0f}€")
    print(f"  Fréquence: {frequence_globale:.2%}")
    print(f"  Sévérité: {severite_globale:,.0f}€")
    print(f"  Prime Pure: {prime_pure_globale:,.0f}€")
    
    # Statistiques par zone géographique
    print("\n📊 STATISTIQUES PAR ZONE GÉOGRAPHIQUE:")
    print("-" * 50)
    
    stats_zone = df.groupby('zone_geo').agg({
        'nb_sinistres': 'sum',
        'cout_sinistre': 'sum',
        'exposition': 'sum'
    })
    stats_zone['frequence'] = stats_zone['nb_sinistres'] / stats_zone['exposition']
    stats_zone['severite'] = stats_zone['cout_sinistre'] / stats_zone['nb_sinistres']
    stats_zone['prime_pure'] = stats_zone['cout_sinistre'] / stats_zone['exposition']
    stats_zone['relativite'] = stats_zone['prime_pure'] / prime_pure_globale
    
    print(stats_zone[['frequence', 'severite', 'prime_pure', 'relativite']].round(2).to_string())
    
    # Statistiques par type de logement
    print("\n📊 STATISTIQUES PAR TYPE DE LOGEMENT:")
    print("-" * 50)
    
    stats_logement = df.groupby('type_logement').agg({
        'nb_sinistres': 'sum',
        'cout_sinistre': 'sum',
        'exposition': 'sum'
    })
    stats_logement['frequence'] = stats_logement['nb_sinistres'] / stats_logement['exposition']
    stats_logement['severite'] = stats_logement['cout_sinistre'] / stats_logement['nb_sinistres']
    stats_logement['prime_pure'] = stats_logement['cout_sinistre'] / stats_logement['exposition']
    stats_logement['relativite'] = stats_logement['prime_pure'] / prime_pure_globale
    
    print(stats_logement[['frequence', 'severite', 'prime_pure', 'relativite']].round(2).to_string())
    
    return stats_zone, stats_logement


# ================================================================================
# 3. MODÉLISATION GLM
# ================================================================================

def glm_frequence(df, features):
    """
    Modèle GLM Poisson pour la fréquence.
    
    Formule: ln(E[N]) = ln(Exposition) + β₀ + β₁X₁ + ... + βₚXₚ
    """
    print("\n" + "=" * 70)
    print(" MODÈLE GLM FRÉQUENCE (Poisson)")
    print("=" * 70)
    
    # Construire la formule
    formula_terms = []
    for f in features:
        if df[f].dtype == 'object':
            formula_terms.append(f'C({f})')
        else:
            formula_terms.append(f)
    
    formula = 'nb_sinistres ~ ' + ' + '.join(formula_terms)
    print(f"Formule: {formula}")
    
    # Ajuster le modèle
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df['log_exposition']
    )
    result = model.fit()
    
    # Afficher le résumé
    print("\n📊 RÉSUMÉ DU MODÈLE:")
    print("-" * 50)
    print(f"AIC: {result.aic:.1f}")
    print(f"Deviance: {result.deviance:.1f}")
    print(f"Log-Likelihood: {result.llf:.1f}")
    
    # Calculer les relativités
    print("\n📊 RELATIVITÉS (exp(β)):")
    print("-" * 50)
    relativites = np.exp(result.params)
    for var, rel in relativites.items():
        if var != 'Intercept':
            impact = (rel - 1) * 100
            signe = "+" if impact > 0 else ""
            print(f"  {var}: {rel:.3f} ({signe}{impact:.1f}%)")
    
    # Diagnostic de surdispersion
    print("\n📊 DIAGNOSTIC SURDISPERSION:")
    print("-" * 50)
    residuals_pearson = result.resid_pearson
    phi = np.sum(residuals_pearson**2) / result.df_resid
    print(f"  Paramètre de dispersion φ = {phi:.3f}")
    
    if phi > 1.5:
        print("  ⚠ Surdispersion significative - Envisager Quasi-Poisson ou Binomiale Négative")
    else:
        print("  ✓ Pas de surdispersion majeure - Modèle Poisson adapté")
    
    return result, relativites


def glm_severite(df, features):
    """
    Modèle GLM Gamma pour la sévérité.
    
    IMPORTANT: Ajusté uniquement sur les sinistres (coût > 0)
    """
    print("\n" + "=" * 70)
    print(" MODÈLE GLM SÉVÉRITÉ (Gamma)")
    print("=" * 70)
    
    # Filtrer uniquement les sinistres
    df_sin = df[df['cout_sinistre'] > 0].copy()
    print(f"Observations avec sinistre: {len(df_sin):,}")
    
    # Construire la formule
    formula_terms = []
    for f in features:
        if df_sin[f].dtype == 'object':
            formula_terms.append(f'C({f})')
        else:
            formula_terms.append(f)
    
    formula = 'cout_sinistre ~ ' + ' + '.join(formula_terms)
    print(f"Formule: {formula}")
    
    # Ajuster le modèle
    model = smf.glm(
        formula=formula,
        data=df_sin,
        family=sm.families.Gamma(link=sm.families.links.Log())
    )
    result = model.fit()
    
    # Afficher le résumé
    print("\n📊 RÉSUMÉ DU MODÈLE:")
    print("-" * 50)
    print(f"AIC: {result.aic:.1f}")
    print(f"Deviance: {result.deviance:.1f}")
    
    # Calculer les relativités
    print("\n📊 RELATIVITÉS (exp(β)):")
    print("-" * 50)
    relativites = np.exp(result.params)
    for var, rel in relativites.items():
        if var != 'Intercept':
            impact = (rel - 1) * 100
            signe = "+" if impact > 0 else ""
            print(f"  {var}: {rel:.3f} ({signe}{impact:.1f}%)")
    
    return result, relativites


# ================================================================================
# 4. MACHINE LEARNING
# ================================================================================

def benchmark_ml(df, features, target='prime_pure'):
    """Benchmark avec Random Forest et XGBoost."""
    print("\n" + "=" * 70)
    print(" BENCHMARK MACHINE LEARNING")
    print("=" * 70)
    
    # Préparer les données
    # Encoder les variables catégorielles
    df_ml = df.copy()
    cat_cols = [f for f in features if df_ml[f].dtype == 'object']
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    
    # Sélectionner les features encodées
    feature_cols = [c for c in df_ml.columns if any(f in c for f in features)]
    
    X = df_ml[feature_cols]
    y = df_ml[target]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    results = []
    
    # 1. Random Forest
    print("\n📊 RANDOM FOREST:")
    print("-" * 50)
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"  RMSE: {rf_rmse:.2f}")
    print(f"  MAE: {rf_mae:.2f}")
    print(f"  R²: {rf_r2:.4f}")
    
    results.append({'Modèle': 'Random Forest', 'RMSE': rf_rmse, 'MAE': rf_mae, 'R²': rf_r2})
    
    # Importance des variables
    print("\n  Top 5 variables importantes:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance.head(5).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.3f}")
    
    # 2. XGBoost (si disponible)
    try:
        import xgboost as xgb
        
        print("\n📊 XGBOOST:")
        print("-" * 50)
        
        # Convertir les booléens en int et nettoyer les noms de colonnes pour XGBoost
        X_train_xgb = X_train.astype(float).copy()
        X_test_xgb = X_test.astype(float).copy()
        
        # Renommer les colonnes pour éviter les caractères spéciaux
        X_train_xgb.columns = [c.replace('[', '_').replace(']', '_').replace('<', 'lt').replace('>', 'gt') for c in X_train_xgb.columns]
        X_test_xgb.columns = [c.replace('[', '_').replace(']', '_').replace('<', 'lt').replace('>', 'gt') for c in X_test_xgb.columns]
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train_xgb, y_train)
        xgb_pred = xgb_model.predict(X_test_xgb)
        
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        print(f"  RMSE: {xgb_rmse:.2f}")
        print(f"  MAE: {xgb_mae:.2f}")
        print(f"  R²: {xgb_r2:.4f}")
        
        results.append({'Modèle': 'XGBoost', 'RMSE': xgb_rmse, 'MAE': xgb_mae, 'R²': xgb_r2})
        
    except ImportError:
        print("\n⚠ XGBoost non installé. Installer avec: pip install xgboost")
    
    # Comparaison
    print("\n📊 COMPARAISON DES MODÈLES:")
    print("-" * 50)
    comparison = pd.DataFrame(results)
    print(comparison.to_string(index=False))
    
    return comparison, rf


# ================================================================================
# 5. CONSTRUCTION DU ZONIER
# ================================================================================

def construire_zonier(df, n_zones=5):
    """Construit un zonier tarifaire basé sur la prime pure par zone."""
    print("\n" + "=" * 70)
    print(" CONSTRUCTION DU ZONIER")
    print("=" * 70)
    
    # Calculer la prime pure par zone géographique
    stats_zone = df.groupby('zone_geo').agg({
        'nb_sinistres': 'sum',
        'cout_sinistre': 'sum',
        'exposition': 'sum'
    })
    stats_zone['prime_pure'] = stats_zone['cout_sinistre'] / stats_zone['exposition']
    
    # Prime pure moyenne (référence)
    prime_pure_moyenne = stats_zone['prime_pure'].mean()
    stats_zone['relativite'] = stats_zone['prime_pure'] / prime_pure_moyenne
    
    print("\n📊 ZONIER PAR ZONE GÉOGRAPHIQUE:")
    print("-" * 50)
    print(stats_zone[['prime_pure', 'relativite']].round(2).sort_values('relativite'))
    
    # Créer le zonier final
    zonier = stats_zone[['prime_pure', 'relativite']].copy()
    zonier['zone_tarifaire'] = pd.qcut(
        zonier['relativite'], 
        q=min(n_zones, len(zonier)),
        labels=[f'Zone_{i+1}' for i in range(min(n_zones, len(zonier)))]
    )
    
    print("\n📊 ZONES TARIFAIRES:")
    print("-" * 50)
    print(zonier.to_string())
    
    return zonier


# ================================================================================
# 6. ANALYSE D'IMPACT
# ================================================================================

def analyse_impact(df, zonier):
    """Simule l'impact d'une révision tarifaire."""
    print("\n" + "=" * 70)
    print(" ANALYSE D'IMPACT")
    print("=" * 70)
    
    # Appliquer les relativités
    df_impact = df.merge(
        zonier[['relativite']],
        left_on='zone_geo',
        right_index=True,
        how='left'
    )
    
    # Simuler les primes
    prime_base = df_impact['prime_pure'].mean() * 1.3  # Coefficient de chargement
    df_impact['prime_actuelle'] = prime_base
    df_impact['prime_nouvelle'] = prime_base * df_impact['relativite']
    df_impact['variation'] = df_impact['prime_nouvelle'] - df_impact['prime_actuelle']
    df_impact['variation_pct'] = (df_impact['variation'] / df_impact['prime_actuelle']) * 100
    
    # Répartition gagnants/perdants
    print("\n📊 RÉPARTITION GAGNANTS/PERDANTS:")
    print("-" * 50)
    
    df_impact['categorie'] = pd.cut(
        df_impact['variation_pct'],
        bins=[-np.inf, -10, -2, 2, 10, np.inf],
        labels=['Forte baisse (>10%)', 'Baisse (2-10%)', 'Stable (±2%)', 
                'Hausse (2-10%)', 'Forte hausse (>10%)']
    )
    
    repartition = df_impact['categorie'].value_counts().sort_index()
    for cat, count in repartition.items():
        pct = count / len(df_impact) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    # Impact global
    print("\n📊 IMPACT GLOBAL:")
    print("-" * 50)
    print(f"  Prime moyenne actuelle: {df_impact['prime_actuelle'].mean():.2f}€")
    print(f"  Prime moyenne nouvelle: {df_impact['prime_nouvelle'].mean():.2f}€")
    print(f"  Variation moyenne: {df_impact['variation_pct'].mean():.2f}%")
    
    return df_impact


# ================================================================================
# FONCTION PRINCIPALE
# ================================================================================

def main():
    """Pipeline principal du projet."""
    print("\n")
    print("=" * 70)
    print(" PROJET ACTUARIAT PRICING IARD")
    print(" Ayoub BENHASSAN - MBA Finance Quantitative")
    print("=" * 70)
    
    # 1. Charger les données
    try:
        df = charger_donnees(DATA_PATH)
    except FileNotFoundError:
        print(f"\n❌ Fichier non trouvé: {DATA_PATH}")
        print("   Veuillez modifier la variable DATA_PATH avec le bon chemin.")
        return
    
    # 2. Préparer les données
    df = preparer_donnees(df)
    
    # 3. Analyse exploratoire
    stats_zone, stats_logement = analyse_exploratoire(df)
    
    # 4. Définir les features
    features = ['type_logement', 'zone_geo', 'age_groupe', 'surface_groupe', 'etage']
    
    # Vérifier que les features existent
    features = [f for f in features if f in df.columns]
    print(f"\n✓ Features utilisées: {features}")
    
    # 5. Modélisation GLM
    if len(features) > 0:
        # GLM Fréquence
        result_freq, rel_freq = glm_frequence(df, features)
        
        # GLM Sévérité
        result_sev, rel_sev = glm_severite(df, features)
    
    # 6. Benchmark ML
    if len(features) > 0:
        comparison_ml, rf_model = benchmark_ml(df, features)
    
    # 7. Zonier
    zonier = construire_zonier(df)
    
    # 8. Analyse d'impact
    df_impact = analyse_impact(df, zonier)
    
    # Résumé final
    print("\n" + "=" * 70)
    print(" PROJET TERMINÉ AVEC SUCCÈS")
    print("=" * 70)
    print("""
    ✅ Données chargées et préparées
    ✅ Analyse exploratoire réalisée
    ✅ Modèle GLM Fréquence (Poisson) ajusté
    ✅ Modèle GLM Sévérité (Gamma) ajusté
    ✅ Benchmark Machine Learning effectué
    ✅ Zonier tarifaire construit
    ✅ Analyse d'impact simulée
    """)
    
    return df, zonier, df_impact


# ================================================================================
# EXÉCUTION
# ================================================================================

if __name__ == "__main__":
    # Exécuter le pipeline
    df, zonier, df_impact = main()
