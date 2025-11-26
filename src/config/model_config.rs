//! Configuration du modèle CNN et des hyperparamètres d'entraînement

use serde::{Deserialize, Serialize};

/// Configuration complète du modèle CNN pour la détection du paludisme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Largeur des images d'entrée
    pub image_width: usize,
    /// Hauteur des images d'entrée
    pub image_height: usize,
    /// Nombre de canaux (3 pour RGB, 1 pour grayscale)
    pub image_channels: usize,
    /// Nombre de filtres pour la première couche convolutive
    pub conv1_filters: usize,
    /// Nombre de filtres pour la deuxième couche convolutive
    pub conv2_filters: usize,
    /// Nombre de filtres pour la troisième couche convolutive
    pub conv3_filters: usize,
    /// Unités pour la première couche fully-connected
    pub fc1_units: usize,
    /// Unités pour la deuxième couche fully-connected
    pub fc2_units: usize,
    /// Nombre de classes de sortie (2: paludisme/non-paludisme)
    pub num_classes: usize,
    /// Taux de dropout pour la régularisation
    pub dropout_rate: f64,
    /// Taux d'apprentissage pour l'optimiseur
    pub learning_rate: f64,
    /// Taille des batches d'entraînement
    pub batch_size: usize,
    /// Nombre d'époques d'entraînement
    pub num_epochs: usize,
    /// Chemin vers le dataset d'entraînement
    pub train_data_path: String,
    /// Chemin vers le dataset de validation
    pub val_data_path: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            image_width: 128,
            image_height: 128,
            image_channels: 3,
            conv1_filters: 32,
            conv2_filters: 64,
            conv3_filters: 128,
            fc1_units: 512,
            fc2_units: 128,
            num_classes: 2,
            dropout_rate: 0.5,
            learning_rate: 0.01,
            batch_size: 32,
            num_epochs: 50,
            train_data_path: "data/train".to_string(),
            val_data_path: "data/val".to_string(),
        }
    }
}