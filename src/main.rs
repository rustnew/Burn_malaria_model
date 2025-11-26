//! Point d'entrée principal de l'application de détection du paludisme

mod model;
mod data;
mod training;
mod config;
mod api ; 

use anyhow::Result;
use burn::backend::{ Autodiff};
use burn_ndarray::NdArray;
use training::trainer::MalariaTrainer;
use config::model_config::ModelConfig;

use actix_web::{web, App, HttpServer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use api::{AppState, routes, Backend};
use config::model_config::ModelConfig;

/// Backend principal avec autodiff
type Backend = Autodiff<NdArray<f64>>;


#[tokio::main]
async fn main() -> Result<()> {
    // Configuration du modèle optimisée pour 13,000 images
    // Dans votre ModelConfig pour CPU
      let config = ModelConfig {
        image_width: 64,
        image_height: 64,
        image_channels: 3,
        conv1_filters: 16,
        conv2_filters: 32,
        conv3_filters: 64,
        fc1_units: 128,
        fc2_units: 64,
        num_classes: 2,
        dropout_rate: 0.5,
        learning_rate: 0.001,
        batch_size: 16,
        num_epochs: 5,
        ..Default::default()
    };

    println!("🚀 Initialisation de l'entraînement du CNN pour la détection du paludisme");
    println!("📊 Configuration: {:?}", config);
    println!("📁 Structure des données attendue:");
    println!("   data/");
    println!("   ├── Parasitized/    (13,000 images infectées)");
    println!("   └── Uninfected/     (13,000 images saines)");


    // Création et démarrage de l'entraînement
    let trainer: MalariaTrainer<Backend> = MalariaTrainer::new(config);
    
    match trainer.run().await {
        Ok(_) => println!("✅ Entraînement terminé avec succès!"),
        Err(e) => {
            eprintln!("❌ Erreur pendant l'entraînement: {}", e);
            eprintln!("💡 Vérifiez que:");
            eprintln!("   - Le dossier 'data/' existe");
            eprintln!("   - Les sous-dossiers 'Parasitized/' et 'Uninfected/' existent");
            eprintln!("   - Les fichiers images sont au format PNG, JPG ou JPEG");
        }
    }
    
    Ok(())
}