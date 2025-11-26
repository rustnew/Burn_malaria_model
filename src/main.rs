//! Point d'entrée principal de l'application de détection du paludisme

mod model;
mod data;
mod training;
mod config;
mod api ; 
//mod  inference;

use anyhow::Result;
use burn::backend::{ Autodiff};
use burn_ndarray::NdArray;
use training::trainer::MalariaTrainer;
use config::model_config::ModelConfig;


/// Backend principal avec autodiff
type Backend = Autodiff<NdArray<f64>>;


#[tokio::main]
async fn main() -> Result<()> {
    // Configuration du modèle optimisée pour 13,000 images
    // Dans votre ModelConfig pour CPU
    let config = ModelConfig {
        image_width: 64,       // Garder 64x64 pour CPU
        image_height: 64,
        conv1_filters: 16,     // Déjà réduit
        conv2_filters: 32,     // Déjà réduit  
        conv3_filters: 64,     // Déjà réduit
        fc1_units: 128,        // Déjà réduit
        fc2_units: 64,         // Déjà réduit
        batch_size: 16,        // Bon pour CPU
        num_epochs: 5,        // Réduire si trop long
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