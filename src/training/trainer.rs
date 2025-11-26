//! Trainer pour l'entraînement du modèle CNN

use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    module::Module,
};

use crate::{
    config::model_config::ModelConfig,
    data::dataset::{MalariaBatcher, MalariaDataset},
    model::malaria_cnn::MalariaCNN,
};

/// Entraîneur principal pour le modèle de détection du paludisme
pub struct MalariaTrainer<B: AutodiffBackend> {
    config: ModelConfig,
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> MalariaTrainer<B> {
    /// Crée un nouvel entraîneur
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            _backend: std::marker::PhantomData,
        }
    }

    /// Exécute l'entraînement complet du modèle
    pub async fn run(&self) -> anyhow::Result<()> {
        println!("🎯 Démarrage de l'entraînement avec données réelles...");

        // Création du modèle avec type explicite
        let device = B::Device::default();
        let model: MalariaCNN<B> = MalariaCNN::new(&device, &self.config);
        println!("✅ Modèle CNN créé avec succès");

        // Chargement des données réelles
        println!("📁 Chargement du dataset depuis: data/");
        let full_dataset = MalariaDataset::from_directory("data")?;
        
        // Mélanger et splitter le dataset
        let (train_dataset, valid_dataset) = full_dataset.split(0.8);
        
        // Création des batchers
        let batcher_train = MalariaBatcher::new(
            self.config.image_height,
            self.config.image_width,
        );
        
        let batcher_valid = MalariaBatcher::new(
            self.config.image_height, 
            self.config.image_width,
        );

        // Création des data loaders
        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(self.config.batch_size)
            .shuffle(42) // Seed pour la reproductibilité
            .num_workers(4)
            .build(train_dataset);

        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(self.config.batch_size)
            .num_workers(4)
            .build(valid_dataset);

        // Configuration de l'optimiseur
        let optim = AdamConfig::new();

        println!("📊 Configuration de l'apprentissage...");
        println!("   - Époques: {}", self.config.num_epochs);
        println!("   - Batch size: {}", self.config.batch_size);
        println!("   - Taux d'apprentissage: {}", self.config.learning_rate);
        println!("   - Dropout: {}", self.config.dropout_rate);
        println!("   - Device: {:?}", device);

        println!("🚀 Lancement de l'entraînement réel...");
        
        // Construction du learner avec entraînement réel
        let learner = LearnerBuilder::new("./malaria-model")
            .metric_train(AccuracyMetric::new())
            .metric_valid(AccuracyMetric::new())
            .metric_train(LossMetric::new())
            .metric_valid(LossMetric::new())
            .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
            .num_epochs(self.config.num_epochs)
            .build(model, optim.init(), self.config.learning_rate);

        // Démarrage de l'entraînement réel
        let model_trained = learner.fit(dataloader_train, dataloader_valid);

        println!("💾 Sauvegarde du modèle entraîné...");
        
        // Le modèle retourné est déjà sur InnerBackend et peut être sauvegardé directement
        // Pas besoin d'appeler .valid() car MalariaCNN implémente Record via le derive Module
        let model_to_save = model_trained.model;
        
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model_to_save.into_record(), "./malaria-model-final".into())?;

        println!("✅ Entraînement terminé avec succès!");
        println!("📁 Modèle sauvegardé dans: ./malaria-model-final");
        
        Ok(())
    }
}