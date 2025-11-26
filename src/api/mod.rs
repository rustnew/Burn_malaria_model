pub mod handlers;
pub mod routes;

use burn::backend::NdArray;
use crate::model::malaria_cnn::MalariaCNN;
use crate::config::model_config::ModelConfig;

pub type Backend = NdArray<f32>;

pub struct AppState {
    pub model: MalariaCNN<Backend>,
    pub config: ModelConfig,
}