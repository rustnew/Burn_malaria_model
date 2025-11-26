use actix_multipart::Multipart;
use actix_web::{web, HttpResponse, Error};
use futures::{StreamExt, TryStreamExt};
use burn::tensor::Tensor;
use std::io::Write;
use super::AppState;

// Structure pour la réponse JSON
#[derive(serde::Serialize)]
pub struct PredictionResponse {
    pub prediction: String,
    pub confidence_parasitized: f32,
    pub confidence_uninfected: f32,
    pub confidence_percentage: f32,
}

// Handler pour la prédiction d'image uploadée
pub async fn predict_upload(
    mut payload: Multipart,
    data: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let mut image_data = Vec::new();
    
    // Traitement du multipart form data [citation:2]
    while let Ok(Some(mut field)) = payload.try_next().await {
        let content_type = field.content_disposition();
        
        if let Some(filename) = content_type.get_filename() {
            if filename.to_lowercase().ends_with(".png") || 
               filename.to_lowercase().ends_with(".jpg") || 
               filename.to_lowercase().ends_with(".jpeg") {
                
                // Collecter les chunks de données [citation:2]
                while let Some(chunk) = field.next().await {
                    let data = chunk?;
                    image_data.extend_from_slice(&data);
                }
                break;
            }
        } else {
            // Drainer le payload si ce n'est pas un fichier [citation:8]
            while let Ok(Some(_)) = field.try_next().await {}
        }
    }
    
    if image_data.is_empty() {
        return Ok(HttpResponse::BadRequest().json("Aucune image valide fournie"));
    }
    
    // Charger et prétraiter l'image
    let image = match image::load_from_memory(&image_data) {
        Ok(img) => img,
        Err(_) => return Ok(HttpResponse::BadRequest().json("Format d'image invalide")),
    };
    
    // Faire la prédiction
    match process_prediction(&image, &data.model, &data.config) {
        Ok(prediction) => Ok(HttpResponse::Ok().json(prediction)),
        Err(e) => {
            eprintln!("Erreur de prédiction: {}", e);
            Ok(HttpResponse::InternalServerError().json("Erreur lors de la prédiction"))
        }
    }
}

// Handler pour la santé de l'API
pub async fn health_check() -> HttpResponse {
    HttpResponse::Ok().json("API de détection du paludisme opérationnelle")
}

// Fonction de traitement de l'image et prédiction
fn process_prediction(
    image: &image::DynamicImage,
    model: &MalariaCNN<super::Backend>,
    config: &crate::config::model_config::ModelConfig,
) -> anyhow::Result<PredictionResponse> {
    
    let device = Default::default();
    
    // Redimensionner l'image
    let img = image.resize_exact(
        config.image_width as u32,
        config.image_height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    
    // Conversion en RGB et normalisation
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    let mut data = Vec::with_capacity((3 * width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x, y);
            data.push(pixel[0] as f32 / 255.0); // R
            data.push(pixel[1] as f32 / 255.0); // G  
            data.push(pixel[2] as f32 / 255.0); // B
        }
    }

    // Création du tenseur [citation:9]
    let total_elements = 3 * height as usize * width as usize;
    let tensor_1d = Tensor::<super::Backend, 1>::from_floats(&data[..total_elements.min(data.len())], &device);
    let input_tensor = tensor_1d.reshape([1, 3, height as usize, width as usize]);
    
    // Inférence
    let output = model.infer(input_tensor);
    let probabilities = burn::tensor::activation::softmax(output, 1);
    let probs = probabilities.into_data().convert::<f32>().value;
    
    let parasitized_prob = probs[1];
    let uninfected_prob = probs[0];
    
    let prediction = if parasitized_prob > uninfected_prob {
        "PARASITÉ".to_string()
    } else {
        "SAIN".to_string()
    };
    
    let confidence_percentage = if parasitized_prob > uninfected_prob {
        parasitized_prob * 100.0
    } else {
        uninfected_prob * 100.0
    };
    
    Ok(PredictionResponse {
        prediction,
        confidence_parasitized: parasitized_prob,
        confidence_uninfected: uninfected_prob,
        confidence_percentage,
    })
}