use actix_web::web;
use super::handlers;

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .route("/predict", web::post().to(handlers::predict_upload))
            .route("/health", web::get().to(handlers::health_check))
    )
    .service(actix_files::Files::new("/", "./static").index_file("index.html"));
}