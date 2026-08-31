//! Serves the pre-built UI (from `ui/dist`) as HTTP static files.
//!
//! Enabled only with the `embed-ui` feature. The assets are inlined at
//! compile time via `rust-embed`, so the binary is fully self-contained and
//! needs no filesystem access at runtime.
//!
//! Build the UI first (`trunk build --release` in `ui/`), then build with
//! `cargo build --release -p hl2-api --features embed-ui`.

use rocket::http::ContentType;
use rocket::response::{self, Responder};
use rust_embed::RustEmbed;
use std::io::Cursor;

#[derive(RustEmbed)]
#[folder = "../ui/dist/"]
struct Assets;

fn content_type(path: &str) -> ContentType {
    let ext = path.rsplit_once('.').map(|e| e.1).unwrap_or("");
    match ext {
        "html" | "htm" => ContentType::new("text", "html"),
        "js" | "mjs" => ContentType::new("text", "javascript"),
        "css" => ContentType::new("text", "css"),
        "wasm" => ContentType::new("application", "wasm"),
        "json" => ContentType::JSON,
        "svg" => ContentType::new("image", "svg+xml"),
        "png" => ContentType::new("image", "png"),
        "jpg" | "jpeg" => ContentType::new("image", "jpeg"),
        "gif" => ContentType::new("image", "gif"),
        "ico" => ContentType::new("image", "x-icon"),
        "woff" => ContentType::new("font", "woff"),
        "woff2" => ContentType::new("font", "woff2"),
        "ttf" => ContentType::new("font", "ttf"),
        "txt" => ContentType::Plain,
        _ => ContentType::Binary,
    }
}

/// An embedded asset (or the index page) with the correct Content-Type.
/// Serves with a fixed-size body — no filesystem involvement at runtime.
pub(crate) struct Asset {
    mime: ContentType,
    data: Vec<u8>,
}

impl<'r> Responder<'r, 'static> for Asset {
    fn respond_to(self, _: &rocket::Request<'_>) -> response::Result<'static> {
        rocket::Response::build()
            .header(self.mime)
            .sized_body(self.data.len(), Cursor::new(self.data))
            .ok()
    }
}

/// The UI index page.
#[rocket::get("/")]
pub fn index_page() -> Asset {
    let file = Assets::get("index.html")
        .expect("ui/dist/index.html not embedded (did you run `trunk build --release`?)");
    Asset {
        mime: ContentType::new("text", "html"),
        data: file.data.into_owned(),
    }
}

/// Any other asset under the UI (WASM, JS, CSS, images, fonts, …).
#[rocket::get("/<path..>")]
pub fn asset(path: std::path::PathBuf) -> Result<Asset, rocket::http::Status> {
    let rel = path.to_string_lossy().into_owned();
    match Assets::get(&rel) {
        Some(file) => Ok(Asset {
            mime: content_type(&rel),
            data: file.data.into_owned(),
        }),
        None => Err(rocket::http::Status::NotFound),
    }
}
