//! Port of the image-API cases of `tests/models/qwen_image_21/model_test.cpp`
//! with a fake runner and codec; the tokenizer, PNG round trip, Lanczos
//! resize and flow schedule cases test the model and stay with it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use gufo_core::text::{Error, Result};
use gufo_serve::http::HttpRequest;
use gufo_serve::image_api::{
    Image, ImageCodec, ImageResult, ImageService, handle_image_api_request,
};

/// `FAKEIMG`, width and height (u32 little-endian), then RGBA.
struct FakeCodec;

const MAGIC: &[u8] = b"FAKEIMG";

impl ImageCodec for FakeCodec {
    fn decode(&self, bytes: &[u8]) -> Result<Image> {
        let header = MAGIC.len() + 8;
        if bytes.len() < header || !bytes.starts_with(MAGIC) {
            return Err(Error::invalid("cannot decode PNG image"));
        }
        let word = |offset: usize| {
            let mut b = [0; 4];
            b.copy_from_slice(&bytes[offset..offset + 4]);
            u32::from_le_bytes(b)
        };
        let (width, height) = (word(MAGIC.len()), word(MAGIC.len() + 4));
        let rgba = bytes[header..].to_vec();
        if rgba.len() != width as usize * height as usize * 4 {
            return Err(Error::invalid("invalid PNG image data"));
        }
        Ok(Image {
            width: width as i32,
            height: height as i32,
            rgba,
        })
    }

    fn encode_png(&self, image: &Image) -> Result<Vec<u8>> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(image.width as u32).to_le_bytes());
        bytes.extend_from_slice(&(image.height as u32).to_le_bytes());
        bytes.extend_from_slice(&image.rgba);
        Ok(bytes)
    }
}

fn image() -> Image {
    Image {
        width: 2,
        height: 2,
        rgba: vec![
            255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 127, 63, 21, 255,
        ],
    }
}

#[test]
fn image_service_contracts() {
    let calls = Arc::new(AtomicU64::new(0));
    let counter = calls.clone();
    let service = Arc::new(
        ImageService::new(
            "image-test",
            Box::new(move |request, _| {
                let call = counter.fetch_add(1, Ordering::SeqCst) + 1;
                assert_eq!(request.prompt, "a red cube");
                assert!(request.width == 256 && request.height == 256);
                assert_eq!(request.seed, call + 40);
                Ok(ImageResult {
                    image: image(),
                    ..ImageResult::default()
                })
            }),
            Arc::new(FakeCodec),
        )
        .expect("service"),
    );
    let mut http = HttpRequest {
        method: "POST".into(),
        path: "/v1/images/generations".into(),
        headers: vec![("Content-Type".into(), "application/json".into())],
        body: br#"{"model":"image-test","prompt":"a red cube","size":"256x256","n":2,"seed":41}"#
            .to_vec(),
        ..HttpRequest::default()
    };
    let response = handle_image_api_request(&http, &service).expect("handler");
    assert!(response.status == 200 && calls.load(Ordering::SeqCst) == 2);
    let json = gufo_core::json::parse(&response.body).expect("JSON");
    assert_eq!(json.find("data").expect("data").len(), 2);
    for body in [
        r#"{"prompt":"#,
        r#"{"prompt":"a red cube","seed":-1}"#,
        r#"{"prompt":"a red cube","n":1.5}"#,
        r#"{"prompt":"a red cube","n":0}"#,
        r#"{"prompt":"a red cube","output_format":"webp"}"#,
        r#"{"prompt":"a red cube","stream":true}"#,
        r#"{"prompt":"a red cube","size":"256x256junk"}"#,
        r#"{"prompt":"a red cube","unknown":true}"#,
    ] {
        http.body = body.as_bytes().to_vec();
        assert_eq!(
            handle_image_api_request(&http, &service)
                .expect("handler")
                .status,
            400,
            "{body}"
        );
    }
    http.body = br#"{"prompt":"a red cube","model":"unknown"}"#.to_vec();
    assert_eq!(
        handle_image_api_request(&http, &service)
            .expect("handler")
            .status,
        404
    );

    http.path = "/v1/images/edits".into();
    http.headers = vec![(
        "Content-Type".into(),
        "multipart/form-data; boundary=\"boundary\"".into(),
    )];
    let png = FakeCodec.encode_png(&image()).expect("encode");
    let mut body =
        b"--boundary\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"x.png\"\r\n\
                     Content-Type: image/png\r\n\r\n"
            .to_vec();
    body.extend_from_slice(&png);
    for (name, value) in [
        ("model", "image-test"),
        ("prompt", "a red cube"),
        ("size", "256x256"),
        ("seed", "43"),
    ] {
        body.extend_from_slice(
            format!(
                "\r\n--boundary\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(b"\r\n--boundary--\r\n");
    http.body = body;
    assert!(
        handle_image_api_request(&http, &service)
            .expect("handler")
            .status
            == 200
            && calls.load(Ordering::SeqCst) == 3
    );
    http.body.truncate(http.body.len() - 8);
    assert_eq!(
        handle_image_api_request(&http, &service)
            .expect("handler")
            .status,
        400
    );
}

#[test]
fn methods_and_failures() {
    let service = Arc::new(
        ImageService::new(
            "image-test",
            Box::new(|_, _| Err(Error::runtime("device lost"))),
            Arc::new(FakeCodec),
        )
        .expect("service"),
    );
    let mut http = HttpRequest {
        method: "GET".into(),
        path: "/v1/images/generations".into(),
        ..HttpRequest::default()
    };
    let response = handle_image_api_request(&http, &service).expect("handler");
    assert!(
        response.status == 400
            && String::from_utf8_lossy(&response.body).contains("invalid_method")
    );
    http.method = "POST".into();
    http.headers = vec![("Content-Type".into(), "application/json".into())];
    http.body = br#"{"prompt":"x","size":"256x256"}"#.to_vec();
    let response = handle_image_api_request(&http, &service).expect("handler");
    let body = String::from_utf8_lossy(&response.body).into_owned();
    assert!(
        response.status == 500
            && body.contains("\"type\":\"server_error\"")
            && body.contains("image_generation_failed")
            && body.contains("\"param\":null"),
        "{body}"
    );
    assert!(
        ImageService::new(
            "",
            Box::new(|_, _| Err(Error::runtime("x"))),
            Arc::new(FakeCodec)
        )
        .is_err()
    );
}
