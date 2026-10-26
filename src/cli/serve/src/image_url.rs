//! Chat `image_url` inputs (`ReadImageUrl`, `src/core/image.cpp`): base64
//! PNG/JPEG data URLs, and downloads from public HTTPS destinations over
//! ureq and rustls with the platform's trust store.

use std::io::Read;
use std::net::SocketAddr;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use gufo_core::text::{Error, Result};
use ureq::Agent;
use ureq::config::{AutoHeaderValue, Config};
use ureq::http::Uri;
use ureq::tls::{RootCerts, TlsConfig};
use ureq::unversioned::resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver};
use ureq::unversioned::transport::{DefaultConnector, NextTimeout};

pub const MAX_ENCODED_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// Shared across all messages of one HTTP request.
#[derive(Debug, Clone)]
/// No image count: the expanded image tokens must fit the model context
///.
pub struct ImageReadBudget {
    /// False when the loaded model takes no image input: reading is refused
    /// before any data URL is decoded or any URL is fetched.
    pub accepts_images: bool,
    pub remaining_bytes: usize,
    pub deadline: Instant,
}

impl Default for ImageReadBudget {
    fn default() -> Self {
        Self {
            accepts_images: true,
            remaining_bytes: MAX_ENCODED_IMAGE_BYTES,
            deadline: Instant::now() + Duration::from_secs(15),
        }
    }
}

/// `IsPublicImageAddress`: a public IPv4 address, or native global IPv6
/// unicast without the mapped IPv4, NAT64, Teredo and 6to4 ranges, which
/// could reach an embedded private IPv4 address.
pub fn is_public_image_address(address: &[u8]) -> bool {
    if let &[a, b, c, _] = address {
        return a != 0
            && a != 10
            && a != 127
            && a < 224
            && !(a == 100 && (64..=127).contains(&b))
            && !(a == 169 && b == 254)
            && !(a == 172 && (16..=31).contains(&b))
            && !(a == 192 && (b == 168 || (b == 0 && (c == 0 || c == 2)) || (b == 88 && c == 99)))
            && !(a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100)))
            && !(a == 203 && b == 0 && c == 113);
    }
    if address.len() != 16 || address[0] & 0xe0 != 0x20 {
        return false;
    }
    if address[0] == 0x20
        && ((address[1] == 0x01 && (address[2] < 2 || (address[2] == 0x0d && address[3] == 0xb8)))
            || address[1] == 0x02)
    {
        return false;
    }
    !(address[0] == 0x3f && address[1] == 0xff && address[2] & 0xf0 == 0)
}

/// The check of libcurl's socket callback: a public address, and no IPv6
/// scope.
fn is_public_socket_address(address: &SocketAddr) -> bool {
    match address {
        SocketAddr::V4(v4) => is_public_image_address(&v4.ip().octets()),
        SocketAddr::V6(v6) => v6.scope_id() == 0 && is_public_image_address(&v6.ip().octets()),
    }
}

/// Resolves only to public addresses, so every connection of an image
/// download, redirects included, reaches one; refuses user names in the
/// URL like `CURLOPT_DISALLOW_USERNAME_IN_URL`, also after a redirect.
#[derive(Debug, Default)]
struct PublicAddresses(DefaultResolver);

impl Resolver for PublicAddresses {
    fn resolve(
        &self,
        uri: &Uri,
        config: &Config,
        timeout: NextTimeout,
    ) -> std::result::Result<ResolvedSocketAddrs, ureq::Error> {
        if uri
            .authority()
            .is_some_and(|authority| authority.as_str().contains('@'))
        {
            return Err(ureq::Error::BadUri("user name in an image URL".into()));
        }
        let mut public = self.empty();
        for address in self.0.resolve(uri, config, timeout)?.iter() {
            if is_public_socket_address(address) {
                public.push(*address);
            }
        }
        if public.is_empty() {
            Err(ureq::Error::HostNotFound)
        } else {
            Ok(public)
        }
    }
}

/// The client of image downloads: HTTPS only, also for at most three
/// redirects; no proxy, which would hide the destination from the address
/// check; HTTP errors fail; and every download connects anew, like a
/// libcurl handle per image.
static IMAGE_AGENT: LazyLock<Agent> = LazyLock::new(|| {
    let config = Agent::config_builder()
        .https_only(true)
        .proxy(None)
        .max_redirects(3)
        .max_idle_connections(0)
        .max_idle_connections_per_host(0)
        .user_agent(AutoHeaderValue::None)
        .tls_config(
            TlsConfig::builder()
                .root_certs(RootCerts::PlatformVerifier)
                .build(),
        )
        .build();
    Agent::with_parts(config, DefaultConnector::new(), PublicAddresses::default())
});

/// libcurl's connection timeout of an image download, within the deadline.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// libcurl's largest write callback (`CURL_MAX_WRITE_SIZE`): the unit in
/// which a download is charged.
const CHUNK_BYTES: usize = 16 * 1024;

/// Downloads `url` within `remaining` (whole milliseconds), charging its
/// bytes to `budget` as they arrive; a failed download stays charged.
fn download(url: &str, remaining: Duration, budget: &mut ImageReadBudget) -> Result<Vec<u8>> {
    let failed = || Error::invalid("cannot download image within size/time limits");
    let mut response = IMAGE_AGENT
        .get(url)
        .config()
        .timeout_global(Some(remaining))
        .timeout_connect(Some(remaining.min(CONNECT_TIMEOUT)))
        .build()
        .call()
        .map_err(|_| failed())?;
    // `CURLOPT_MAXFILESIZE`: a declared size beyond the budget fails first.
    if response
        .body()
        .content_length()
        .is_some_and(|length| length > budget.remaining_bytes as u64)
    {
        return Err(failed());
    }
    let mut reader = response.body_mut().as_reader();
    let mut bytes = Vec::new();
    let mut chunk = vec![0; CHUNK_BYTES];
    loop {
        let read = reader.read(&mut chunk).map_err(|_| failed())?;
        if read == 0 {
            break;
        }
        if read > budget.remaining_bytes {
            return Err(failed());
        }
        budget.remaining_bytes -= read;
        bytes.extend_from_slice(&chunk[..read]);
    }
    if bytes.is_empty() {
        return Err(failed());
    }
    Ok(bytes)
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The value of each byte as a base64 digit; 0xff for every other byte,
/// padding included.
const DIGITS: [u8; 256] = {
    let mut table = [0xff; 256];
    let mut i = 0;
    while i < 64 {
        table[BASE64[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// One quad the way `DecodeBase64` reads it, character by character, with
/// its error messages; `last` allows padding.
fn decode_quad(quad: &[u8], last: bool) -> Result<(u32, u32)> {
    let mut bits: u32 = 0;
    let mut padding = 0;
    for (j, &c) in quad.iter().enumerate() {
        if c == b'=' {
            if j < 2 || !last {
                return Err(Error::invalid("invalid base64 image padding"));
            }
            padding += 1;
            bits <<= 6;
        } else {
            match DIGITS[usize::from(c)] {
                digit if digit != 0xff && padding == 0 => bits = (bits << 6) | u32::from(digit),
                _ => return Err(Error::invalid("invalid base64 image character")),
            }
        }
    }
    if (padding == 1 && bits & 0xff != 0) || (padding == 2 && bits & 0xffff != 0) {
        return Err(Error::invalid("noncanonical base64 image"));
    }
    Ok((bits, padding))
}

/// Canonical base64 of an encoded image (`DecodeBase64`). Whole quads
/// decode through a table; a quad with anything else takes the
/// character-by-character path for its error.
pub fn decode_base64(input: &[u8]) -> Result<Vec<u8>> {
    if input.is_empty()
        || !input.len().is_multiple_of(4)
        || input.len() > MAX_ENCODED_IMAGE_BYTES.div_ceil(3) * 4
    {
        return Err(Error::invalid("invalid or oversized base64 image"));
    }
    let mut output = Vec::with_capacity(input.len() / 4 * 3);
    let (body, last) = input.split_at(input.len() - 4);
    for quad in body.as_chunks::<4>().0 {
        let digits = quad.map(|c| DIGITS[usize::from(c)]);
        if digits.contains(&0xff) {
            decode_quad(quad, false)?;
            return Err(Error::invalid("invalid base64 image character"));
        }
        let bits = u32::from(digits[0]) << 18
            | u32::from(digits[1]) << 12
            | u32::from(digits[2]) << 6
            | u32::from(digits[3]);
        output.extend_from_slice(&[(bits >> 16) as u8, (bits >> 8) as u8, bits as u8]);
    }
    let (bits, padding) = decode_quad(last, true)?;
    output.push((bits >> 16) as u8);
    if padding < 2 {
        output.push((bits >> 8) as u8);
    }
    if padding < 1 {
        output.push(bits as u8);
    }
    Ok(output)
}

/// Base64 data URLs or public HTTPS destinations.
pub fn read_image_url(url: &str, budget: &mut ImageReadBudget) -> Result<Vec<u8>> {
    if !budget.accepts_images {
        return Err(Error::invalid("model does not support image input"));
    }
    // Whole milliseconds, which also bound the download.
    let remaining = Duration::from_millis(
        u64::try_from(
            budget
                .deadline
                .saturating_duration_since(Instant::now())
                .as_millis(),
        )
        .unwrap_or(u64::MAX),
    );
    if budget.remaining_bytes == 0 || remaining.is_zero() {
        return Err(Error::invalid("request image byte or time budget exceeded"));
    }
    if url.starts_with("data:") {
        let Some(comma) = url.find(',').filter(|&comma| comma <= 256) else {
            return Err(Error::invalid("image data URL has no data"));
        };
        // A media type, parameters and ";base64", case-insensitive (RFC
        // 2397): clients differ in case, "image/jpg" and parameters such as
        // a file name, and the decoder tells the formats apart by their
        // bytes anyway.
        let header = url[5..comma].to_ascii_lowercase();
        if !header.ends_with(";base64") {
            return Err(Error::invalid("image data URL must be base64-encoded"));
        }
        let media = header.split(';').next().unwrap_or_default();
        if !matches!(
            media,
            "image/png" | "image/jpeg" | "image/jpg" | "image/webp"
        ) {
            return Err(Error::invalid(format!(
                "image data URL type \"{media}\" is not supported; use PNG, JPEG or WebP"
            )));
        }
        let encoded = &url[comma + 1..];
        // Check the decoded size before allocating, including padding.
        let padding = if encoded.ends_with("==") {
            2
        } else if encoded.ends_with('=') {
            1
        } else {
            0
        };
        let decoded = encoded.len() / 4 * 3;
        if decoded < padding || decoded - padding > budget.remaining_bytes {
            return Err(Error::invalid("request image byte budget exceeded"));
        }
        let bytes = decode_base64(encoded.as_bytes())?;
        budget.remaining_bytes -= bytes.len();
        return Ok(bytes);
    }
    if !url.starts_with("https://") || url.len() > 8192 {
        return Err(Error::invalid(
            "image URL must use HTTPS or a base64 data URL",
        ));
    }
    download(url, remaining, budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_urls_decode_within_the_budget() {
        let mut budget = ImageReadBudget::default();
        assert_eq!(
            read_image_url("data:image/png;base64,AAEC", &mut budget).expect("decode"),
            [0, 1, 2]
        );
        assert_eq!(budget.remaining_bytes, MAX_ENCODED_IMAGE_BYTES - 3);
        for bad in [
            "data:image/gif;base64,AAEC",
            "data:image/png;base64,AA=C",
            "http://example.com/a.png",
        ] {
            assert!(read_image_url(bad, &mut budget).is_err(), "{bad}");
        }
        // No image count: more than sixteen fit, until the bytes run out
        //.
        budget.remaining_bytes = 17 * 3;
        for _ in 0..17 {
            read_image_url("data:image/png;base64,AAEC", &mut budget).expect("within the bytes");
        }
        let error = read_image_url("data:image/png;base64,AAEC", &mut budget).unwrap_err();
        assert!(error.message().contains("byte"), "{}", error.message());
    }

    /// Clients' spellings of image data URLs.
    #[test]
    fn data_url_spellings() {
        for accepted in [
            "data:image/jpeg;base64,/9gAAQI=",
            "data:image/jpg;base64,/9gAAQI=",
            "data:IMAGE/JPEG;BASE64,/9gAAQI=",
            "data:image/jpeg;name=photo.jpg;base64,/9gAAQI=",
            "data:image/webp;base64,/9gAAQI=",
        ] {
            let mut budget = ImageReadBudget::default();
            assert_eq!(
                read_image_url(accepted, &mut budget).expect(accepted),
                [0xff, 0xd8, 0, 1, 2],
                "{accepted}"
            );
        }
        for (rejected, message) in [
            (
                "data:image/gif;base64,AA==",
                "\"image/gif\" is not supported",
            ),
            ("data:image/png,AA==", "base64"),
            ("data:image/png;base64", "no data"),
            ("data:image/png;base64,A===", "base64"),
        ] {
            let mut budget = ImageReadBudget::default();
            let error = read_image_url(rejected, &mut budget)
                .expect_err(rejected)
                .to_string();
            assert!(error.contains(message), "{rejected}: {error}");
        }
    }

    /// `TestImageTransportLimits` of `tests/models/qwen27b/vision_test.cpp`.
    #[test]
    fn only_public_addresses_qualify() {
        let octets = |address: &str| match address.parse().expect("address") {
            std::net::IpAddr::V4(v4) => v4.octets().to_vec(),
            std::net::IpAddr::V6(v6) => v6.octets().to_vec(),
        };
        for address in [
            "0.0.0.0",
            "10.1.2.3",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.169.254",
            "172.16.1.2",
            "192.168.0.1",
            "198.18.0.1",
            "224.0.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "64:ff9b::a00:1",
            "fc00::1",
            "fe80::1",
            "2002:7f00:1::",
            "2001::1",
            "2001:db8::1",
            "3fff::1",
        ] {
            assert!(!is_public_image_address(&octets(address)), "{address}");
        }
        for address in ["1.1.1.1", "8.8.8.8", "2001:4860:4860::8888"] {
            assert!(is_public_image_address(&octets(address)), "{address}");
        }
        let scoped: SocketAddr = "[2001:4860:4860::8888%2]:443".parse().expect("address");
        assert!(!is_public_socket_address(&scoped));
    }

    /// Private destinations fail before any connection, so no network is
    /// needed; the image stays counted and no bytes are charged.
    #[test]
    fn https_images_reach_only_public_addresses() {
        for url in [
            "https://127.0.0.1:1/image.png",
            "https://[::1]:1/image.png",
            "https://2130706433:1/image.png",
            "https://localhost:1/image.png",
            "https://10.0.0.1/a.png",
            "https://user@example.com/a.png",
        ] {
            let mut budget = ImageReadBudget {
                deadline: Instant::now() + Duration::from_secs(5),
                ..ImageReadBudget::default()
            };
            let error = read_image_url(url, &mut budget).expect_err(url);
            assert_eq!(
                error.kind(),
                gufo_core::text::ErrorKind::InvalidArgument,
                "{url}"
            );
            assert_eq!(
                error.message(),
                "cannot download image within size/time limits",
                "{url}"
            );
            assert_eq!(budget.remaining_bytes, MAX_ENCODED_IMAGE_BYTES, "{url}");
        }
    }

    #[test]
    fn base64_decodes_every_length_and_names_each_error() {
        let bytes: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        for len in 1..40 {
            let data = &bytes[..len];
            let mut text = String::new();
            for chunk in data.chunks(3) {
                let n = u32::from(chunk[0]) << 16
                    | u32::from(*chunk.get(1).unwrap_or(&0)) << 8
                    | u32::from(*chunk.get(2).unwrap_or(&0));
                for i in 0..4 {
                    text.push(if i <= chunk.len() {
                        char::from(BASE64[(n >> (18 - 6 * i) & 63) as usize])
                    } else {
                        '='
                    });
                }
            }
            assert_eq!(decode_base64(text.as_bytes()).expect("decode"), data);
        }
        let error = |text: &str| {
            decode_base64(text.as_bytes())
                .unwrap_err()
                .message()
                .to_owned()
        };
        assert_eq!(error("AA=AAAAA"), "invalid base64 image padding");
        assert_eq!(error("A@=AAAAA"), "invalid base64 image character");
        assert_eq!(error("AAAAA==="), "invalid base64 image padding");
        assert_eq!(error("AAAAAA=A"), "invalid base64 image character");
        assert_eq!(error("AAAAAB=="), "noncanonical base64 image");
        assert_eq!(error("AAA"), "invalid or oversized base64 image");
    }
}
