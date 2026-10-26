//! The client of the OpenAI-compatible endpoint (`http_client.cpp`), over
//! ureq and rustls: JSON GET and POST requests below a base URL, with a
//! bearer token, no redirects, a 30 s connection timeout, a deadline for the
//! whole request (at most 30 s for a GET) and at most 64 MiB of response.
//! Each request opens its own connection, as C++ does with a handle per
//! request, and the proxy comes from the environment like libcurl's.

use std::time::{Duration, Instant};

use ureq::tls::{RootCerts, TlsConfig};
use ureq::{Agent, Proxy};

/// The largest response body; a longer one fails the transfer.
const MAXIMUM_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// The deadline of a GET (model discovery); a POST waits for a whole
/// non-streaming completion.
const MAXIMUM_GET_TIMEOUT: Duration = Duration::from_secs(30);
const USER_AGENT: &str = "gufo-eval/1";

/// The outcome of one request; `transport_code` names a failed transfer:
/// "request_timeout" as in C++, otherwise the failure class (libcurl's
/// numeric codes in C++).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HttpResult {
    pub transport_ok: bool,
    pub status_code: u16,
    pub body: Vec<u8>,
    pub transport_code: &'static str,
    pub elapsed_ms: f64,
}

pub struct HttpClient {
    base_url: String,
    bearer_token: String,
    request_timeout: Duration,
    agent: Agent,
}

impl HttpClient {
    /// The deadline of one request (`kRequestTimeout`).
    pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

    /// A client of `base_url` (trailing slashes dropped); an empty
    /// `bearer_token` sends no `Authorization` header.
    pub fn new(base_url: &str, bearer_token: &str, request_timeout: Duration) -> Self {
        let base_url = normalize_base_url(base_url);
        let agent = Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .max_idle_connections(0)
            .max_idle_connections_per_host(0)
            .proxy(environment_proxy(&base_url))
            .user_agent(USER_AGENT)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .tls_config(
                TlsConfig::builder()
                    .root_certs(RootCerts::PlatformVerifier)
                    .build(),
            )
            .build()
            .new_agent();
        Self {
            base_url,
            bearer_token: bearer_token.to_owned(),
            request_timeout,
            agent,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn get(&self, relative_path: &str) -> HttpResult {
        self.request(false, relative_path, "")
    }

    pub fn post_json(&self, relative_path: &str, body: &str) -> HttpResult {
        self.request(true, relative_path, body)
    }

    fn request(&self, post: bool, relative_path: &str, body: &str) -> HttpResult {
        let mut url = self.base_url.clone();
        if !relative_path.starts_with('/') {
            url.push('/');
        }
        url.push_str(relative_path);
        let started = Instant::now();
        let mut result = match self.send(post, &url, body) {
            Ok((status_code, body)) => HttpResult {
                transport_ok: true,
                status_code,
                body,
                ..HttpResult::default()
            },
            Err(error) => HttpResult {
                transport_code: transport_code(&error),
                ..HttpResult::default()
            },
        };
        result.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        result
    }

    /// The status and body of any HTTP response.
    fn send(&self, post: bool, url: &str, body: &str) -> Result<(u16, Vec<u8>), ureq::Error> {
        let authorization = format!("Bearer {}", self.bearer_token);
        let mut response = if post {
            let mut request = self
                .agent
                .post(url)
                .config()
                .timeout_global(Some(self.request_timeout))
                .build()
                .header("Accept", "application/json")
                .header("Content-Type", "application/json");
            if !self.bearer_token.is_empty() {
                request = request.header("Authorization", &authorization);
            }
            request.send(body)?
        } else {
            let mut request = self
                .agent
                .get(url)
                .config()
                .timeout_global(Some(self.request_timeout.min(MAXIMUM_GET_TIMEOUT)))
                .build()
                .header("Accept", "application/json");
            if !self.bearer_token.is_empty() {
                request = request.header("Authorization", &authorization);
            }
            request.call()?
        };
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .with_config()
            .limit(MAXIMUM_RESPONSE_BYTES)
            .read_to_vec()?;
        Ok((status, body))
    }
}

/// The failure class of a transfer; a timeout in any phase is
/// "request_timeout", as libcurl's `CURLE_OPERATION_TIMEDOUT` is.
fn transport_code(error: &ureq::Error) -> &'static str {
    use std::io::ErrorKind;
    use ureq::Error;
    match error {
        Error::Timeout(_) => "request_timeout",
        Error::Io(io) if io.kind() == ErrorKind::TimedOut => "request_timeout",
        Error::HostNotFound => "host_not_found",
        Error::ConnectionFailed => "connection_failed",
        Error::Io(io) if io.kind() == ErrorKind::ConnectionRefused => "connection_failed",
        Error::BodyExceedsLimit(_) => "response_too_large",
        Error::Tls(_) | Error::Pem(_) | Error::Rustls(_) | Error::TlsRequired => "tls_failed",
        Error::Protocol(_) | Error::LargeResponseHeader(..) | Error::Http(_) => "protocol_error",
        Error::Io(_) => "io_error",
        _ => "transfer_failed",
    }
}

/// libcurl's proxy of `base_url` from the environment: `http_proxy` (only
/// lowercase) for HTTP, `HTTPS_PROXY` or `https_proxy` for HTTPS, else
/// `ALL_PROXY` or `all_proxy`; none for a host in `NO_PROXY`/`no_proxy`.
fn environment_proxy(base_url: &str) -> Option<Proxy> {
    let variable = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    let scheme: &[&str] = if base_url.starts_with("https://") {
        &["HTTPS_PROXY", "https_proxy"]
    } else {
        &["http_proxy"]
    };
    let url = scheme
        .iter()
        .chain(&["ALL_PROXY", "all_proxy"])
        .find_map(|name| variable(name))?;
    let no_proxy = variable("NO_PROXY").or_else(|| variable("no_proxy"));
    if no_proxy.is_some_and(|list| bypasses_proxy(&list, url_host(base_url))) {
        return None;
    }
    Proxy::new(&url).ok()
}

/// The host of a URL, without port, user information or IPv6 brackets.
fn url_host(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = &rest[..rest.find('/').unwrap_or(rest.len())];
    let authority = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    match authority.strip_prefix('[') {
        Some(bracketed) => &bracketed[..bracketed.find(']').unwrap_or(bracketed.len())],
        None => &authority[..authority.find(':').unwrap_or(authority.len())],
    }
}

/// libcurl's `NO_PROXY` match: `*` matches every host, a name matches
/// itself and its subdomains (a leading dot is ignored), and an address
/// matches itself or, with `/bits`, its network.
fn bypasses_proxy(list: &str, host: &str) -> bool {
    let address = host.parse::<std::net::IpAddr>().ok();
    list.split([',', ' '])
        .filter(|entry| !entry.is_empty())
        .any(|entry| {
            if entry == "*" {
                return true;
            }
            if let Some(address) = address {
                let (network, bits) = entry.split_once('/').unwrap_or((entry, ""));
                let network = network.trim_start_matches('[').trim_end_matches(']');
                let Ok(network) = network.parse::<std::net::IpAddr>() else {
                    return false;
                };
                let prefix = |a: &[u8], b: &[u8], bits: usize| {
                    let bytes = bits / 8;
                    let rest = bits % 8;
                    a[..bytes] == b[..bytes]
                        && (rest == 0 || (a[bytes] ^ b[bytes]) >> (8 - rest) == 0)
                };
                return match (address, network) {
                    (std::net::IpAddr::V4(a), std::net::IpAddr::V4(n)) => match bits.parse() {
                        Ok(bits @ 0..=32) => prefix(&a.octets(), &n.octets(), bits),
                        _ => bits.is_empty() && a == n,
                    },
                    (std::net::IpAddr::V6(a), std::net::IpAddr::V6(n)) => match bits.parse() {
                        Ok(bits @ 0..=128) => prefix(&a.octets(), &n.octets(), bits),
                        _ => bits.is_empty() && a == n,
                    },
                    _ => false,
                };
            }
            let name = entry.strip_prefix('.').unwrap_or(entry);
            let host = host.strip_suffix('.').unwrap_or(host);
            host.eq_ignore_ascii_case(name)
                || host.len() > name.len()
                    && host.as_bytes()[host.len() - name.len() - 1] == b'.'
                    && host[host.len() - name.len()..].eq_ignore_ascii_case(name)
        })
}

/// `ValidateBaseUrl`.
pub fn validate_base_url(base_url: &str) -> Result<(), String> {
    let Some(scheme_end) = ["http://", "https://"]
        .iter()
        .find_map(|scheme| base_url.starts_with(scheme).then_some(scheme.len()))
    else {
        return Err("--base-url must use http:// or https://".into());
    };
    if base_url.contains('?') || base_url.contains('#') {
        return Err("--base-url must not contain a query or fragment".into());
    }
    let rest = &base_url[scheme_end..];
    let authority = &rest[..rest.find('/').unwrap_or(rest.len())];
    if authority.is_empty() {
        return Err("--base-url is missing a host".into());
    }
    if authority.contains('@') {
        return Err("--base-url must not embed credentials".into());
    }
    Ok(())
}

/// `NormalizeBaseUrl`: without trailing slashes.
pub fn normalize_base_url(base_url: &str) -> String {
    base_url.trim_end_matches('/').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_proxy_matches_like_libcurl() {
        assert_eq!(url_host("http://user@[::1]:8080/v1"), "::1");
        assert_eq!(url_host("https://Example.com:443"), "Example.com");
        for (list, host, bypass) in [
            ("*", "anything", true),
            ("example.com", "example.com", true),
            ("example.com", "api.EXAMPLE.com", true),
            (".example.com", "example.com", true),
            ("example.com", "badexample.com", false),
            ("localhost, 127.0.0.1", "127.0.0.1", true),
            ("10.0.0.0/8", "10.1.2.3", true),
            ("10.0.0.0/8", "11.1.2.3", false),
            ("192.168.1.0/25", "192.168.1.130", false),
            ("::1", "::1", true),
            ("fd00::/8", "fd12::1", true),
            ("127.0.0.1", "localhost", false),
        ] {
            assert_eq!(bypasses_proxy(list, host), bypass, "{list} {host}");
        }
    }
}
