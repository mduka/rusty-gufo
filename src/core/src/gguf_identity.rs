//! Full-content GGUF identities for restart-safe caches, a port of
//! `GgufIdentityHex` (`src/core/gguf_identity.cpp`): the SHA-256 of the scheme
//! tag and, per shard, its size and the SHA-256 of all its bytes. File digests
//! are cached by device, inode, size, mtime and ctime in the same private
//! directory as the C++ server, so either side reuses the other's scans.

use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

/// Scheme tag for full-content GGUF identities (`kGgufIdentityScheme`).
pub const SCHEME: &str = "gguf-sha256-v1";

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// `dev:ino:size:mtime_s:mtime_ns:ctime_s:ctime_ns` (`FileStamp`).
fn stamp(file: &File, size: u64) -> Result<String, String> {
    let m = file
        .metadata()
        .map_err(|_| "GGUF file changed or cannot be inspected".to_string())?;
    if !m.is_file() || m.size() != size {
        return Err("GGUF file changed or cannot be inspected".into());
    }
    Ok(format!(
        "{}:{}:{}:{}:{}:{}:{}",
        m.dev(),
        m.ino(),
        m.size(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec()
    ))
}

/// `$XDG_CACHE_HOME/gufo/gguf-sha256-v1` (else `~/.cache/...`), owned by
/// this user and closed to others; `None` disables the cache.
fn cache_directory() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|b| !b.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .map(|h| PathBuf::from(h).join(".cache"))
        })?;
    let parent = base.join("gufo");
    fs::create_dir_all(&parent).ok()?;
    let path = parent.join(SCHEME);
    let _ = DirBuilder::new().mode(0o700).create(&path);
    let m = fs::symlink_metadata(&path).ok()?;
    let owned = m.uid() == rustix::process::geteuid().as_raw();
    (m.is_dir() && owned && m.mode() & 0o077 == 0).then_some(path)
}

fn read_digest(directory: &Path, key: &str) -> Option<String> {
    let path = directory.join(key);
    let m = fs::symlink_metadata(&path).ok()?;
    if !m.is_file()
        || m.uid() != rustix::process::geteuid().as_raw()
        || m.mode() & 0o022 != 0
        || m.size() != 65
    {
        return None;
    }
    let mut bytes = Vec::with_capacity(65);
    File::open(&path).ok()?.read_to_end(&mut bytes).ok()?;
    let (digest, newline) = bytes.split_at(64);
    (newline == b"\n"
        && digest
            .iter()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)))
    .then(|| String::from_utf8_lossy(digest).into_owned())
}

fn store_digest(directory: &Path, key: &str, digest: &str) {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let temp = directory.join(format!(
        ".{key}.{}.{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    let Ok(mut file) = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp)
    else {
        return;
    };
    if file.write_all(format!("{digest}\n").as_bytes()).is_ok() {
        let _ = fs::rename(&temp, directory.join(key));
    }
    let _ = fs::remove_file(&temp);
}

fn hash_file(file: &mut File, size: u64) -> Result<String, String> {
    const CHUNK: usize = 8 * 1024 * 1024;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; CHUNK];
    let mut done = 0u64;
    while done < size {
        let want = CHUNK.min((size - done) as usize);
        file.read_exact(&mut buffer[..want])
            .map_err(|_| "cannot read complete GGUF for identity".to_string())?;
        hash.update(&buffer[..want]);
        done += want as u64;
    }
    Ok(hex(&hash.finalize()))
}

/// The SHA-256 of one shard's bytes, from the cache when its stamp matches
/// (`RegionDigest`).
fn file_digest(path: &Path, cache: Option<&Path>) -> Result<(u64, String), String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .size();
    let before = stamp(&file, size)?;
    let key = sha256_hex(before.as_bytes());
    let cached = cache.and_then(|dir| read_digest(dir, &key));
    let digest = match &cached {
        Some(digest) => digest.clone(),
        None => hash_file(&mut file, size)?,
    };
    if stamp(&file, size)? != before {
        return Err("GGUF file changed during identity lookup".into());
    }
    if cached.is_none()
        && let Some(dir) = cache
    {
        store_digest(dir, &key, &digest);
    }
    Ok((size, digest))
}

/// The identity of the artifact made of `shards`, in order.
pub fn identity_hex(shards: &[PathBuf]) -> Result<String, String> {
    identity_with_cache(shards, cache_directory().as_deref())
}

fn identity_with_cache(shards: &[PathBuf], cache: Option<&Path>) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(SCHEME.as_bytes());
    for shard in shards {
        let (size, digest) = file_digest(shard, cache)?;
        hash.update(format!(":{size}:").as_bytes());
        hash.update(digest.as_bytes());
    }
    Ok(hex(&hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_hashes_scheme_sizes_and_file_digests() {
        let dir = std::env::temp_dir().join(format!("gufo-identity-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a.gguf"), dir.join("b.gguf"));
        fs::write(&a, b"GGUF first shard").unwrap();
        fs::write(&b, vec![7u8; 9 * 1024 * 1024]).unwrap();
        let expected = sha256_hex(
            format!(
                "{SCHEME}:16:{}:{}:{}",
                sha256_hex(b"GGUF first shard"),
                9 * 1024 * 1024,
                sha256_hex(&vec![7u8; 9 * 1024 * 1024])
            )
            .as_bytes(),
        );
        let cache = dir.join("cache");
        DirBuilder::new().mode(0o700).create(&cache).unwrap();
        let shards = [a, b];
        assert_eq!(identity_with_cache(&shards, None).unwrap(), expected);
        assert_eq!(
            identity_with_cache(&shards, Some(&cache)).unwrap(),
            expected
        );
        assert_eq!(
            fs::read_dir(&cache).unwrap().count(),
            2,
            "one digest per shard"
        );
        // Cached digests are used as they are.
        for entry in fs::read_dir(&cache).unwrap() {
            fs::write(entry.unwrap().path(), format!("{}\n", "0".repeat(64))).unwrap();
        }
        assert_ne!(
            identity_with_cache(&shards, Some(&cache)).unwrap(),
            expected
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
