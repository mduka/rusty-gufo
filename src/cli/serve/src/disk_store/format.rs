//! The on-disk format: keys, headers, metadata and whole images.

use super::*;

/// The runner's persistence descriptor; an input identity (an image) extends
/// the compatibility identity to `IMG\1`, the model identity, its length as
/// `u64` and the input identity (`DescriptorForInput`).
pub(super) fn persistence_for_input<R: TextModelRunner + ?Sized>(
    runner: &R,
    input_identity: &[u8],
) -> Option<TextRunnerPersistenceDescriptor> {
    let mut persistence = runner.descriptor().persistence?;
    if input_identity.is_empty() {
        return Some(persistence);
    }
    let identity = &mut persistence.compatibility_identity;
    let model_size = identity.len() as u64;
    identity.splice(0..0, *b"IMG\x01");
    identity.extend_from_slice(&model_size.to_le_bytes());
    identity.extend_from_slice(input_identity);
    Some(persistence)
}

/// Calls `visit(begin, end, identity)` for each input identity range: entries
/// of `begin..=end` tokens (at most `token_count`) were saved with `identity`
/// (`VisitInputRanges`).
pub(super) fn visit_input_ranges(
    token_count: usize,
    input: InputIdentity<'_>,
    mut visit: impl FnMut(usize, usize, &[u8]),
) {
    let mut begin = 0;
    for prefix in input.prefixes {
        let end = prefix.token_count.min(token_count);
        if end >= begin {
            visit(begin, end, &prefix.identity);
        }
        if end == token_count {
            return;
        }
        begin = begin.max(end + 1);
    }
    visit(begin, token_count, input.complete);
}

pub(super) fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut value = [0; 4];
    value.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(value)
}

pub(super) fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut value = [0; 8];
    value.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(value)
}

/// The hashed lookup key: identity length, payload version and token count
/// as `u32`, the identity, then the tokens as `u32` (`MakeKeyBytes`).
pub(super) fn make_key_bytes(
    persistence: &TextRunnerPersistenceDescriptor,
    tokens: &[Token],
) -> Result<Vec<u8>> {
    let identity = &persistence.compatibility_identity;
    let (Ok(identity_bytes), Ok(token_count)) =
        (u32::try_from(identity.len()), u32::try_from(tokens.len()))
    else {
        return Err(Error::runtime("continuation disk key is too large"));
    };
    let mut key = Vec::with_capacity(12 + identity.len() + tokens.len() * 4);
    key.extend_from_slice(&identity_bytes.to_le_bytes());
    key.extend_from_slice(&persistence.payload_version.to_le_bytes());
    key.extend_from_slice(&token_count.to_le_bytes());
    key.extend_from_slice(identity);
    for token in tokens {
        key.extend_from_slice(&token.to_le_bytes());
    }
    Ok(key)
}

pub(super) fn checked_file_bytes(
    identity_bytes: usize,
    token_count: usize,
    payload_bytes: usize,
) -> Result<usize> {
    if identity_bytes > MAX_IDENTITY_BYTES || token_count > MAX_TOKEN_COUNT {
        return Err(Error::invalid(
            "continuation disk metadata exceeds format limits",
        ));
    }
    let metadata_bytes = HEADER_BYTES + identity_bytes + token_count * 4;
    metadata_bytes
        .checked_add(payload_bytes)
        .ok_or_else(|| Error::runtime("continuation disk file size overflows"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ParseFailure {
    Corrupt,
    Checksum,
}

pub(super) struct ParsedImage {
    pub(super) persistence: TextRunnerPersistenceDescriptor,
    pub(super) tokens: Vec<Token>,
    pub(super) payload_offset: usize,
    pub(super) payload_bytes: usize,
}

/// The sizes a file header declares (`ParseHeader`).
pub(super) struct Header {
    pub(super) payload_version: u32,
    pub(super) identity_bytes: usize,
    pub(super) token_count: usize,
    pub(super) payload_bytes: usize,
}

impl Header {
    /// The bytes before the payload: header, identity and tokens.
    pub(super) fn metadata_bytes(&self) -> usize {
        HEADER_BYTES + self.identity_bytes + self.token_count * 4
    }
}

/// Checks a header against the length of its file, without the payload:
/// magic, versions, limits, the declared length and a well-formed checksum
/// field.
pub(super) fn parse_header(
    header: &[u8],
    file_bytes: usize,
) -> std::result::Result<Header, ParseFailure> {
    if header.len() < HEADER_BYTES || &header[..MAGIC.len()] != MAGIC {
        return Err(ParseFailure::Corrupt);
    }
    let file_version = le_u32(header, FILE_VERSION_OFFSET);
    let payload_version = le_u32(header, PAYLOAD_VERSION_OFFSET);
    let identity_bytes = le_u32(header, IDENTITY_BYTES_OFFSET) as usize;
    let token_count = le_u32(header, TOKEN_COUNT_OFFSET) as usize;
    let payload_bytes = le_u64(header, PAYLOAD_BYTES_OFFSET);
    if file_version != FILE_VERSION
        || payload_version == 0
        || identity_bytes == 0
        || identity_bytes > MAX_IDENTITY_BYTES
        || token_count == 0
        || token_count > MAX_TOKEN_COUNT
        || payload_bytes == 0
    {
        return Err(ParseFailure::Corrupt);
    }
    let payload_bytes = usize::try_from(payload_bytes).map_err(|_| ParseFailure::Corrupt)?;
    let expected = checked_file_bytes(identity_bytes, token_count, payload_bytes)
        .map_err(|_| ParseFailure::Corrupt)?;
    if expected != file_bytes || !is_lower_hex_digest(&header[CHECKSUM_OFFSET..HEADER_BYTES]) {
        return Err(ParseFailure::Corrupt);
    }
    Ok(Header {
        payload_version,
        identity_bytes,
        token_count,
        payload_bytes,
    })
}

/// The identity and tokens after a checked header (`metadata` holds at
/// least [`Header::metadata_bytes`]).
pub(super) fn parse_metadata(metadata: &[u8], header: &Header) -> ParsedImage {
    let identity = metadata[HEADER_BYTES..HEADER_BYTES + header.identity_bytes].to_vec();
    let token_offset = HEADER_BYTES + header.identity_bytes;
    let tokens = (0..header.token_count)
        .map(|index| le_u32(metadata, token_offset + index * 4))
        .collect();
    ParsedImage {
        persistence: TextRunnerPersistenceDescriptor {
            compatibility_identity: identity,
            payload_version: header.payload_version,
        },
        tokens,
        payload_offset: header.metadata_bytes(),
        payload_bytes: header.payload_bytes,
    }
}

/// Checks the header and the checksum; the checksum field is zeroed while
/// hashing and restored afterwards (`ParseAndVerifyImage`).
pub(super) fn parse_and_verify_image(
    image: &mut [u8],
) -> std::result::Result<ParsedImage, ParseFailure> {
    let header = parse_header(image, image.len())?;
    let checksum = CHECKSUM_OFFSET..HEADER_BYTES;
    let stored = image[checksum.clone()].to_vec();
    image[checksum.clone()].fill(0);
    let computed = sha256_hex(image);
    image[checksum].copy_from_slice(&stored);
    if computed.as_bytes() != stored.as_slice() {
        return Err(ParseFailure::Checksum);
    }
    Ok(parse_metadata(image, &header))
}

/// Header, identity and tokens with a zeroed checksum field (`BuildHeader`).
pub(super) fn build_header(
    persistence: &TextRunnerPersistenceDescriptor,
    tokens: &[Token],
    payload_bytes: usize,
) -> Result<Vec<u8>> {
    let identity = &persistence.compatibility_identity;
    let file_bytes = checked_file_bytes(identity.len(), tokens.len(), payload_bytes)?;
    let mut image = Vec::with_capacity(file_bytes - payload_bytes);
    image.extend_from_slice(MAGIC);
    image.extend_from_slice(&FILE_VERSION.to_le_bytes());
    image.extend_from_slice(&persistence.payload_version.to_le_bytes());
    // Both fit: checked_file_bytes bounds them far below u32::MAX.
    image.extend_from_slice(&(identity.len() as u32).to_le_bytes());
    image.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    image.extend_from_slice(&(payload_bytes as u64).to_le_bytes());
    image.resize(HEADER_BYTES, 0);
    image.extend_from_slice(identity);
    for token in tokens {
        image.extend_from_slice(&token.to_le_bytes());
    }
    Ok(image)
}

pub(super) fn common_prefix(a: &[Token], b: &[Token]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

pub(super) fn write_hashed(file: &mut File, hasher: &mut Sha256, bytes: &[u8]) -> Result<()> {
    for chunk in bytes.chunks(WRITE_CHUNK_BYTES) {
        file.write_all(chunk)
            .map_err(|_| Error::runtime("continuation disk write failed"))?;
        hasher.update(chunk);
    }
    Ok(())
}

/// Header, streamed payload, then the checksum over both written into the
/// header (`PublishImage`).
pub(super) fn write_image<R: TextModelRunner>(
    file: &mut File,
    header: &[u8],
    runner: &R,
    snapshot: &R::Snapshot,
    payload_bytes: usize,
) -> Result<()> {
    let mut hasher = Sha256::new();
    write_hashed(file, &mut hasher, header)?;
    let mut written = 0;
    runner.stream_persistent_snapshot(snapshot, &mut |bytes| {
        if bytes.len() > payload_bytes - written {
            return Err(Error::runtime("snapshot serialization overflow"));
        }
        write_hashed(file, &mut hasher, bytes)?;
        written += bytes.len();
        Ok(())
    })?;
    if written != payload_bytes {
        return Err(Error::runtime("snapshot serialization truncated"));
    }
    let checksum = hex(&hasher.finalize());
    file.seek(SeekFrom::Start(CHECKSUM_OFFSET as u64))
        .and_then(|_| file.write_all(checksum.as_bytes()))
        .map_err(|_| Error::runtime("snapshot checksum write failed"))
}
