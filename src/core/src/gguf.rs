//! GGUF (versions 2 and 3): header, typed metadata and the tensor index, read
//! without copying from the mapped file. The checks are those of the C++
//! `GgufReader` (`src/core/gguf_reader.cpp`): descriptor counts bounded by
//! the file and a 128 MiB host budget, known value and tensor types, whole
//! blocks along the first dimension, aligned tensor data inside the file.

use std::collections::HashMap;

use gufo_formats::GgmlType;

const MAGIC: &[u8; 4] = b"GGUF";
const HOST_BUDGET: usize = 128 << 20;
const DEFAULT_ALIGNMENT: u64 = 32;

/// A metadata value. Integers and booleans in arrays widen to 64 bits and
/// floats to `f64`, as in the C++ reader.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    UInt(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(&'a [u8]),
    StrArray(Vec<&'a [u8]>),
    UIntArray(Vec<u64>),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

/// One tensor of the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<'a> {
    pub name: &'a str,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset in the file.
    pub offset: usize,
    pub bytes: usize,
}

impl Tensor<'_> {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

#[derive(Debug)]
pub struct Gguf<'a> {
    pub version: u32,
    pub alignment: u64,
    metadata: HashMap<&'a str, Value<'a>>,
    tensors: Vec<Tensor<'a>>,
    index: HashMap<&'a str, usize>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
    budget: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or("GGUF file is truncated")?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }

    /// Accounts host memory for `count` descriptors of `item` bytes.
    fn charge(&mut self, count: u64, item: usize) -> Result<(), String> {
        let bytes = usize::try_from(count)
            .ok()
            .and_then(|c| c.checked_mul(item))
            .filter(|&b| b <= self.budget)
            .ok_or("GGUF descriptors exceed the host memory budget")?;
        self.budget -= bytes;
        Ok(())
    }

    fn string(&mut self) -> Result<&'a [u8], String> {
        let len = usize::try_from(self.u64()?).map_err(|_| "GGUF string is too long")?;
        let s = self.take(len)?;
        self.charge(len as u64, 1)?;
        Ok(s)
    }

    fn value(&mut self, ty: u32) -> Result<Value<'a>, String> {
        Ok(match ty {
            0 => Value::UInt(u64::from(self.u8()?)),
            1 => Value::Int(i64::from(self.u8()? as i8)),
            2 => Value::UInt(u64::from(self.u16()?)),
            3 => Value::Int(i64::from(self.u16()? as i16)),
            4 => Value::UInt(u64::from(self.u32()?)),
            5 => Value::Int(i64::from(self.u32()? as i32)),
            6 => Value::Float(f64::from(f32::from_bits(self.u32()?))),
            7 => match self.u8()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err("invalid GGUF boolean".into()),
            },
            8 => Value::Str(self.string()?),
            9 => self.array_value()?,
            10 => Value::UInt(self.u64()?),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            _ => return Err(format!("unknown GGUF value type {ty}")),
        })
    }

    fn array_value(&mut self) -> Result<Value<'a>, String> {
        const ITEM_BYTES: [usize; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 8, 0, 8, 8, 8];
        let item = self.u32()?;
        let len = self.u64()?;
        if item > 12 || item == 9 {
            return Err(format!("invalid GGUF array item type {item}"));
        }
        let fits = usize::try_from(len)
            .ok()
            .is_some_and(|l| l <= self.remaining() / ITEM_BYTES[item as usize]);
        if !fits {
            return Err("GGUF array exceeds the file".into());
        }
        self.charge(len, if item == 8 { 16 } else { 8 })?;
        let len = len as usize;
        // Writers pick the narrowest integer type; every width widens.
        Ok(match item {
            8 => Value::StrArray((0..len).map(|_| self.string()).collect::<Result<_, _>>()?),
            0 => Value::UIntArray(
                (0..len)
                    .map(|_| self.u8().map(u64::from))
                    .collect::<Result<_, _>>()?,
            ),
            2 => Value::UIntArray(
                (0..len)
                    .map(|_| self.u16().map(u64::from))
                    .collect::<Result<_, _>>()?,
            ),
            4 => Value::UIntArray(
                (0..len)
                    .map(|_| self.u32().map(u64::from))
                    .collect::<Result<_, _>>()?,
            ),
            10 => Value::UIntArray((0..len).map(|_| self.u64()).collect::<Result<_, _>>()?),
            // Booleans widen to 0 and 1, as in the C++ reader.
            7 => Value::UIntArray(
                (0..len)
                    .map(|_| match self.u8()? {
                        value @ 0..=1 => Ok(u64::from(value)),
                        _ => Err("invalid GGUF boolean".to_string()),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            1 => Value::IntArray(
                (0..len)
                    .map(|_| self.u8().map(|v| i64::from(v as i8)))
                    .collect::<Result<_, _>>()?,
            ),
            3 => Value::IntArray(
                (0..len)
                    .map(|_| self.u16().map(|v| i64::from(v as i16)))
                    .collect::<Result<_, _>>()?,
            ),
            5 => Value::IntArray(
                (0..len)
                    .map(|_| self.u32().map(|v| i64::from(v as i32)))
                    .collect::<Result<_, _>>()?,
            ),
            11 => Value::IntArray(
                (0..len)
                    .map(|_| self.u64().map(|v| v as i64))
                    .collect::<Result<_, _>>()?,
            ),
            6 => Value::FloatArray(
                (0..len)
                    .map(|_| self.u32().map(|v| f64::from(f32::from_bits(v))))
                    .collect::<Result<_, _>>()?,
            ),
            12 => Value::FloatArray(
                (0..len)
                    .map(|_| self.u64().map(f64::from_bits))
                    .collect::<Result<_, _>>()?,
            ),
            _ => return Err(format!("invalid GGUF array item type {item}")),
        })
    }
}

fn utf8<'a>(bytes: &'a [u8], what: &str) -> Result<&'a str, String> {
    std::str::from_utf8(bytes).map_err(|_| format!("GGUF {what} is not UTF-8"))
}

impl<'a> Gguf<'a> {
    /// Parses the header, metadata and tensor index of a whole GGUF file.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        Self::parse_prefix(bytes, bytes.len())
    }

    /// Parses from the first bytes of a file of `file_len` bytes; tensor
    /// extents are checked against `file_len`. Fails as truncated when the
    /// prefix ends inside the header.
    pub fn parse_prefix(bytes: &'a [u8], file_len: usize) -> Result<Self, String> {
        if bytes.len() < 24 {
            return Err("file too small for a GGUF header".into());
        }
        if &bytes[..4] != MAGIC {
            return Err("invalid GGUF magic".into());
        }
        let mut c = Cursor {
            bytes,
            at: 4,
            budget: HOST_BUDGET,
        };
        let version = c.u32()?;
        if version != 2 && version != 3 {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let tensor_count = c.u64()?;
        let metadata_count = c.u64()?;
        if metadata_count > (c.remaining() / 13) as u64
            || tensor_count > (c.remaining() / 32) as u64
        {
            return Err("GGUF descriptor counts exceed the file".into());
        }
        c.charge(metadata_count, 128)?;
        c.charge(tensor_count, 192)?;

        let mut alignment = DEFAULT_ALIGNMENT;
        let mut metadata = HashMap::with_capacity(metadata_count as usize);
        for _ in 0..metadata_count {
            let key = c.string()?;
            if key.is_empty() || key.len() > 65535 {
                return Err("GGUF metadata key exceeds limits".into());
            }
            let key = utf8(key, "metadata key")?;
            let ty = c.u32()?;
            let value = c.value(ty)?;
            if key == "general.alignment" {
                alignment = match value {
                    Value::UInt(a) if a.is_power_of_two() && a <= u64::from(u32::MAX) => a,
                    _ => return Err("invalid GGUF alignment".into()),
                };
            }
            if metadata.insert(key, value).is_some() {
                return Err(format!("duplicate GGUF metadata key {key}"));
            }
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut index = HashMap::with_capacity(tensor_count as usize);
        let mut relative = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = c.string()?;
            if name.is_empty() || name.len() > 65535 {
                return Err("invalid GGUF tensor name".into());
            }
            let name = utf8(name, "tensor name")?;
            let n_dims = c.u32()?;
            if n_dims == 0 || n_dims > 8 {
                return Err(format!("invalid dimension count of {name}"));
            }
            let dims: Vec<u64> = (0..n_dims).map(|_| c.u64()).collect::<Result<_, _>>()?;
            let raw_type = c.u32()?;
            let ty = GgmlType::from_id(raw_type)
                .ok_or_else(|| format!("unsupported storage type {raw_type} of {name}"))?;
            let elements = dims.iter().try_fold(1u64, |acc, &d| {
                (d != 0).then(|| acc.checked_mul(d)).flatten()
            });
            let bytes = elements
                .and_then(|e| ty.encoded_bytes(e))
                .filter(|&b| b > 0)
                .and_then(|b| usize::try_from(b).ok());
            let block = ty.block().0 as u64;
            let (Some(bytes), true) = (bytes, dims[0].is_multiple_of(block)) else {
                return Err(format!("invalid GGUF tensor shape of {name}"));
            };
            let offset = c.u64()?;
            if index.insert(name, tensors.len()).is_some() {
                return Err(format!("duplicate GGUF tensor {name}"));
            }
            relative.push(offset);
            tensors.push(Tensor {
                name,
                dims,
                ty,
                offset: 0,
                bytes,
            });
        }

        let data_base = (c.at as u64).next_multiple_of(alignment);
        let data_base = usize::try_from(data_base)
            .ok()
            .filter(|&b| b <= file_len)
            .ok_or("GGUF data section lies outside the file")?;
        for (tensor, offset) in tensors.iter_mut().zip(relative) {
            let start = usize::try_from(offset)
                .ok()
                .filter(|_| offset % alignment == 0)
                .and_then(|o| o.checked_add(data_base))
                .filter(|&s| s <= file_len && tensor.bytes <= file_len - s)
                .ok_or_else(|| format!("tensor extent or alignment is invalid: {}", tensor.name))?;
            tensor.offset = start;
        }
        Ok(Self {
            version,
            alignment,
            metadata,
            tensors,
            index,
        })
    }

    pub fn metadata(&self, key: &str) -> Option<&Value<'a>> {
        self.metadata.get(key)
    }

    pub fn string(&self, key: &str) -> Option<&'a [u8]> {
        match self.metadata.get(key)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// An unsigned or signed integer that fits in `u32`.
    pub fn u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            Value::UInt(v) => u32::try_from(*v).ok(),
            Value::Int(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key)? {
            Value::UInt(v) => Some(*v),
            Value::Int(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            Value::Float(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key)? {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn strings(&self, key: &str) -> Option<&[&'a [u8]]> {
        match self.metadata.get(key)? {
            Value::StrArray(items) => Some(items),
            _ => None,
        }
    }

    pub fn tensors(&self) -> &[Tensor<'a>] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor<'a>> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }
}

/// Reads just the header of the file at `path`: the bytes to pass to
/// [`Gguf::parse_prefix`], and the file length.
pub fn read_header(path: &std::path::Path) -> Result<(Vec<u8>, usize), String> {
    use std::io::Read;
    let error = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut file = std::fs::File::open(path).map_err(error)?;
    let len = usize::try_from(file.metadata().map_err(error)?.len())
        .map_err(|_| "GGUF file is too large")?;
    let mut bytes = Vec::new();
    let mut want = 16usize << 20;
    loop {
        let target = want.min(len);
        (&mut file)
            .take((target - bytes.len()) as u64)
            .read_to_end(&mut bytes)
            .map_err(error)?;
        match Gguf::parse_prefix(&bytes, len).map(|_| ()) {
            Err(e) if e.contains("truncated") && target < len => want *= 2,
            Err(e) => return Err(format!("{}: {e}", path.display())),
            Ok(_) => return Ok((bytes, len)),
        }
    }
}

/// The files of a GGUF set, in shard order, given any one of them and its
/// parsed header: the file alone when it is not split, else every
/// `<name>-<i>-of-<n>.gguf` beside it (`GgufReader::OpenFile` and
/// `OpenSplitFileSet`). Check the parsed shards with [`check_split`].
pub fn split_paths(
    path: &std::path::Path,
    header: &Gguf<'_>,
) -> Result<Vec<std::path::PathBuf>, String> {
    let count = match header.metadata("split.count") {
        None => return Ok(vec![path.to_path_buf()]),
        Some(_) => match header.u32("split.count") {
            Some(count) if (1..=65535).contains(&count) => count,
            _ => return Err("Invalid GGUF split count".into()),
        },
    };
    if count == 1 {
        return Ok(vec![path.to_path_buf()]);
    }
    let invalid = || format!("Invalid split GGUF metadata in {}", path.display());
    match (header.u32("split.no"), header.u64("split.tensors.count")) {
        (Some(no), Some(_)) if no < count => {}
        _ => return Err(invalid()),
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    // `^(.*)-([0-9]+)-of-([0-9]+)\.gguf$`, the prefix as long as possible.
    let parts = filename.strip_suffix(".gguf").and_then(|stem| {
        let (left, total) = stem.rsplit_once("-of-")?;
        let (prefix, index) = left.rsplit_once('-')?;
        (digits(index) && digits(total)).then_some((prefix, index.len(), total.len()))
    });
    let Some((prefix, index_width, count_width)) = parts else {
        return Err(format!(
            "Split GGUF filename does not match <name>-00001-of-00002.gguf: {filename}"
        ));
    };
    let directory = path.parent().unwrap_or(std::path::Path::new(""));
    Ok((0..count)
        .map(|i| {
            directory.join(format!(
                "{prefix}-{:0index_width$}-of-{count:0count_width$}.gguf",
                i + 1
            ))
        })
        .collect())
}

/// Checks that parsed shards (in order) form one split set: each names its
/// place and the shard count, all share the version and architecture, and
/// the tensors add up to `split.tensors.count` without a name twice.
pub fn check_split(shards: &[Gguf<'_>]) -> Result<(), String> {
    let Some(first) = shards.first() else {
        return Err("empty GGUF shard set".into());
    };
    if shards.len() == 1 && first.metadata("split.count").is_none() {
        return Ok(());
    }
    let architecture = first.string("general.architecture");
    for (i, shard) in shards.iter().enumerate() {
        let architecture_ok = shard
            .string("general.architecture")
            .is_none_or(|a| Some(a) == architecture);
        if shard.u32("split.no") != u32::try_from(i).ok()
            || shard.u32("split.count") != u32::try_from(shards.len()).ok()
            || shard.version != first.version
            || !architecture_ok
        {
            return Err(format!(
                "Inconsistent split GGUF shard metadata: shard {}",
                i + 1
            ));
        }
    }
    let total: usize = shards.iter().map(|s| s.tensors().len()).sum();
    if first.u64("split.tensors.count") != u64::try_from(total).ok() {
        return Err("Split GGUF tensor count mismatch".into());
    }
    let mut names = std::collections::HashSet::new();
    for tensor in shards.iter().flat_map(|s| s.tensors()) {
        if !names.insert(tensor.name) {
            return Err(format!(
                "Duplicate tensor across split GGUF shards: {}",
                tensor.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Builds a GGUF image: metadata as (key, type, encoded value bytes),
    /// tensors as (name, dims, type id, relative offset), then `payload`
    /// bytes of data.
    pub(crate) fn image(
        metadata: &[(&str, u32, Vec<u8>)],
        tensors: &[(&str, &[u64], u32, u64)],
        payload: usize,
    ) -> Vec<u8> {
        let string = |s: &[u8]| [&(s.len() as u64).to_le_bytes()[..], s].concat();
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, ty, value) in metadata {
            b.extend(string(key.as_bytes()));
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(value);
        }
        for (name, dims, ty, offset) in tensors {
            b.extend(string(name.as_bytes()));
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
        }
        b.resize(b.len().next_multiple_of(32) + payload, 0);
        b
    }

    pub(crate) fn string_array(items: &[&str]) -> Vec<u8> {
        let mut v = 8u32.to_le_bytes().to_vec();
        v.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for item in items {
            v.extend_from_slice(&(item.len() as u64).to_le_bytes());
            v.extend_from_slice(item.as_bytes());
        }
        v
    }

    #[test]
    fn metadata_and_tensors_are_indexed() {
        let bytes = image(
            &[
                (
                    "general.architecture",
                    8,
                    [&6u64.to_le_bytes()[..], b"qwen35"].concat(),
                ),
                ("qwen35.block_count", 4, 64u32.to_le_bytes().to_vec()),
                ("qwen35.eps", 6, 1e-6f32.to_bits().to_le_bytes().to_vec()),
                ("neg", 5, (-3i32).to_le_bytes().to_vec()),
                ("flag", 7, vec![1]),
                ("tokens", 9, string_array(&["a", "bc"])),
                (
                    "flags",
                    9,
                    [&7u32.to_le_bytes()[..], &3u64.to_le_bytes(), &[0, 1, 0]].concat(),
                ),
            ],
            &[("w", &[256, 2], 12, 0), ("n", &[5], 0, 320)],
            320 + 20,
        );
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.string("general.architecture"), Some(&b"qwen35"[..]));
        assert_eq!(g.u32("qwen35.block_count"), Some(64));
        assert_eq!(g.f32("qwen35.eps"), Some(1e-6));
        assert_eq!(g.u32("neg"), None);
        assert_eq!(g.bool("flag"), Some(true));
        assert_eq!(g.strings("tokens").unwrap(), [&b"a"[..], b"bc"]);
        assert_eq!(g.metadata("flags"), Some(&Value::UIntArray(vec![0, 1, 0])));
        let w = g.tensor("w").unwrap();
        assert_eq!((w.ty, w.bytes, w.elements()), (GgmlType::Q4K, 288, 512));
        assert_eq!(w.offset % 32, 0);
        assert_eq!(g.tensor("n").unwrap().offset, w.offset + 320);
    }

    #[test]
    fn narrow_integer_arrays_widen() {
        let array = |item: u32, values: &[u8], len: u64| {
            [&item.to_le_bytes()[..], &len.to_le_bytes(), values].concat()
        };
        let bytes = image(
            &[
                ("u8", 9, array(0, &[0, 255, 7], 3)),
                ("u16", 9, array(2, &[0xff, 0xff, 2, 0], 2)),
                ("i8", 9, array(1, &[0x80, 5], 2)),
                ("i16", 9, array(3, &[0, 0x80, 0xff, 0x7f], 2)),
            ],
            &[],
            0,
        );
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.metadata("u8"), Some(&Value::UIntArray(vec![0, 255, 7])));
        assert_eq!(g.metadata("u16"), Some(&Value::UIntArray(vec![65535, 2])));
        assert_eq!(g.metadata("i8"), Some(&Value::IntArray(vec![-128, 5])));
        assert_eq!(
            g.metadata("i16"),
            Some(&Value::IntArray(vec![-32768, 32767]))
        );
    }

    #[test]
    fn malformed_files_are_rejected() {
        let good = || image(&[], &[("w", &[32], 8, 0)], 34);
        assert!(Gguf::parse(&good()).is_ok());
        let mut bad = good();
        bad[0] = b'X';
        assert!(Gguf::parse(&bad).unwrap_err().contains("magic"));
        let partial = image(&[], &[("w", &[31], 8, 0)], 34);
        assert!(Gguf::parse(&partial).unwrap_err().contains("shape"));
        let outside = image(&[], &[("w", &[32], 8, 0)], 20);
        assert!(Gguf::parse(&outside).unwrap_err().contains("extent"));
        let unaligned = image(&[], &[("w", &[32], 8, 8)], 64);
        assert!(Gguf::parse(&unaligned).unwrap_err().contains("alignment"));
        let unknown = image(&[], &[("w", &[32], 4, 0)], 64);
        assert!(Gguf::parse(&unknown).unwrap_err().contains("storage type"));
        let duplicate = image(&[("k", 0, vec![1]), ("k", 0, vec![2])], &[], 0);
        assert!(Gguf::parse(&duplicate).unwrap_err().contains("duplicate"));
        let alignment = image(
            &[("general.alignment", 4, 24u32.to_le_bytes().to_vec())],
            &[],
            0,
        );
        assert!(Gguf::parse(&alignment).unwrap_err().contains("alignment"));
        let flag = [&7u32.to_le_bytes()[..], &1u64.to_le_bytes(), &[2]].concat();
        let boolean = image(&[("flags", 9, flag)], &[], 0);
        assert!(Gguf::parse(&boolean).unwrap_err().contains("boolean"));
        let truncated = &good()[..40];
        assert!(Gguf::parse(truncated).is_err());
    }

    #[test]
    fn split_sets_name_their_shards_and_must_agree() {
        let shard = |no: u32, count: u32, total: u64, tensor: &str| {
            image(
                &[
                    ("split.no", 4, no.to_le_bytes().to_vec()),
                    ("split.count", 4, count.to_le_bytes().to_vec()),
                    ("split.tensors.count", 10, total.to_le_bytes().to_vec()),
                ],
                &[(tensor, &[32], 8, 0)],
                34,
            )
        };
        let single = image(&[], &[("w", &[32], 8, 0)], 34);
        let path = std::path::Path::new("/m/model.gguf");
        let g = Gguf::parse(&single).unwrap();
        assert_eq!(split_paths(path, &g).unwrap(), [path]);
        check_split(&[g]).unwrap();

        let second = shard(1, 3, 3, "b");
        let g = Gguf::parse(&second).unwrap();
        let paths = split_paths(std::path::Path::new("/m/a-b-00002-of-00003.gguf"), &g).unwrap();
        assert_eq!(
            paths,
            [
                "/m/a-b-00001-of-00003.gguf",
                "/m/a-b-00002-of-00003.gguf",
                "/m/a-b-00003-of-00003.gguf"
            ]
            .map(std::path::PathBuf::from)
        );
        assert!(
            split_paths(std::path::Path::new("/m/model.gguf"), &g)
                .unwrap_err()
                .contains("does not match")
        );

        let images = [
            shard(0, 3, 3, "a"),
            shard(1, 3, 3, "b"),
            shard(2, 3, 3, "c"),
        ];
        let parsed: Vec<_> = images.iter().map(|b| Gguf::parse(b).unwrap()).collect();
        check_split(&parsed).unwrap();
        let swapped: Vec<_> = [&images[1], &images[0], &images[2]]
            .iter()
            .map(|b| Gguf::parse(b).unwrap())
            .collect();
        assert!(check_split(&swapped).unwrap_err().contains("Inconsistent"));
        let duplicate = [shard(0, 2, 2, "a"), shard(1, 2, 2, "a")];
        let parsed: Vec<_> = duplicate.iter().map(|b| Gguf::parse(b).unwrap()).collect();
        assert!(check_split(&parsed).unwrap_err().contains("Duplicate"));
        let short = [shard(0, 2, 3, "a"), shard(1, 2, 3, "b")];
        let parsed: Vec<_> = short.iter().map(|b| Gguf::parse(b).unwrap()).collect();
        assert!(check_split(&parsed).unwrap_err().contains("count mismatch"));
    }
}
