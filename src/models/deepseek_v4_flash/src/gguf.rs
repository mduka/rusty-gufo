//! The GGUF reader of the DS4 engine (`model_data.cpp`): GGUF v3 only,
//! metadata values left in the mapping and decoded on demand, the tensor
//! directory with absolute offsets. Its checks and messages are those of
//! the C++ reader; tensors of unknown types only warn and have no size.

use std::path::Path;
use std::sync::Arc;

use gufo_hip::MappedFile;

use crate::shape::MAX_DIMS;

const MAGIC: u32 = 0x4655_4747;

pub(crate) const VALUE_UINT8: u32 = 0;
pub(crate) const VALUE_INT8: u32 = 1;
pub(crate) const VALUE_UINT16: u32 = 2;
pub(crate) const VALUE_INT16: u32 = 3;
pub(crate) const VALUE_UINT32: u32 = 4;
pub(crate) const VALUE_INT32: u32 = 5;
pub(crate) const VALUE_FLOAT32: u32 = 6;
pub(crate) const VALUE_BOOL: u32 = 7;
pub(crate) const VALUE_STRING: u32 = 8;
pub(crate) const VALUE_ARRAY: u32 = 9;
pub(crate) const VALUE_UINT64: u32 = 10;
pub(crate) const VALUE_INT64: u32 = 11;
pub(crate) const VALUE_FLOAT64: u32 = 12;

/// Name, elements per block and bytes per block of the GGML types the
/// reader knows (`gguf_types`).
const TYPES: [Option<(&str, u32, u32)>; 31] = {
    let mut types = [None; 31];
    types[0] = Some(("f32", 1, 4));
    types[1] = Some(("f16", 1, 2));
    types[2] = Some(("q4_0", 32, 18));
    types[3] = Some(("q4_1", 32, 20));
    types[6] = Some(("q5_0", 32, 22));
    types[7] = Some(("q5_1", 32, 24));
    types[8] = Some(("q8_0", 32, 34));
    types[9] = Some(("q8_1", 32, 40));
    types[10] = Some(("q2_k", 256, 84));
    types[11] = Some(("q3_k", 256, 110));
    types[12] = Some(("q4_k", 256, 144));
    types[13] = Some(("q5_k", 256, 176));
    types[14] = Some(("q6_k", 256, 210));
    types[15] = Some(("q8_k", 256, 292));
    types[16] = Some(("iq2_xxs", 256, 66));
    types[17] = Some(("iq2_xs", 256, 74));
    types[18] = Some(("iq3_xxs", 256, 98));
    types[19] = Some(("iq1_s", 256, 110));
    types[20] = Some(("iq4_nl", 256, 50));
    types[21] = Some(("iq3_s", 256, 110));
    types[22] = Some(("iq2_s", 256, 82));
    types[23] = Some(("iq4_xs", 256, 136));
    types[24] = Some(("i8", 1, 1));
    types[25] = Some(("i16", 1, 2));
    types[26] = Some(("i32", 1, 4));
    types[27] = Some(("i64", 1, 8));
    types[28] = Some(("f64", 1, 8));
    types[29] = Some(("iq1_m", 256, 56));
    types[30] = Some(("bf16", 1, 2));
    types
};

/// `ds4_tensor_type_name`.
pub fn type_name(ty: u32) -> &'static str {
    TYPES
        .get(ty as usize)
        .copied()
        .flatten()
        .map_or("unknown", |(name, _, _)| name)
}

fn tensor_bytes(ty: u32, elements: u64) -> Option<u64> {
    let (_, block_elems, block_bytes) = TYPES.get(ty as usize).copied().flatten()?;
    let blocks = elements.div_ceil(u64::from(block_elems));
    blocks.checked_mul(u64::from(block_bytes))
}

/// A byte range of the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Span {
    pub pos: u64,
    pub len: u64,
}

struct Kv {
    key: Span,
    ty: u32,
    value_pos: u64,
}

/// An entry of the tensor directory (`ds4_tensor`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorInfo {
    pub(crate) name: Span,
    pub ndim: u32,
    pub dim: [u64; MAX_DIMS],
    pub ty: u32,
    pub rel_offset: u64,
    /// Offset of the data in the file.
    pub offset: u64,
    pub elements: u64,
    /// Size of the data; 0 for an unknown type.
    pub bytes: u64,
}

/// A mapped GGUF (`ds4_model`).
pub struct Model {
    map: Arc<MappedFile>,
    pub version: u32,
    pub alignment: u64,
    pub tensor_data_pos: u64,
    pub max_tensor_bytes: u64,
    kv: Vec<Kv>,
    tensors: Vec<TensorInfo>,
}

/// The reader's failure: the message C++ throws.
pub type Error = String;

/// Bounds-checked reads of the mapping (`ds4_cursor`), keeping the first
/// error with its position.
struct Cursor<'a> {
    base: &'a [u8],
    pos: u64,
    error: Option<String>,
}

impl<'a> Cursor<'a> {
    fn at(base: &'a [u8], pos: u64) -> Self {
        Self {
            base,
            pos,
            error: None,
        }
    }

    fn fail(&mut self, message: &str) {
        if self.error.is_none() {
            self.error = Some(format!("{message} at byte {}", self.pos));
        }
    }

    fn error(&self) -> String {
        self.error.clone().unwrap_or_default()
    }

    fn has(&mut self, n: u64) -> bool {
        let size = self.base.len() as u64;
        if n > size || self.pos > size - n {
            self.fail("truncated GGUF file");
            return false;
        }
        true
    }

    fn bytes(&mut self, n: u64) -> Option<&'a [u8]> {
        if !self.has(n) {
            return None;
        }
        let start = self.pos as usize;
        self.pos += n;
        Some(&self.base[start..start + n as usize])
    }

    fn skip(&mut self, n: u64) -> bool {
        if !self.has(n) {
            return false;
        }
        self.pos += n;
        true
    }

    fn u32(&mut self) -> Option<u32> {
        self.bytes(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Option<u64> {
        self.bytes(8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn i32(&mut self) -> Option<i32> {
        self.u32().map(|v| v as i32)
    }

    fn f32(&mut self) -> Option<f32> {
        self.u32().map(f32::from_bits)
    }

    fn f64(&mut self) -> Option<f64> {
        self.u64().map(f64::from_bits)
    }

    fn string(&mut self) -> Option<Span> {
        let len = self.u64()?;
        if !self.has(len) {
            return None;
        }
        let span = Span { pos: self.pos, len };
        self.pos += len;
        Some(span)
    }

    fn skip_value(&mut self, ty: u32, depth: u32) -> bool {
        if depth > 8 {
            self.fail("metadata array nesting is too deep");
            return false;
        }
        let scalar = scalar_size(ty);
        if scalar != 0 {
            return self.skip(scalar);
        }
        if ty == VALUE_STRING {
            return self.string().is_some();
        }
        if ty == VALUE_ARRAY {
            let (Some(item), Some(len)) = (self.u32(), self.u64()) else {
                return false;
            };
            let item_size = scalar_size(item);
            if item_size != 0 {
                return match len.checked_mul(item_size) {
                    Some(bytes) => self.skip(bytes),
                    None => {
                        self.fail("metadata array is too large");
                        false
                    }
                };
            }
            for _ in 0..len {
                if !self.skip_value(item, depth + 1) {
                    return false;
                }
            }
            return true;
        }
        self.fail("unknown GGUF metadata type");
        false
    }
}

fn scalar_size(ty: u32) -> u64 {
    match ty {
        VALUE_UINT8 | VALUE_INT8 | VALUE_BOOL => 1,
        VALUE_UINT16 | VALUE_INT16 => 2,
        VALUE_UINT32 | VALUE_INT32 | VALUE_FLOAT32 => 4,
        VALUE_UINT64 | VALUE_INT64 | VALUE_FLOAT64 => 8,
        _ => 0,
    }
}

/// An array value: its item type, length and the position of its items.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArrayRef {
    pub ty: u32,
    pub len: u64,
    pub data_pos: u64,
}

impl Model {
    /// Maps and parses `path` (`model_open`).
    pub fn open(path: &Path) -> Result<Self, Error> {
        let map = MappedFile::open(path).map_err(|error| {
            eprintln!("ds4: cannot open model '{}': {error}", path.display());
            "incompatible or invalid DeepSeek artifact".to_owned()
        })?;
        Self::parse(map)
    }

    pub(crate) fn parse(map: Arc<MappedFile>) -> Result<Self, Error> {
        let bytes = map.bytes();
        if bytes.len() < 32 {
            return Err("model file is too small to be GGUF".into());
        }
        let mut c = Cursor::at(bytes, 0);
        let magic = c.u32().ok_or_else(|| c.error())?;
        if magic != MAGIC {
            return Err("model is not a GGUF file".into());
        }
        let version = c.u32().ok_or_else(|| c.error())?;
        let n_tensors = c.u64().ok_or_else(|| c.error())?;
        let n_kv = c.u64().ok_or_else(|| c.error())?;
        if version != 3 {
            return Err("only GGUF v3 is supported".into());
        }

        // parse_metadata
        let mut alignment = 32;
        let mut kv = Vec::new();
        for _ in 0..n_kv {
            let key = c.string().ok_or_else(|| c.error())?;
            let ty = c.u32().ok_or_else(|| c.error())?;
            let value_pos = c.pos;
            if span_bytes(bytes, key) == b"general.alignment"
                && ty == VALUE_UINT32
                && let Some(value) = Cursor::at(bytes, value_pos).u32().filter(|&a| a != 0)
            {
                alignment = u64::from(value);
            }
            if !c.skip_value(ty, 0) {
                return Err(c.error());
            }
            kv.push(Kv { key, ty, value_pos });
        }

        // parse_tensors
        let mut tensors = Vec::new();
        for _ in 0..n_tensors {
            let name = c.string().ok_or_else(|| c.error())?;
            let ndim = c.u32().ok_or_else(|| c.error())?;
            if ndim == 0 || ndim as usize > MAX_DIMS {
                return Err("tensor has an unsupported number of dimensions".into());
            }
            let mut dim = [0u64; MAX_DIMS];
            let mut elements: u64 = 1;
            for d in dim.iter_mut().take(ndim as usize) {
                *d = c.u64().ok_or_else(|| c.error())?;
                if *d != 0 && elements > u64::MAX / *d {
                    return Err("tensor element count overflow".into());
                }
                elements *= *d;
            }
            let ty = c.u32().ok_or_else(|| c.error())?;
            let rel_offset = c.u64().ok_or_else(|| c.error())?;
            let bytes_len = tensor_bytes(ty, elements).unwrap_or_else(|| {
                eprintln!(
                    "ds4: warning: tensor {} has unsupported GGUF type {ty}",
                    String::from_utf8_lossy(span_bytes(bytes, name))
                );
                0
            });
            tensors.push(TensorInfo {
                name,
                ndim,
                dim,
                ty,
                rel_offset,
                offset: 0,
                elements,
                bytes: bytes_len,
            });
        }
        let tensor_data_pos = align_up(c.pos, alignment);
        let size = bytes.len() as u64;
        let mut max_tensor_bytes = 0;
        for t in &mut tensors {
            if t.rel_offset > u64::MAX - tensor_data_pos {
                return Err("tensor offset overflow".into());
            }
            t.offset = tensor_data_pos + t.rel_offset;
            if t.bytes != 0 && (t.offset > size || t.bytes > size - t.offset) {
                return Err("tensor points outside GGUF file".into());
            }
            max_tensor_bytes = max_tensor_bytes.max(t.bytes);
        }
        Ok(Self {
            map,
            version,
            alignment,
            tensor_data_pos,
            max_tensor_bytes,
            kv,
            tensors,
        })
    }

    pub fn map(&self) -> &Arc<MappedFile> {
        &self.map
    }

    pub fn size(&self) -> u64 {
        self.map.len() as u64
    }

    pub(crate) fn bytes_of(&self, span: Span) -> &[u8] {
        span_bytes(self.map.bytes(), span)
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor_name(&self, tensor: &TensorInfo) -> &[u8] {
        self.bytes_of(tensor.name)
    }

    /// The first tensor named `name` (`model_find_tensor`).
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors
            .iter()
            .find(|t| self.bytes_of(t.name) == name.as_bytes())
    }

    fn find_kv(&self, key: &str) -> Option<&Kv> {
        self.kv
            .iter()
            .find(|kv| self.bytes_of(kv.key) == key.as_bytes())
    }

    fn cursor(&self, pos: u64) -> Cursor<'_> {
        Cursor::at(self.map.bytes(), pos)
    }

    /// `model_get_u32`: a UINT32 value.
    pub(crate) fn get_u32(&self, key: &str) -> Option<u32> {
        let kv = self.find_kv(key).filter(|kv| kv.ty == VALUE_UINT32)?;
        self.cursor(kv.value_pos).u32()
    }

    /// `model_get_u64`: a UINT64 or UINT32 value.
    pub(crate) fn get_u64(&self, key: &str) -> Option<u64> {
        let kv = self.find_kv(key)?;
        let mut c = self.cursor(kv.value_pos);
        match kv.ty {
            VALUE_UINT64 => c.u64(),
            VALUE_UINT32 => c.u32().map(u64::from),
            _ => None,
        }
    }

    /// `model_get_f32`: a FLOAT32, FLOAT64, UINT32 or INT32 value as `f32`.
    pub(crate) fn get_f32(&self, key: &str) -> Option<f32> {
        let kv = self.find_kv(key)?;
        let mut c = self.cursor(kv.value_pos);
        match kv.ty {
            VALUE_FLOAT32 => c.f32(),
            VALUE_FLOAT64 => c.f64().map(|v| v as f32),
            VALUE_UINT32 => c.u32().map(|v| v as f32),
            VALUE_INT32 => c.i32().map(|v| v as f32),
            _ => None,
        }
    }

    /// `model_get_bool`.
    pub(crate) fn get_bool(&self, key: &str) -> Option<bool> {
        let kv = self.find_kv(key).filter(|kv| kv.ty == VALUE_BOOL)?;
        self.cursor(kv.value_pos).bytes(1).map(|b| b[0] != 0)
    }

    /// `model_get_array`.
    pub(crate) fn get_array(&self, key: &str) -> Option<ArrayRef> {
        let kv = self.find_kv(key).filter(|kv| kv.ty == VALUE_ARRAY)?;
        let mut c = self.cursor(kv.value_pos);
        let ty = c.u32()?;
        let len = c.u64()?;
        Some(ArrayRef {
            ty,
            len,
            data_pos: c.pos,
        })
    }

    /// A STRING value (as `dspark_validate_metadata` reads
    /// `general.architecture`).
    pub(crate) fn get_string(&self, key: &str) -> Option<&[u8]> {
        let kv = self.find_kv(key).filter(|kv| kv.ty == VALUE_STRING)?;
        let span = self.cursor(kv.value_pos).string()?;
        Some(self.bytes_of(span))
    }

    /// The strings of a STRING array, in order (`ds4_model_string_array_*`);
    /// `None` for a missing or differently typed array or a truncated item.
    pub(crate) fn string_array(&self, key: &str) -> Option<StringArray<'_>> {
        let array = self.get_array(key).filter(|a| a.ty == VALUE_STRING)?;
        Some(StringArray {
            cursor: self.cursor(array.data_pos),
            remaining: array.len,
        })
    }

    /// The `n` items of a UINT32/INT32 or FLOAT32/FLOAT64 array, read with
    /// the cursor's messages.
    pub(crate) fn array_cursor(&self, array: ArrayRef) -> ArrayCursor<'_> {
        ArrayCursor(self.cursor(array.data_pos))
    }
}

/// Item reads of a numeric metadata array.
pub(crate) struct ArrayCursor<'a>(Cursor<'a>);

impl ArrayCursor<'_> {
    pub fn u32(&mut self) -> Result<u32, Error> {
        self.0.u32().ok_or_else(|| self.0.error())
    }
    pub fn i32(&mut self) -> Result<i32, Error> {
        self.0.i32().ok_or_else(|| self.0.error())
    }
    pub fn f32(&mut self) -> Result<f32, Error> {
        self.0.f32().ok_or_else(|| self.0.error())
    }
    pub fn f64(&mut self) -> Result<f64, Error> {
        self.0.f64().ok_or_else(|| self.0.error())
    }
}

/// The items of a STRING array.
pub(crate) struct StringArray<'a> {
    cursor: Cursor<'a>,
    remaining: u64,
}

impl<'a> StringArray<'a> {
    pub fn remaining(&self) -> u64 {
        self.remaining
    }
}

impl<'a> Iterator for StringArray<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.remaining == 0 {
            return None;
        }
        let span = self.cursor.string()?;
        self.remaining -= 1;
        Some(span_bytes(self.cursor.base, span))
    }
}

fn span_bytes(bytes: &[u8], span: Span) -> &[u8] {
    &bytes[span.pos as usize..(span.pos + span.len) as usize]
}

/// `ds4_align_up`.
pub fn align_up(value: u64, alignment: u64) -> u64 {
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + alignment - rem
    }
}
