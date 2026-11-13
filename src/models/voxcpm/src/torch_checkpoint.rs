//! A read-only mapping of a state dict saved by `torch.save` in the zip
//! format (PyTorch >= 1.6), such as VoxCPM2's `audiovae.pth`
//! (`torch_checkpoint.hpp`, `torch_checkpoint.cpp`). The pickle is
//! interpreted, never executed: only the opcodes and globals a tensor state
//! dict needs are accepted (OrderedDict, `torch._utils._rebuild_tensor_v2`
//! and the typed storages), every tensor must be contiguous and lie inside
//! its stored, uncompressed storage. A top-level dict with a "state_dict"
//! entry is unwrapped, as VoxCPM's loader does.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use gufo_hip::MappedFile;

use crate::safetensors::{DType, Tensor};

type Result<T> = std::result::Result<T, String>;

fn require(condition: bool, message: impl FnOnce() -> String) -> Result<()> {
    if condition { Ok(()) } else { Err(message()) }
}

// ---------------------------------------------------------------- zip ---

#[derive(Clone, Copy)]
struct Bytes<'a> {
    data: &'a [u8],
}

impl Bytes<'_> {
    /// A little-endian integer of `width` bytes at `offset`.
    fn read(&self, offset: u64, width: usize) -> Result<u64> {
        let end = offset.checked_add(width as u64);
        require(end.is_some_and(|end| end <= self.data.len() as u64), || {
            "zip structure runs past the file".into()
        })?;
        let at = offset as usize;
        Ok(self.data[at..at + width]
            .iter()
            .rev()
            .fold(0u64, |value, &byte| (value << 8) | u64::from(byte)))
    }
}

#[derive(Clone, Copy)]
struct Entry {
    /// Of the data in the file.
    offset: u64,
    size: u64,
}

/// Stored (uncompressed) entries of the zip archive by name.
fn read_zip(file: Bytes<'_>) -> Result<BTreeMap<String, Entry>> {
    const END_SIZE: u64 = 22;
    let size = file.data.len() as u64;
    require(size >= END_SIZE, || "not a zip archive".into())?;
    let mut end = size - END_SIZE;
    let lowest = if size > END_SIZE + 0xFFFF {
        size - END_SIZE - 0xFFFF
    } else {
        0
    };
    while file.read(end, 4)? != 0x0605_4b50 {
        require(end > lowest, || "not a zip archive".into())?;
        end -= 1;
    }
    let mut count = file.read(end + 10, 2)?;
    let mut directory = file.read(end + 16, 4)?;
    if count == 0xFFFF || directory == 0xFFFF_FFFF {
        require(end >= 20 && file.read(end - 20, 4)? == 0x0706_4b50, || {
            "zip64 locator missing".into()
        })?;
        let record = file.read(end - 20 + 8, 8)?;
        require(file.read(record, 4)? == 0x0606_4b50, || {
            "zip64 record missing".into()
        })?;
        count = file.read(record + 32, 8)?;
        directory = file.read(record + 48, 8)?;
    }
    let mut entries = BTreeMap::new();
    let mut at = directory;
    for _ in 0..count {
        require(file.read(at, 4)? == 0x0201_4b50, || {
            "bad zip directory entry".into()
        })?;
        let method = file.read(at + 10, 2)?;
        let mut compressed = file.read(at + 20, 4)?;
        let mut stored = file.read(at + 24, 4)?;
        let name_size = file.read(at + 28, 2)?;
        let extra_size = file.read(at + 30, 2)?;
        let comment_size = file.read(at + 32, 2)?;
        let mut local = file.read(at + 42, 4)?;
        require(at + 46 + name_size <= size, || {
            "zip name runs past the file".into()
        })?;
        let name_start = (at + 46) as usize;
        let name = String::from_utf8_lossy(&file.data[name_start..name_start + name_size as usize])
            .into_owned();
        // Zip64 extra field: the sizes and offset that overflowed, in order.
        let mut extra = at + 46 + name_size;
        while extra + 4 <= at + 46 + name_size + extra_size {
            let id = file.read(extra, 2)?;
            let length = file.read(extra + 2, 2)?;
            if id == 0x0001 {
                let mut field = extra + 4;
                if stored == 0xFFFF_FFFF {
                    stored = file.read(field, 8)?;
                    field += 8;
                }
                if compressed == 0xFFFF_FFFF {
                    compressed = file.read(field, 8)?;
                    field += 8;
                }
                if local == 0xFFFF_FFFF {
                    local = file.read(field, 8)?;
                }
            }
            extra += 4 + length;
        }
        require(method == 0 && compressed == stored, || {
            format!("zip entry {name} is compressed")
        })?;
        require(file.read(local, 4)? == 0x0403_4b50, || {
            "bad zip local header".into()
        })?;
        let data = local + 30 + file.read(local + 26, 2)? + file.read(local + 28, 2)?;
        require(data <= size && stored <= size - data, || {
            format!("zip entry {name} runs past the file")
        })?;
        entries.insert(
            name,
            Entry {
                offset: data,
                size: stored,
            },
        );
        at += 46 + name_size + extra_size + comment_size;
    }
    Ok(entries)
}

// ------------------------------------------------------------- pickle ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Kind {
    #[default]
    None,
    Bool,
    Int,
    Float,
    String,
    Tuple,
    List,
    Dict,
    Global,
    Storage,
    Tensor,
}

type Ref = Rc<RefCell<Object>>;

#[derive(Default)]
struct Object {
    kind: Kind,
    integer: i64,
    /// A string, "module name" of a global, a storage key.
    text: String,
    items: Vec<Ref>,
    members: Vec<(Ref, Ref)>,
    /// Storage: element type and count. Tensor: storage, offset, size,
    /// stride.
    dtype: Option<DType>,
    count: u64,
    storage: Option<Ref>,
    size: Vec<i64>,
    stride: Vec<i64>,
}

fn make(kind: Kind) -> Ref {
    Rc::new(RefCell::new(Object {
        kind,
        ..Object::default()
    }))
}

fn make_int(value: i64) -> Ref {
    let object = make(Kind::Int);
    object.borrow_mut().integer = value;
    object
}

fn kind(object: &Ref) -> Kind {
    object.borrow().kind
}

fn storage_type(global: &str) -> Option<DType> {
    Some(match global {
        "torch FloatStorage" => DType::F32,
        "torch BFloat16Storage" => DType::Bf16,
        "torch HalfStorage" => DType::F16,
        "torch LongStorage" => DType::I64,
        "torch IntStorage" => DType::I32,
        "torch ByteStorage" => DType::U8,
        _ => return None,
    })
}

fn integers(tuple: &Ref, what: &str) -> Result<Vec<i64>> {
    let tuple = tuple.borrow();
    require(tuple.kind == Kind::Tuple, || {
        format!("tensor {what} is not a tuple")
    })?;
    tuple
        .items
        .iter()
        .map(|item| {
            let item = item.borrow();
            if item.kind == Kind::Int && item.integer >= 0 {
                Ok(item.integer)
            } else {
                Err(format!("tensor {what} holds a non-integer"))
            }
        })
        .collect()
}

/// A pickle interpreter for protocol-2 tensor state dicts.
struct Unpickler<'a> {
    bytes: Bytes<'a>,
    position: usize,
    stack: Vec<Ref>,
    marks: Vec<usize>,
    memo: HashMap<u32, Ref>,
}

impl<'a> Unpickler<'a> {
    fn new(bytes: Bytes<'a>) -> Self {
        Self {
            bytes,
            position: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn load(&mut self) -> Result<Ref> {
        loop {
            let op = self.byte()?;
            match op {
                0x80 => {
                    // PROTO
                    require(self.byte()? <= 5, || "unsupported pickle protocol".into())?;
                }
                b'.' => {
                    // STOP
                    require(self.stack.len() == 1, || {
                        "pickle stack is unbalanced".into()
                    })?;
                    return Ok(self.stack[0].clone());
                }
                b'(' => self.marks.push(self.stack.len()), // MARK
                b'}' => self.push(make(Kind::Dict)),       // EMPTY_DICT
                b']' => self.push(make(Kind::List)),       // EMPTY_LIST
                b')' => self.push(make(Kind::Tuple)),      // EMPTY_TUPLE
                b'N' => self.push(make(Kind::None)),       // NONE
                0x88 | 0x89 => {
                    // NEWTRUE, NEWFALSE
                    let value = make(Kind::Bool);
                    value.borrow_mut().integer = i64::from(op == 0x88);
                    self.push(value);
                }
                b'J' => {
                    // BININT
                    let value = self.read(4)? as u32 as i32;
                    self.push(make_int(i64::from(value)));
                }
                b'K' => {
                    // BININT1
                    let value = self.read(1)?;
                    self.push(make_int(value as i64));
                }
                b'M' => {
                    // BININT2
                    let value = self.read(2)?;
                    self.push(make_int(value as i64));
                }
                0x8a => {
                    // LONG1
                    let length = usize::from(self.byte()?);
                    require(length <= 8, || "pickle integer is too long".into())?;
                    let mut value = self.read(length)?;
                    if length > 0 && length < 8 && (value >> (8 * length - 1)) != 0 {
                        value |= !0u64 << (8 * length);
                    }
                    self.push(make_int(value as i64));
                }
                b'G' => {
                    // BINFLOAT (big endian); a state dict uses no float's value.
                    self.read(8)?;
                    self.push(make(Kind::Float));
                }
                b'X' => {
                    // BINUNICODE
                    let length = self.read(4)?;
                    self.push_string(length)?;
                }
                0x8c => {
                    // SHORT_BINUNICODE
                    let length = u64::from(self.byte()?);
                    self.push_string(length)?;
                }
                b'q' => {
                    // BINPUT
                    let index = self.read(1)? as u32;
                    let top = self.top()?;
                    self.memo.insert(index, top);
                }
                b'r' => {
                    // LONG_BINPUT
                    let index = self.read(4)? as u32;
                    let top = self.top()?;
                    self.memo.insert(index, top);
                }
                b'h' => {
                    // BINGET
                    let index = self.read(1)? as u32;
                    let value = self.memo_get(index)?;
                    self.push(value);
                }
                b'j' => {
                    // LONG_BINGET
                    let index = self.read(4)? as u32;
                    let value = self.memo_get(index)?;
                    self.push(value);
                }
                0x85..=0x87 => {
                    // TUPLE1, TUPLE2, TUPLE3
                    let count = usize::from(op - 0x84);
                    require(self.stack.len() >= count, || {
                        "pickle stack underflow".into()
                    })?;
                    let tuple = make(Kind::Tuple);
                    let items = self.stack.split_off(self.stack.len() - count);
                    tuple.borrow_mut().items = items;
                    self.push(tuple);
                }
                b't' => {
                    // TUPLE
                    let tuple = make(Kind::Tuple);
                    let items = self.pop_mark()?;
                    tuple.borrow_mut().items = items;
                    self.push(tuple);
                }
                b'a' => {
                    // APPEND
                    let value = self.pop()?;
                    let list = self.top()?;
                    require(kind(&list) == Kind::List, || "APPEND to a non-list".into())?;
                    list.try_borrow_mut()
                        .map_err(|_| "APPEND to itself".to_owned())?
                        .items
                        .push(value);
                }
                b'e' => {
                    // APPENDS
                    let values = self.pop_mark()?;
                    let list = self.top()?;
                    require(kind(&list) == Kind::List, || "APPENDS to a non-list".into())?;
                    list.try_borrow_mut()
                        .map_err(|_| "APPENDS to itself".to_owned())?
                        .items
                        .extend(values);
                }
                b's' => {
                    // SETITEM
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let dict = self.top()?;
                    set_item(&dict, key, value)?;
                }
                b'u' => {
                    // SETITEMS
                    let values = self.pop_mark()?;
                    require(values.len().is_multiple_of(2), || {
                        "SETITEMS with an odd count".into()
                    })?;
                    let dict = self.top()?;
                    for pair in values.as_chunks::<2>().0 {
                        set_item(&dict, pair[0].clone(), pair[1].clone())?;
                    }
                }
                b'c' => {
                    // GLOBAL
                    let module = self.line()?;
                    let name = self.line()?;
                    let global = format!("{module} {name}");
                    require(
                        global == "collections OrderedDict"
                            || global == "torch._utils _rebuild_tensor_v2"
                            || storage_type(&global).is_some(),
                        || format!("checkpoint references {global}"),
                    )?;
                    let value = make(Kind::Global);
                    value.borrow_mut().text = global;
                    self.push(value);
                }
                b'R' => {
                    // REDUCE
                    let arguments = self.pop()?;
                    let callable = self.pop()?;
                    let value = reduce(&callable, &arguments)?;
                    self.push(value);
                }
                b'Q' => {
                    // BINPERSID
                    let id = self.pop()?;
                    let value = persistent_load(&id)?;
                    self.push(value);
                }
                b'b' => {
                    // BUILD: OrderedDict._metadata and similar; dropped.
                    self.pop()?;
                    require(!self.stack.is_empty(), || "BUILD without an object".into())?;
                }
                _ => return Err(format!("unsupported pickle opcode {op}")),
            }
        }
    }

    fn byte(&mut self) -> Result<u8> {
        require(self.position < self.bytes.data.len(), || {
            "pickle ends early".into()
        })?;
        let byte = self.bytes.data[self.position];
        self.position += 1;
        Ok(byte)
    }

    fn read(&mut self, width: usize) -> Result<u64> {
        let value = self.bytes.read(self.position as u64, width)?;
        self.position += width;
        Ok(value)
    }

    fn line(&mut self) -> Result<String> {
        let mut text = Vec::new();
        loop {
            let byte = self.byte()?;
            if byte == b'\n' {
                return Ok(String::from_utf8_lossy(&text).into_owned());
            }
            text.push(byte);
        }
    }

    fn push_string(&mut self, length: u64) -> Result<()> {
        require(
            (self.position as u64).saturating_add(length) <= self.bytes.data.len() as u64,
            || "pickle string runs past".into(),
        )?;
        let end = self.position + length as usize;
        let value = make(Kind::String);
        value.borrow_mut().text =
            String::from_utf8_lossy(&self.bytes.data[self.position..end]).into_owned();
        self.position = end;
        self.push(value);
        Ok(())
    }

    fn push(&mut self, value: Ref) {
        self.stack.push(value);
    }

    fn pop(&mut self) -> Result<Ref> {
        require(
            !self.stack.is_empty()
                && self
                    .marks
                    .last()
                    .is_none_or(|&mark| self.stack.len() > mark),
            || "pickle stack underflow".into(),
        )?;
        Ok(self.stack.pop().unwrap_or_else(|| make(Kind::None)))
    }

    fn top(&self) -> Result<Ref> {
        self.stack
            .last()
            .cloned()
            .ok_or_else(|| "pickle stack underflow".into())
    }

    fn pop_mark(&mut self) -> Result<Vec<Ref>> {
        let mark = self.marks.pop().ok_or("pickle mark missing")?;
        require(mark <= self.stack.len(), || "pickle stack underflow".into())?;
        Ok(self.stack.split_off(mark))
    }

    fn memo_get(&self, index: u32) -> Result<Ref> {
        self.memo
            .get(&index)
            .cloned()
            .ok_or_else(|| "pickle memo entry missing".into())
    }
}

fn set_item(dict: &Ref, key: Ref, value: Ref) -> Result<()> {
    require(kind(dict) == Kind::Dict, || "SETITEM on a non-dict".into())?;
    dict.try_borrow_mut()
        .map_err(|_| "SETITEM on itself".to_owned())?
        .members
        .push((key, value));
    Ok(())
}

fn reduce(callable: &Ref, arguments: &Ref) -> Result<Ref> {
    let callable = callable.borrow();
    let arguments = arguments.borrow();
    require(
        callable.kind == Kind::Global && arguments.kind == Kind::Tuple,
        || "REDUCE of a non-global".into(),
    )?;
    if callable.text == "collections OrderedDict" {
        require(arguments.items.is_empty(), || {
            "OrderedDict with arguments".into()
        })?;
        return Ok(make(Kind::Dict));
    }
    require(callable.text == "torch._utils _rebuild_tensor_v2", || {
        format!("REDUCE of {}", callable.text)
    })?;
    // (storage, storage_offset, size, stride, requires_grad,
    //  backward_hooks[, metadata])
    let items = &arguments.items;
    require(
        items.len() >= 4 && kind(&items[0]) == Kind::Storage && kind(&items[1]) == Kind::Int,
        || "malformed _rebuild_tensor_v2 call".into(),
    )?;
    let tensor = make(Kind::Tensor);
    {
        let mut object = tensor.borrow_mut();
        object.storage = Some(items[0].clone());
        object.integer = items[1].borrow().integer;
        object.size = integers(&items[2], "size")?;
        object.stride = integers(&items[3], "stride")?;
        require(object.size.len() == object.stride.len(), || {
            "tensor size and stride differ in rank".into()
        })?;
    }
    Ok(tensor)
}

fn persistent_load(id: &Ref) -> Result<Ref> {
    // ("storage", storage type, key, location, element count)
    let id = id.borrow();
    let valid = id.kind == Kind::Tuple && id.items.len() == 5 && {
        let tag = id.items[0].borrow();
        tag.kind == Kind::String
            && tag.text == "storage"
            && kind(&id.items[1]) == Kind::Global
            && kind(&id.items[2]) == Kind::String
            && kind(&id.items[4]) == Kind::Int
    };
    require(valid, || "unsupported persistent id".into())?;
    let global = id.items[1].borrow().text.clone();
    let dtype = storage_type(&global).ok_or_else(|| format!("unsupported storage {global}"))?;
    let storage = make(Kind::Storage);
    {
        let mut object = storage.borrow_mut();
        object.dtype = Some(dtype);
        object.text = id.items[2].borrow().text.clone();
        object.count = id.items[4].borrow().integer as u64;
    }
    Ok(storage)
}

/// The tensors of a mapped `torch.save` state dict (`TorchCheckpoint`).
pub struct TorchCheckpoint {
    mapping: Arc<MappedFile>,
    tensors: Vec<Tensor>,
}

impl TorchCheckpoint {
    /// Maps `path` and reads its state dict (`Open`).
    pub fn open(path: &Path) -> std::result::Result<Self, String> {
        let mapping = MappedFile::open(path)
            .map_err(|error| format!("cannot map {}: {error}", path.display()))?;
        let tensors =
            read_tensors(&mapping).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self { mapping, tensors })
    }

    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    pub fn mapping(&self) -> &Arc<MappedFile> {
        &self.mapping
    }

    /// The first tensor named `name` (`Find`).
    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

fn read_tensors(mapping: &Arc<MappedFile>) -> Result<Vec<Tensor>> {
    let file = Bytes {
        data: mapping.bytes(),
    };
    let entries = read_zip(file)?;
    let prefix = entries
        .keys()
        .find_map(|name| {
            let slash = name.find('/')?;
            (&name[slash + 1..] == "data.pkl").then(|| name[..=slash].to_owned())
        })
        .ok_or("the archive lacks data.pkl")?;
    if let Some(order) = entries.get(&format!("{prefix}byteorder")) {
        let at = order.offset as usize;
        require(
            &file.data[at..at + order.size as usize] == b"little",
            || "the checkpoint is not little endian".into(),
        )?;
    }
    let pickle = *entries
        .get(&format!("{prefix}data.pkl"))
        .ok_or("the archive lacks data.pkl")?;
    let start = pickle.offset as usize;
    let mut root = Unpickler::new(Bytes {
        data: &file.data[start..start + pickle.size as usize],
    })
    .load()?;
    require(kind(&root) == Kind::Dict, || {
        "the checkpoint is not a dict".into()
    })?;
    let unwrapped = root.borrow().members.iter().find_map(|(key, value)| {
        let key = key.borrow();
        (key.kind == Kind::String && key.text == "state_dict" && kind(value) == Kind::Dict)
            .then(|| value.clone())
    });
    if let Some(state_dict) = unwrapped {
        root = state_dict;
    }
    let mut tensors = Vec::new();
    for (key, value) in &root.borrow().members {
        let key = key.borrow();
        let value = value.borrow();
        require(
            key.kind == Kind::String && value.kind == Kind::Tensor,
            || "the state dict holds a non-tensor entry".into(),
        )?;
        let storage_ref = value.storage.clone().ok_or("tensor without storage")?;
        let storage = storage_ref.borrow();
        let found = entries
            .get(&format!("{prefix}data/{}", storage.text))
            .ok_or_else(|| format!("storage {} of {} is missing", storage.text, key.text))?;
        let dtype = storage.dtype.ok_or("storage without type")?;
        let element = dtype.size() as u64;
        require(found.size >= storage.count.saturating_mul(element), || {
            format!("storage {} is truncated", storage.text)
        })?;
        let mut elements = 1u64;
        for dim in (0..value.size.len()).rev() {
            let size = value.size[dim] as u64;
            require(size <= 1 || value.stride[dim] as u64 == elements, || {
                format!("{} is not contiguous", key.text)
            })?;
            elements = elements.saturating_mul(size);
        }
        let shape: Vec<usize> = value.size.iter().map(|&size| size as usize).collect();
        let offset = value.integer as u64;
        require(offset.saturating_add(elements) <= storage.count, || {
            format!("{} lies outside its storage", key.text)
        })?;
        tensors.push(Tensor::new(
            key.text.clone(),
            dtype,
            shape,
            (elements * element) as usize,
            found.offset + offset * element,
            mapping.clone(),
        ));
    }
    Ok(tensors)
}
