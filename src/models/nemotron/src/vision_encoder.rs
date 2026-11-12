//! NVIDIA's vision projector on the GPU (`vision_encoder.cpp`): the C-RADIO
//! v4-H ViT over an image's patches and the projector into the trunk's
//! embedding space. The residual stream, norms, softmax and accumulation are
//! float32; the matrix products take 16-bit operands (FP16 by default).

use std::path::Path;
use std::sync::{Arc, Mutex};

use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceSlice, DeviceTensor, F16, HipError, Launch, LaunchError,
    MappedFile, Placement, Stream, WeightLoader,
};
use gufo_nemotron_kernels as k;

use crate::config::{Config, VisionConfig};
use crate::vision_gemm::{VisionGemm, VisionOperand};
use crate::vision_image::ImageGrid;
use crate::weights::{GgufShard, TensorRef, VisionWeights};

/// The trunk input rows of one image on the device: one float32 row of the
/// trunk's width per merged token, row-major over the halved grid. Readable
/// on every stream.
pub struct VisionEmbedding {
    data: DeviceTensor<f32>,
    rows: u32,
    width: u32,
}

impl VisionEmbedding {
    pub fn data(&self) -> DeviceSlice<'_, f32> {
        self.data.view()
    }

    pub fn tensor(&self) -> &DeviceTensor<f32> {
        &self.data
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn width(&self) -> u32 {
        self.width
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionEncoderOptions {
    /// Format of the matrix-product operands. FP16 keeps the encoder about
    /// ten times closer to the float32 model than BF16, in which NVIDIA runs
    /// it; every operand stays far inside FP16's range.
    pub operand: VisionOperand,
}

impl Default for VisionEncoderOptions {
    fn default() -> Self {
        Self {
            operand: VisionOperand::F16,
        }
    }
}

/// Receives float32 activations after a stage (`embed`, `block.N`,
/// `output`), for parity checks.
pub type Observer<'a> = &'a dyn Fn(&str, &[f32]);

struct DeviceLayer {
    norm1: DeviceTensor<f32>,
    norm1_b: DeviceTensor<f32>,
    /// `[3 width][width]`: q, k, v
    qkv: DeviceBuffer<F16>,
    /// `[3 width]`
    qkv_b: DeviceTensor<f32>,
    out: DeviceBuffer<F16>,
    out_b: DeviceTensor<f32>,
    norm2: DeviceTensor<f32>,
    norm2_b: DeviceTensor<f32>,
    up: DeviceBuffer<F16>,
    up_b: DeviceTensor<f32>,
    down: DeviceBuffer<F16>,
    down_b: DeviceTensor<f32>,
}

/// Device memory that only grows.
fn reserve<T: DeviceCopy>(
    stream: &Stream,
    buffer: &mut Option<DeviceBuffer<T>>,
    count: usize,
) -> Result<(), HipError> {
    if buffer.as_ref().is_none_or(|b| b.len() < count) {
        *buffer = None;
        *buffer = Some(stream.alloc(count.max(1))?);
    }
    Ok(())
}

struct Inner {
    stream: Stream,
    gemm: VisionGemm,
    /// `[width][patch values]` operand
    patch: DeviceBuffer<F16>,
    position: DeviceTensor<f32>,
    prefix: DeviceTensor<f32>,
    layers: Vec<DeviceLayer>,
    projector_norm: DeviceTensor<f32>,
    projector_up: DeviceBuffer<F16>,
    projector_down: DeviceBuffer<F16>,
    /// `[tokens][width]` residual stream
    x: Option<DeviceBuffer<f32>>,
    /// `[tokens][width]` operand rows
    h: Option<DeviceBuffer<F16>>,
    /// `[tokens][ff_size]` product outputs
    y: Option<DeviceBuffer<f32>>,
    /// `[tokens][ff_size]` operand rows
    big: Option<DeviceBuffer<F16>>,
}

/// NVIDIA's vision projector on the GPU (`VisionEncoder`). Encode calls are
/// serialized.
pub struct VisionEncoder {
    config: VisionConfig,
    operand: VisionOperand,
    identity: String,
    resident_bytes: usize,
    inner: Mutex<Inner>,
}

fn hip(error: HipError) -> String {
    error.to_string()
}

fn launched(result: Result<Launch, LaunchError>, what: &str) -> Result<(), String> {
    match result {
        Ok(Launch::Launched) => Ok(()),
        Ok(Launch::Unsupported) => Err(format!("{what}: unsupported launch")),
        Err(error) => Err(format!("{what}: {error}")),
    }
}

fn gemm(result: Result<(), LaunchError>) -> Result<(), String> {
    result.map_err(|e| format!("vision product: {e}"))
}

/// Queues the f32 tensors that stay resident, on their own loader so that
/// the matrices' staging can be freed.
struct Resident<'s> {
    loader: WeightLoader<'s>,
    file: &'s Arc<MappedFile>,
    queued: Vec<gufo_hip::Pending<f32>>,
    bytes: usize,
}

impl Resident<'_> {
    fn add(&mut self, parts: &[&TensorRef]) -> Result<usize, String> {
        let ranges: Vec<(&Arc<MappedFile>, usize, usize)> = parts
            .iter()
            .map(|t| {
                usize::try_from(t.file_offset)
                    .map(|offset| (self.file, offset, t.size_bytes()))
                    .map_err(|_| "vision weight allocation failed".to_string())
            })
            .collect::<Result<_, _>>()?;
        self.bytes += ranges.iter().map(|r| r.2).sum::<usize>();
        let pending = self
            .loader
            .add_parts::<f32>(&ranges, 0)
            .map_err(|e| format!("vision weight allocation failed: {e}"))?;
        self.queued.push(pending);
        Ok(self.queued.len() - 1)
    }
}

/// Loads `parts` (BF16 matrices joined by rows) and converts them into an
/// operand buffer.
fn operand_matrix(
    stream: &Stream,
    file: &Arc<MappedFile>,
    parts: &[&TensorRef],
    operand: VisionOperand,
    placement: (Placement, bool),
) -> Result<DeviceBuffer<F16>, String> {
    let mut loader = WeightLoader::new(stream, placement.0).managed_arenas(placement.1);
    let mut ranges = Vec::with_capacity(parts.len());
    let mut count = 0;
    for t in parts {
        let offset = usize::try_from(t.file_offset)
            .map_err(|_| "vision weight allocation failed".to_string())?;
        ranges.push((file, offset, t.size_bytes()));
        count += t.size_bytes() / 2;
    }
    let pending = loader
        .add_parts::<Bf16>(&ranges, 0)
        .map_err(|e| format!("vision weight allocation failed: {e}"))?;
    let staged = loader.finish().map_err(hip)?.get(pending);
    let mut out = stream.alloc_persistent::<F16>(count.max(1)).map_err(hip)?;
    launched(
        k::vision_bf16_to_operand(
            stream,
            staged.view(),
            out.slice_mut(0..count),
            count,
            operand.code(),
        ),
        "vision weight conversion",
    )?;
    // The staging goes once the conversion has read it.
    stream.synchronize().map_err(hip)?;
    Ok(out)
}

impl VisionEncoder {
    /// Loads and uploads the mmproj GGUF at `path` for `trunk`.
    pub fn load(
        path: &Path,
        trunk: &Config,
        options: VisionEncoderOptions,
    ) -> Result<Self, String> {
        Self::load_inner(path, trunk, options)
            .map_err(|e| format!("vision projector {}: {e}", path.display()))
    }

    fn load_inner(
        path: &Path,
        trunk: &Config,
        options: VisionEncoderOptions,
    ) -> Result<Self, String> {
        let file = MappedFile::open(path).map_err(|e| e.to_string())?;
        let shards = [GgufShard::parse(file.bytes())?];
        let w = VisionWeights::bind(&shards, trunk)?;
        let c = w.config.clone();
        let operand = options.operand;
        let identity = gufo_core::gguf_identity::identity_hex(&[path.to_path_buf()])?;
        let stream = Stream::new().map_err(hip)?;
        let gemm = VisionGemm::new(operand, &stream).map_err(|e| e.to_string())?;
        let width = c.width as usize;

        // Every tensor goes up in its file format, placed as the C++
        // allocator places their total; matrices become operands.
        let mut total = w.patch.size_bytes()
            + w.position.size_bytes()
            + w.prefix.size_bytes()
            + w.projector_norm.size_bytes()
            + w.projector_up.size_bytes()
            + w.projector_down.size_bytes();
        for l in &w.layers {
            for t in [
                &l.norm1, &l.norm1_b, &l.q, &l.q_b, &l.k, &l.k_b, &l.v, &l.v_b, &l.out, &l.out_b,
                &l.norm2, &l.norm2_b, &l.up, &l.up_b, &l.down, &l.down_b,
            ] {
                total += t.size_bytes();
            }
        }
        let (free, _) = gufo_hip::device_memory().map_err(hip)?;
        let managed = !gufo_hip::heap_fits_above_reserve(free as u64, total as u64);
        let placement = (Placement::DirectCopy, managed);

        let mut resident = Resident {
            loader: WeightLoader::new(&stream, Placement::DirectCopy).managed_arenas(managed),
            file: &file,
            queued: Vec::new(),
            bytes: 0,
        };
        let position = resident.add(&[&w.position])?;
        let prefix = resident.add(&[&w.prefix])?;
        let mut layer_f32 = Vec::with_capacity(w.layers.len());
        for l in &w.layers {
            layer_f32.push([
                resident.add(&[&l.norm1])?,
                resident.add(&[&l.norm1_b])?,
                resident.add(&[&l.q_b, &l.k_b, &l.v_b])?,
                resident.add(&[&l.out_b])?,
                resident.add(&[&l.norm2])?,
                resident.add(&[&l.norm2_b])?,
                resident.add(&[&l.up_b])?,
                resident.add(&[&l.down_b])?,
            ]);
        }
        let projector_norm = resident.add(&[&w.projector_norm])?;
        let Resident {
            loader,
            queued,
            bytes: resident_f32,
            ..
        } = resident;
        let loaded = loader.finish().map_err(hip)?;
        let mut tensors: Vec<Option<DeviceTensor<f32>>> =
            queued.into_iter().map(|p| Some(loaded.get(p))).collect();
        let mut take = |index: usize| -> Result<DeviceTensor<f32>, String> {
            tensors
                .get_mut(index)
                .and_then(Option::take)
                .ok_or_else(|| "vision weight upload failed".to_string())
        };

        // The patch projection is F32 and becomes an operand.
        let patch_values = c.patch_values() as usize;
        let patch_count = width * patch_values;
        let patch = {
            let mut staging =
                WeightLoader::new(&stream, Placement::DirectCopy).managed_arenas(managed);
            let offset = usize::try_from(w.patch.file_offset)
                .map_err(|_| "vision weight allocation failed".to_string())?;
            let pending = staging
                .add::<f32>(&file, offset, w.patch.size_bytes() / 4)
                .map_err(|e| format!("vision upload: {e}"))?;
            let staged = staging.finish().map_err(hip)?.get(pending);
            let mut out = stream
                .alloc_persistent::<F16>(patch_count.max(1))
                .map_err(hip)?;
            launched(
                k::vision_to_operand(
                    &stream,
                    staged.view().range(0..patch_count),
                    out.slice_mut(0..patch_count),
                    patch_count,
                    operand.code(),
                ),
                "vision weight conversion",
            )?;
            stream.synchronize().map_err(hip)?;
            out
        };
        let mut layers = Vec::with_capacity(w.layers.len());
        let mut matrix_bytes = 0;
        for (l, f) in w.layers.iter().zip(&layer_f32) {
            matrix_bytes += l.q.size_bytes()
                + l.k.size_bytes()
                + l.v.size_bytes()
                + l.out.size_bytes()
                + l.up.size_bytes()
                + l.down.size_bytes();
            layers.push(DeviceLayer {
                norm1: take(f[0])?,
                norm1_b: take(f[1])?,
                qkv: operand_matrix(&stream, &file, &[&l.q, &l.k, &l.v], operand, placement)?,
                qkv_b: take(f[2])?,
                out: operand_matrix(&stream, &file, &[&l.out], operand, placement)?,
                out_b: take(f[3])?,
                norm2: take(f[4])?,
                norm2_b: take(f[5])?,
                up: operand_matrix(&stream, &file, &[&l.up], operand, placement)?,
                up_b: take(f[6])?,
                down: operand_matrix(&stream, &file, &[&l.down], operand, placement)?,
                down_b: take(f[7])?,
            });
        }
        let projector_up = operand_matrix(&stream, &file, &[&w.projector_up], operand, placement)?;
        let projector_down =
            operand_matrix(&stream, &file, &[&w.projector_down], operand, placement)?;
        matrix_bytes += w.projector_up.size_bytes() + w.projector_down.size_bytes();
        let resident_bytes = resident_f32 + matrix_bytes + patch_count * std::mem::size_of::<F16>();
        let inner = Inner {
            position: take(position)?,
            prefix: take(prefix)?,
            projector_norm: take(projector_norm)?,
            stream,
            gemm,
            patch,
            layers,
            projector_up,
            projector_down,
            x: None,
            h: None,
            y: None,
            big: None,
        };
        Ok(Self {
            config: c,
            operand,
            identity,
            resident_bytes,
            inner: Mutex::new(inner),
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub fn operand(&self) -> VisionOperand {
        self.operand
    }

    /// Full-content identity of the projector file (`gguf-sha256-v1`).
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Device scratch an image of the largest grid needs.
    pub fn max_scratch_bytes(&self) -> usize {
        let c = &self.config;
        let tokens = (c.prefix_tokens + c.max_patches) as usize;
        let merged = (c.max_patches / (c.merge * c.merge)) as usize;
        tokens * c.width as usize * (4 + 2)
            + tokens * c.ff_size as usize * (4 + 2)
            + merged * c.projection as usize * 4
    }

    /// Picks the matrix-product algorithms for every image size now instead
    /// of during the first images.
    pub fn prepare(&self) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let c = &self.config;
        let Inner { stream, gemm, .. } = &mut *inner;
        let fail = |e: LaunchError| e.to_string();
        // ViT products have at least min_patches rows; every class from there
        // up shares one timing. The projector's rows start at a quarter of
        // that.
        let (width, ff, patch_values) = (
            c.width as usize,
            c.ff_size as usize,
            c.patch_values() as usize,
        );
        let tokens = (c.prefix_tokens + c.min_patches) as usize;
        for rows in [c.min_patches as usize, tokens] {
            gemm.prepare(stream, rows, patch_values, width)
                .map_err(fail)?;
        }
        gemm.prepare(stream, tokens, width, 3 * width)
            .map_err(fail)?;
        gemm.prepare(stream, tokens, width, width).map_err(fail)?;
        gemm.prepare(stream, tokens, width, ff).map_err(fail)?;
        gemm.prepare(stream, tokens, ff, width).map_err(fail)?;
        let merge = (c.merge * c.merge) as usize;
        let mut rows = c.min_patches as usize / merge;
        while rows <= 2 * c.max_patches as usize / merge {
            gemm.prepare(
                stream,
                rows,
                c.merged_width() as usize,
                c.projector_hidden as usize,
            )
            .map_err(fail)?;
            gemm.prepare(
                stream,
                rows,
                c.projector_hidden as usize,
                c.projection as usize,
            )
            .map_err(fail)?;
            rows *= 2;
        }
        stream
            .synchronize()
            .map_err(|e| format!("vision preparation: {e}"))
    }

    /// Encodes `preprocess_image`'s patches of `grid`. Calls are serialized.
    pub fn encode(
        &self,
        patches: &[f32],
        grid: &ImageGrid,
        observer: Option<Observer<'_>>,
    ) -> Result<Arc<VisionEmbedding>, String> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let result = self.run(&mut inner, patches, grid, observer);
        if result.is_err() {
            let _ = inner.stream.synchronize();
        }
        result.map_err(|e| format!("vision encoder: {e}"))
    }

    fn run(
        &self,
        inner: &mut Inner,
        patches: &[f32],
        grid: &ImageGrid,
        observer: Option<Observer<'_>>,
    ) -> Result<Arc<VisionEmbedding>, String> {
        let c = &self.config;
        let op = self.operand.code();
        let width = c.width as usize;
        let ff = c.ff_size as usize;
        let n = grid.patches() as usize;
        let patch_values = c.patch_values() as usize;
        if !grid.rows.is_multiple_of(c.merge)
            || !grid.cols.is_multiple_of(c.merge)
            || n == 0
            || n > c.max_patches as usize
            || patches.len() != n * patch_values
        {
            return Err("invalid vision encoder input".into());
        }
        let prefix_tokens = c.prefix_tokens as usize;
        let tokens = prefix_tokens + n;
        let merged = n / (c.merge * c.merge) as usize;
        let projector_hidden = c.projector_hidden as usize;
        let projection = c.projection as usize;
        let big_count = (tokens * ff)
            .max(merged * projector_hidden)
            .max(patches.len());
        let Inner {
            stream,
            gemm: products,
            patch,
            position,
            prefix,
            layers,
            projector_norm,
            projector_up,
            projector_down,
            x,
            h,
            y,
            big,
        } = inner;
        let s = &*stream;
        reserve(s, x, tokens * width).map_err(hip)?;
        reserve(s, h, tokens * width).map_err(hip)?;
        reserve(s, y, big_count).map_err(hip)?;
        reserve(s, big, big_count).map_err(hip)?;
        let (Some(xs), Some(hs), Some(ys), Some(bs)) = (x, h, y, big) else {
            return Err("vision scratch allocation".into());
        };
        let observe = |name: &str, device: DeviceSlice<'_, f32>| -> Result<(), String> {
            if let Some(observer) = observer {
                let mut host = vec![0.0f32; device.len()];
                s.download_blocking(device, &mut host)
                    .map_err(|e| format!("vision observer copy: {e}"))?;
                observer(name, &host);
            }
            Ok(())
        };

        // Patch projection into the rows after the prefix, then positions.
        s.upload(patches, ys.slice_mut(0..patches.len()))
            .map_err(|e| format!("vision patch upload: {e}"))?;
        launched(
            k::vision_to_operand(
                s,
                ys.slice(0..patches.len()),
                bs.slice_mut(0..patches.len()),
                patches.len(),
                op,
            ),
            "vision patches",
        )?;
        gemm(products.linear(
            s,
            bs.slice(0..n * patch_values),
            patch.slice(0..width * patch_values),
            xs.slice_mut(prefix_tokens * width..tokens * width),
            n,
            patch_values,
            width,
        ))?;
        launched(
            k::vision_embed(
                s,
                xs.slice_mut(0..tokens * width),
                prefix.view(),
                position.view(),
                prefix_tokens,
                grid.rows as usize,
                grid.cols as usize,
                c.position_side as usize,
                width,
            ),
            "vision embedding",
        )?;
        observe("embed", xs.slice(0..tokens * width))?;

        let scale = 1.0f32 / (c.head_dim() as f32).sqrt();
        let heads = c.num_heads as usize;
        for (il, l) in layers.iter().enumerate() {
            launched(
                k::vision_layer_norm(
                    s,
                    xs.slice(0..tokens * width),
                    l.norm1.view(),
                    l.norm1_b.view(),
                    hs.slice_mut(0..tokens * width),
                    tokens,
                    width,
                    c.eps,
                    op,
                ),
                "vision norm",
            )?;
            gemm(products.linear(
                s,
                hs.slice(0..tokens * width),
                l.qkv.slice(0..3 * width * width),
                ys.slice_mut(0..tokens * 3 * width),
                tokens,
                width,
                3 * width,
            ))?;
            launched(
                k::vision_bias(
                    s,
                    ys.slice(0..tokens * 3 * width),
                    l.qkv_b.view(),
                    bs.slice_mut(0..tokens * 3 * width),
                    tokens,
                    3 * width,
                    op,
                ),
                "vision qkv",
            )?;
            launched(
                k::vision_attention(
                    s,
                    bs.slice(0..tokens * 3 * width),
                    hs.slice_mut(0..tokens * width),
                    tokens,
                    heads,
                    scale,
                    op,
                ),
                "vision attention",
            )?;
            gemm(products.linear(
                s,
                hs.slice(0..tokens * width),
                l.out.slice(0..width * width),
                ys.slice_mut(0..tokens * width),
                tokens,
                width,
                width,
            ))?;
            launched(
                k::vision_residual(
                    s,
                    xs.slice_mut(0..tokens * width),
                    ys.slice(0..tokens * width),
                    l.out_b.view(),
                    tokens,
                    width,
                ),
                "vision residual",
            )?;
            launched(
                k::vision_layer_norm(
                    s,
                    xs.slice(0..tokens * width),
                    l.norm2.view(),
                    l.norm2_b.view(),
                    hs.slice_mut(0..tokens * width),
                    tokens,
                    width,
                    c.eps,
                    op,
                ),
                "vision norm",
            )?;
            gemm(products.linear(
                s,
                hs.slice(0..tokens * width),
                l.up.slice(0..width * ff),
                ys.slice_mut(0..tokens * ff),
                tokens,
                width,
                ff,
            ))?;
            launched(
                k::vision_bias_gelu(
                    s,
                    ys.slice(0..tokens * ff),
                    l.up_b.view(),
                    bs.slice_mut(0..tokens * ff),
                    tokens,
                    ff,
                    op,
                ),
                "vision mlp",
            )?;
            gemm(products.linear(
                s,
                bs.slice(0..tokens * ff),
                l.down.slice(0..ff * width),
                ys.slice_mut(0..tokens * width),
                tokens,
                ff,
                width,
            ))?;
            launched(
                k::vision_residual(
                    s,
                    xs.slice_mut(0..tokens * width),
                    ys.slice(0..tokens * width),
                    l.down_b.view(),
                    tokens,
                    width,
                ),
                "vision residual",
            )?;
            if observer.is_some() {
                observe(&format!("block.{il}"), xs.slice(0..tokens * width))?;
            }
        }

        // The projector: merged patch features, RMSNorm, ReLU² MLP.
        let merged_width = c.merged_width() as usize;
        launched(
            k::vision_merge_norm(
                s,
                xs.slice(0..tokens * width),
                projector_norm.view(),
                hs.slice_mut(0..merged * merged_width),
                prefix_tokens,
                grid.rows as usize,
                grid.cols as usize,
                width,
                c.projector_eps,
                op,
            ),
            "vision merge",
        )?;
        gemm(products.linear(
            s,
            hs.slice(0..merged * merged_width),
            projector_up.slice(0..merged_width * projector_hidden),
            ys.slice_mut(0..merged * projector_hidden),
            merged,
            merged_width,
            projector_hidden,
        ))?;
        launched(
            k::vision_relu2(
                s,
                ys.slice(0..merged * projector_hidden),
                bs.slice_mut(0..merged * projector_hidden),
                merged * projector_hidden,
                op,
            ),
            "vision projector",
        )?;
        let mut embedding = s
            .alloc::<f32>((merged * projection).max(1))
            .map_err(|e| format!("vision embedding allocation: {e}"))?;
        gemm(products.linear(
            s,
            bs.slice(0..merged * projector_hidden),
            projector_down.slice(0..projector_hidden * projection),
            embedding.slice_mut(0..merged * projection),
            merged,
            projector_hidden,
            projection,
        ))?;
        s.synchronize()
            .map_err(|e| format!("vision encoder: {e}"))?;
        observe("output", embedding.slice(0..merged * projection))?;
        let data = embedding.freeze(s).map_err(hip)?;
        Ok(Arc::new(VisionEmbedding {
            data,
            rows: merged as u32,
            width: c.projection,
        }))
    }
}
