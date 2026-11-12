//! The PLE n-gram embedding, a port of `ngram.cpp`: the rows each token
//! gathers (a hash of its n-gram window per head) and the table reader that
//! fetches them from the GGUF shard through the kernel page cache,
//! through a small cache of compressed rows and a resident worker pool.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, TryLockError};
use std::thread::JoinHandle;

use gufo_formats::GgmlType;
use gufo_formats::dequant::dequantize;

use crate::config::{Config, MAX_PLE_NGRAM};

/// Next-chunk prefetch hides later gathers behind the current pass, but a
/// prefill's first chunk waits for its own gather. Readers block in
/// `pread`, so the pool sets that gather's queue depth.
const WORKERS: usize = 128;
const READ_BATCH: usize = 8;
const BATCH_JOBS: usize = 1024;
const CACHE_BYTES: usize = 8 * 1024 * 1024;

/// The newest-first previous tokens of one sequence, the whole hash state of
/// its PLE window; copying it snapshots the state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramHistory {
    pub prev: [i32; MAX_PLE_NGRAM - 1],
}

impl NgramHistory {
    pub const NONE: i32 = -1;

    pub fn reset(&mut self) {
        self.prev = [Self::NONE; MAX_PLE_NGRAM - 1];
    }
}

impl Default for NgramHistory {
    fn default() -> Self {
        Self {
            prev: [Self::NONE; MAX_PLE_NGRAM - 1],
        }
    }
}

/// The table rows every token gathers, `ple_heads` per token, head-major, as
/// the reference hashes them (`HashNgramRows`). An EOS anywhere in the
/// window cuts every older token, and a missing predecessor reads as EOS.
/// `history` advances past `tokens`.
#[allow(clippy::needless_range_loop)] // indices name window positions
pub fn hash_ngram_rows(c: &Config, history: &mut NgramHistory, tokens: &[i32], rows: &mut [u32]) {
    let n = c.ple_ngram_size as usize;
    let eos = i64::from(c.ple_eos_token);
    let heads = c.ple_heads as usize;
    let per_ngram = c.ple_heads_per_ngram as usize;
    let mut context = [0u64; MAX_PLE_NGRAM];
    for (i, &token) in tokens.iter().enumerate() {
        // A token's own EOS does not cut its context, only an older one does.
        context[0] = token as u64;
        let mut cut = false;
        for s in 1..n {
            let t = if cut {
                NgramHistory::NONE
            } else {
                history.prev[s - 1]
            };
            cut = cut || t < 0 || i64::from(t) == eos;
            context[s] = if cut { eos as u64 } else { t as u64 };
        }
        let out = &mut rows[i * heads..(i + 1) * heads];
        for order in 2..=n {
            let mut mixed = context[0].wrapping_mul(c.ple_multipliers[0]);
            for j in 1..order {
                mixed ^= context[j].wrapping_mul(c.ple_multipliers[j]);
            }
            let base = (order - 2) * per_ngram;
            for g in 0..per_ngram {
                let h = base + g;
                out[h] = (mixed % u64::from(c.ple_head_vocab[h])) as u32 + c.ple_head_offsets[h];
            }
        }
        history.prev.copy_within(0..MAX_PLE_NGRAM - 2, 1);
        history.prev[0] = token;
    }
}

/// One cached compressed row. A busy slot is skipped, never waited for.
struct CacheSlot {
    row: u32,
    valid: bool,
    bytes: Box<[u8]>,
}

struct Job {
    row: u32,
    /// Row index in the output.
    at: usize,
}

struct Gather {
    jobs: Vec<Job>,
    /// (source, destination) output rows of repeated IDs.
    copies: Vec<(usize, usize)>,
    next: usize,
    pending: usize,
    active: bool,
    failed: bool,
    stop: bool,
}

struct Shared {
    file: File,
    base_offset: u64,
    rows: u64,
    row_dim: usize,
    row_bytes: usize,
    ty: GgmlType,
    cache: Vec<Mutex<CacheSlot>>,
    gather: Mutex<Gather>,
    /// Decoded rows, `row_dim` floats each.
    output: Mutex<Vec<f32>>,
    wake: Condvar,
    done: Condvar,
}

/// Reads n-gram embedding rows (dequantized to F32, laid out
/// [token][head][ple_head_dim]) from the table's file.
pub struct NgramTable {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panicking worker leaves consistent state: every field is plain data.
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

impl NgramTable {
    /// Opens the table of `rows` rows of `row_dim` values at `offset` of
    /// `file`, on its own descriptor.
    ///
    /// Reads go through the page cache, unlike `ngram.cpp` (`O_DIRECT`): hot
    /// rows then come from free memory, which the kernel reclaims under
    /// pressure, and neither count as the process's resident memory (as a
    /// mapping would). Random-access advice keeps a missing row a single
    /// page read instead of a readahead window.
    pub fn open(
        file: &File,
        offset: u64,
        rows: u64,
        row_dim: u32,
        ty: GgmlType,
    ) -> Result<Self, String> {
        let (block, block_bytes) = match ty {
            GgmlType::Iq4Nl => (32, 18),
            GgmlType::Bf16 => (1, 2),
            _ => return Err("unsupported n-gram table format".into()),
        };
        if row_dim == 0 || !row_dim.is_multiple_of(block) {
            return Err("unsupported n-gram table format".into());
        }
        let row_bytes = row_dim as usize / block as usize * block_bytes;
        let path = format!("/proc/self/fd/{}", std::os::fd::AsRawFd::as_raw_fd(file));
        let file = File::open(&path).map_err(|e| format!("cannot open bound n-gram table: {e}"))?;
        let table_bytes = rows
            .checked_mul(row_bytes as u64)
            .ok_or("n-gram table size overflows")?;
        gufo_hip::advise_random_access(&file, offset, table_bytes)
            .map_err(|e| format!("n-gram table access advice: {e}"))?;
        let slots = (CACHE_BYTES / row_bytes)
            .checked_ilog2()
            .map_or(0, |b| 1usize << b);
        let cache = (0..slots)
            .map(|_| {
                Mutex::new(CacheSlot {
                    row: 0,
                    valid: false,
                    bytes: vec![0u8; row_bytes].into_boxed_slice(),
                })
            })
            .collect();
        let shared = Arc::new(Shared {
            file,
            base_offset: offset,
            rows,
            row_dim: row_dim as usize,
            row_bytes,
            ty,
            cache,
            gather: Mutex::new(Gather {
                jobs: Vec::new(),
                copies: Vec::new(),
                next: 0,
                pending: 0,
                active: false,
                failed: false,
                stop: false,
            }),
            output: Mutex::new(Vec::new()),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(WORKERS);
        for _ in 0..WORKERS {
            let shared = Arc::clone(&shared);
            let worker = std::thread::Builder::new()
                .name("ngram-reader".into())
                .spawn(move || shared.work())
                .map_err(|e| format!("n-gram reader thread: {e}"))?;
            workers.push(worker);
        }
        Ok(Self { shared, workers })
    }

    pub fn row_dim(&self) -> usize {
        self.shared.row_dim
    }

    pub fn rows(&self) -> u64 {
        self.shared.rows
    }

    /// Starts gathering `rows`; false (nothing started) for a row past the
    /// table or while a gather is outstanding. Only one gather at a time.
    pub fn start_read(&self, rows: &[u32]) -> bool {
        let s = &self.shared;
        let mut gather = lock(&s.gather);
        if gather.active || rows.iter().any(|&r| u64::from(r) >= s.rows) {
            return false;
        }
        lock(&s.output).resize(rows.len() * s.row_dim, 0.0);
        // Each distinct row is read once and copied to its other places.
        let mut first = HashMap::with_capacity(rows.len());
        gather.jobs.clear();
        gather.copies.clear();
        for (at, &row) in rows.iter().enumerate() {
            match first.entry(row) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(at);
                    gather.jobs.push(Job { row, at });
                }
                std::collections::hash_map::Entry::Occupied(o) => {
                    gather.copies.push((*o.get(), at))
                }
            }
        }
        if gather.jobs.len() >= BATCH_JOBS {
            gather.jobs.sort_by_key(|job| job.row);
        }
        gather.next = 0;
        gather.pending = gather.jobs.len();
        gather.failed = false;
        gather.active = true;
        let pending = gather.pending;
        drop(gather);
        // One wake-up here, the caller's only cost besides hashing: each
        // worker that finds more jobs wakes the next (`work`). Cached rows
        // are decoded by the workers too, while the first layers run
        // (`ngram.cpp` decodes them here and wakes every worker).
        if pending != 0 {
            s.wake.notify_one();
        }
        true
    }

    /// Waits for the outstanding gather and copies its rows into `out`
    /// (at least as long as the gather's rows). False when none was
    /// started or a read failed.
    pub fn wait_read(&self, out: &mut [f32]) -> bool {
        self.collect(|rows| {
            if out.len() < rows.len() {
                return false;
            }
            out[..rows.len()].copy_from_slice(rows);
            true
        })
    }

    /// [`wait_read`](Self::wait_read) that hands over the gathered rows
    /// instead of copying them: they end up in `rows`, whose old storage the
    /// next gather reuses.
    pub fn wait_read_swap(&self, rows: &mut Vec<f32>) -> bool {
        self.collect(|gathered| {
            std::mem::swap(gathered, rows);
            true
        })
    }

    /// Waits for the outstanding gather and hands its complete rows to
    /// `take`.
    fn collect(&self, take: impl FnOnce(&mut Vec<f32>) -> bool) -> bool {
        let s = &self.shared;
        let mut gather = lock(&s.gather);
        if !gather.active {
            return false;
        }
        while gather.pending != 0 {
            gather = s
                .done
                .wait(gather)
                .unwrap_or_else(|poison| poison.into_inner());
        }
        gather.active = false;
        if gather.failed {
            return false;
        }
        let mut output = lock(&s.output);
        let d = s.row_dim;
        for &(src, dst) in &gather.copies {
            output.copy_within(src * d..(src + 1) * d, dst * d);
        }
        take(&mut output)
    }
}

impl Drop for NgramTable {
    fn drop(&mut self) {
        let _ = self.wait_read_swap(&mut Vec::new());
        lock(&self.shared.gather).stop = true;
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Shared {
    fn decode(&self, src: &[u8], dst: &mut [f32]) {
        match self.ty {
            GgmlType::Iq4Nl => {
                let _ = dequantize(GgmlType::Iq4Nl, src, dst);
            }
            _ => {
                let (pairs, _) = src.as_chunks::<2>();
                for (value, bits) in dst.iter_mut().zip(pairs) {
                    *value = f32::from_bits(u32::from(u16::from_le_bytes(*bits)) << 16);
                }
            }
        }
    }

    fn slot(&self, row: u32) -> usize {
        (row.wrapping_mul(2_654_435_761) as usize) & (self.cache.len() - 1)
    }

    fn read_cached(&self, row: u32, dst: &mut [f32]) -> bool {
        if self.cache.is_empty() {
            return false;
        }
        match self.cache[self.slot(row)].try_lock() {
            Ok(slot) if slot.valid && slot.row == row => {
                self.decode(&slot.bytes, dst);
                true
            }
            Ok(_) | Err(TryLockError::WouldBlock) => false,
            Err(TryLockError::Poisoned(_)) => false,
        }
    }

    /// Reads and decodes one row into `dst`; `buffer` is the worker's
    /// staging.
    fn read_one(&self, row: u32, dst: &mut [f32], buffer: &mut Vec<u8>) -> io::Result<()> {
        if u64::from(row) >= self.rows {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        if self.read_cached(row, dst) {
            return Ok(());
        }
        let offset = self.base_offset + u64::from(row) * self.row_bytes as u64;
        buffer.resize(self.row_bytes, 0);
        let mut got = 0;
        while got < self.row_bytes {
            match self.file.read_at(&mut buffer[got..], offset + got as u64) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => got += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        let bytes = &buffer[..];
        // Cache the complete row in its quantized format; a busy slot is
        // skipped (readers never wait for another row's I/O).
        if !self.cache.is_empty()
            && let Ok(mut slot) = self.cache[self.slot(row)].try_lock()
        {
            slot.bytes.copy_from_slice(bytes);
            slot.row = row;
            slot.valid = true;
        }
        self.decode(bytes, dst);
        Ok(())
    }

    fn work(&self) {
        let mut buffer = Vec::new();
        let mut decoded = Vec::new();
        let mut gather = lock(&self.gather);
        loop {
            while !gather.stop && !(gather.active && gather.next < gather.jobs.len()) {
                gather = self
                    .wake
                    .wait(gather)
                    .unwrap_or_else(|poison| poison.into_inner());
            }
            if gather.stop {
                return;
            }
            // Large gathers amortize the lock; small ones keep one read per
            // worker, so decode does not serialize its few rows.
            let take = if gather.jobs.len() >= BATCH_JOBS {
                READ_BATCH
            } else {
                1
            };
            let count = take.min(gather.jobs.len() - gather.next);
            let batch: Vec<(u32, usize)> = gather.jobs[gather.next..gather.next + count]
                .iter()
                .map(|job| (job.row, job.at))
                .collect();
            gather.next += count;
            let more = gather.next < gather.jobs.len();
            drop(gather);
            if more {
                self.wake.notify_one();
            }
            decoded.resize(count * self.row_dim, 0.0);
            let mut ok = true;
            for (i, &(row, _)) in batch.iter().enumerate() {
                let dst = &mut decoded[i * self.row_dim..(i + 1) * self.row_dim];
                ok = self.read_one(row, dst, &mut buffer).is_ok() && ok;
            }
            {
                let mut output = lock(&self.output);
                for (i, &(_, at)) in batch.iter().enumerate() {
                    output[at * self.row_dim..(at + 1) * self.row_dim]
                        .copy_from_slice(&decoded[i * self.row_dim..(i + 1) * self.row_dim]);
                }
            }
            gather = lock(&self.gather);
            gather.failed = gather.failed || !ok;
            gather.pending -= count;
            if gather.pending == 0 {
                self.done.notify_all();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        let mut c = crate::config::tests_support::flash_next();
        c.ple_eos_token = 7;
        c
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn hashes_follow_the_window_and_cut_at_eos() {
        let c = config();
        let heads = c.ple_heads as usize;
        let mut history = NgramHistory::default();
        let mut rows = vec![0u32; 3 * heads];
        hash_ngram_rows(&c, &mut history, &[5, 7, 9], &mut rows);
        assert_eq!(history.prev, [9, 7]);
        // Bigram heads of token 0: (5, EOS), since there is no predecessor.
        let expected = |ctx: [u64; 3], order: usize, g: usize| {
            let mut mixed = ctx[0].wrapping_mul(c.ple_multipliers[0]);
            for j in 1..order {
                mixed ^= ctx[j].wrapping_mul(c.ple_multipliers[j]);
            }
            let h = (order - 2) * c.ple_heads_per_ngram as usize + g;
            (mixed % u64::from(c.ple_head_vocab[h])) as u32 + c.ple_head_offsets[h]
        };
        assert_eq!(rows[0], expected([5, 7, 7], 2, 0));
        assert_eq!(rows[heads - 1], expected([5, 7, 7], 3, 7));
        // Token 2 (9) after EOS (7): the EOS predecessor cuts everything older.
        assert_eq!(rows[2 * heads + 8], expected([9, 7, 7], 3, 0));
        // The history carries across calls like one long call.
        let mut split = NgramHistory::default();
        let mut parts = vec![0u32; 3 * heads];
        hash_ngram_rows(&c, &mut split, &[5], &mut parts[..heads]);
        hash_ngram_rows(&c, &mut split, &[7, 9], &mut parts[heads..]);
        assert_eq!(parts, rows);
    }

    #[test]
    fn the_table_reads_decodes_and_repeats_rows() {
        let path = std::env::temp_dir().join(format!("gufo-ngram-{}.bin", std::process::id()));
        // 3000 IQ4_NL rows of 64 values (36 bytes) after a 100-byte header.
        let rows = 3000usize;
        let mut bytes = vec![0u8; 100];
        for r in 0..rows {
            for b in 0..2 {
                bytes.extend_from_slice(&0x3C00u16.to_le_bytes());
                bytes.extend((0..16).map(|i| ((r + b * 3 + i) % 256) as u8));
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let table = NgramTable::open(&file, 100, rows as u64, 64, GgmlType::Iq4Nl).unwrap();
        let reference = |r: usize| {
            let mut out = vec![0f32; 64];
            dequantize(
                GgmlType::Iq4Nl,
                &bytes[100 + r * 36..100 + (r + 1) * 36],
                &mut out,
            )
            .unwrap();
            out
        };
        for request in [
            vec![5u32, 2999, 5, 0],
            (0..2000u32).map(|i| (i * 7) % 3000).collect(),
        ] {
            assert!(table.start_read(&request));
            assert!(!table.start_read(&request), "one gather at a time");
            let mut out = vec![0f32; request.len() * 64];
            assert!(table.wait_read(&mut out));
            for (i, &r) in request.iter().enumerate() {
                assert_eq!(
                    out[i * 64..(i + 1) * 64],
                    reference(r as usize)[..],
                    "row {r}"
                );
            }
        }
        assert!(!table.start_read(&[3000]), "a row past the table");
        assert!(!table.wait_read(&mut []), "nothing outstanding");
        drop(table);
        let _ = std::fs::remove_file(&path);
    }

    /// The PLE constants of the Flash-Next artifact.
    fn flash_next_ple() -> Config {
        let mut c = crate::config::tests_support::flash_next();
        c.ple_eos_token = 248_044;
        c.ple_multipliers = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];
        c.ple_head_offsets = [
            0,
            20_000_003,
            40_000_026,
            60_000_059,
            80_000_106,
            100_000_165,
            120_000_228,
            140_000_297,
            160_000_374,
            180_000_455,
            200_000_548,
            220_000_655,
            240_000_802,
            260_000_955,
            280_001_114,
            300_001_275,
        ];
        c.ple_head_vocab = [
            20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
            20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
            20_000_161, 20_000_171,
        ];
        c.ple_rows = 320_001_446;
        c
    }

    /// Rows computed independently (Python, llama.cpp's hash) for
    /// `[<|im_start|>, "user", "\n", EOS, "Hello", ","]`; the EOS at position
    /// 3 cuts the window (`ngram_test.cpp`).
    #[test]
    fn hashes_match_the_independent_rows() {
        let c = flash_next_ple();
        let tokens = [151_644, 872, 198, 248_044, 9707, 11];
        let expected: [[u32; 16]; 6] = [
            [
                9_213_577,
                25_503_567,
                59_963_735,
                67_280_845,
                97_263_283,
                108_604_657,
                126_879_742,
                153_604_170,
                163_991_649,
                192_963_728,
                210_734_887,
                224_850_479,
                244_505_954,
                265_606_139,
                292_960_741,
                312_140_635,
            ],
            [
                6_954_764,
                37_428_000,
                54_939_688,
                74_004_108,
                95_568_253,
                116_575_022,
                138_540_160,
                142_009_604,
                165_349_151,
                193_458_441,
                202_840_010,
                231_812_024,
                252_625_381,
                274_082_424,
                294_711_253,
                318_927_564,
            ],
            [
                6_746_895,
                31_942_778,
                54_550_641,
                78_212_738,
                81_361_934,
                102_413_837,
                123_993_646,
                146_103_739,
                175_246_506,
                190_873_805,
                206_877_969,
                222_022_136,
                242_132_242,
                262_461_054,
                282_619_366,
                303_774_792,
            ],
            [
                16_849_591,
                37_537_304,
                42_290_399,
                73_883_125,
                88_405_327,
                114_186_760,
                123_740_665,
                158_125_394,
                169_423_291,
                194_333_347,
                208_742_846,
                238_789_911,
                242_351_884,
                266_971_822,
                295_413_699,
                319_385_148,
            ],
            [
                16_410_909,
                39_682_429,
                55_103_279,
                60_931_720,
                87_006_904,
                116_506_179,
                131_512_017,
                152_932_897,
                169_641_436,
                182_022_480,
                209_277_433,
                237_891_023,
                256_841_529,
                277_007_269,
                290_665_954,
                300_984_276,
            ],
            [
                18_158_303,
                36_390_029,
                45_652_312,
                70_783_524,
                98_191_154,
                114_024_957,
                127_804_916,
                159_566_246,
                175_132_467,
                192_583_600,
                217_919_476,
                237_199_054,
                259_336_807,
                261_799_367,
                289_359_313,
                307_699_687,
            ],
        ];
        let mut history = NgramHistory::default();
        let mut rows = vec![0u32; tokens.len() * 16];
        hash_ngram_rows(&c, &mut history, &tokens, &mut rows);
        assert_eq!(rows, expected.as_flattened());
        assert_eq!(history.prev, [11, 9707]);
        let mut step = NgramHistory::default();
        for (token, expected) in tokens.iter().zip(&expected) {
            let mut one = [0u32; 16];
            hash_ngram_rows(&c, &mut step, std::slice::from_ref(token), &mut one);
            assert_eq!(&one, expected, "single-step hash row");
        }
    }

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("gufo-ngram-{name}-{}.bin", std::process::id()))
    }

    fn bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn truncated(value: f32) -> f32 {
        f32::from_bits(value.to_bits() & 0xffff_0000)
    }

    fn gather(table: &NgramTable, ids: &[u32], out: &mut [f32]) -> bool {
        table.start_read(ids) && table.wait_read(out)
    }

    /// A BF16 table at an unaligned offset where row `r` holds `r + i/1000`
    /// in column `i`: gathers return exactly those values, duplicates and
    /// shuffled wide batches included; a failed read leaves the readers
    /// usable (`TestTable`).
    #[test]
    fn bf16_rows_gather_exactly_and_recover_from_failed_reads() {
        let path = temp("bf16");
        let (dim, rows, offset) = (160usize, 2049u32, 4096 + 90u64);
        let mut bytes = vec![0u8; offset as usize];
        for r in 0..rows {
            for i in 0..dim {
                let value = r as f32 + i as f32 / 1000.0;
                bytes.extend_from_slice(&bf16(value).to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let expect = |r: u32, i: usize| truncated(r as f32 + i as f32 / 1000.0);
        let file = File::open(&path).unwrap();
        let table =
            NgramTable::open(&file, offset, u64::from(rows), dim as u32, GgmlType::Bf16).unwrap();
        let ids = [7, rows - 1, 7, 0, 128, rows - 1];
        let mut out = vec![0f32; ids.len() * dim];
        assert!(gather(&table, &ids, &mut out));
        for (j, &r) in ids.iter().enumerate() {
            for i in 0..dim {
                assert_eq!(out[j * dim + i], expect(r, i), "row {r} column {i}");
            }
        }
        let expected = out.clone();
        assert!(!table.start_read(&[rows]), "out-of-range row rejected");
        assert!(table.start_read(&ids));
        assert!(!table.start_read(&ids), "overlapping gather rejected");
        out.fill(-1.0);
        assert!(table.wait_read(&mut out));
        assert_eq!(out, expected, "asynchronous rows and duplicates match");
        assert!(!table.wait_read(&mut out), "a gather is consumed once");
        assert!(gather(&table, &[], &mut []), "empty gather completes");
        assert!(table.start_read(&ids));
        assert!(!table.wait_read(&mut out[..dim]), "short output rejected");
        assert!(gather(&table, &ids, &mut out), "the reader stays usable");
        // More distinct rows than the batching threshold, shuffled with
        // duplicates: file-order reads keep every output slot.
        let mut wide: Vec<u32> = (0..rows as usize * 2 + 17)
            .map(|i| ((i * 109 + 19) % rows as usize) as u32)
            .collect();
        let guard = 16;
        let mut values = vec![-123.0f32; wide.len() * dim + guard];
        let size = wide.len() * dim;
        assert!(gather(&table, &wide, &mut values[..size]));
        for (j, &r) in wide.iter().enumerate() {
            for i in 0..dim {
                assert_eq!(values[j * dim + i], expect(r, i), "batched slot");
            }
        }
        assert!(
            values[size..].iter().all(|&v| v == -123.0),
            "batched output guard"
        );
        // Dropping a reader with an outstanding gather joins its workers.
        assert!(table.start_read(&ids));
        drop(table);
        // Valid metadata over a truncated file fails at the read and leaves
        // the readers usable.
        let table = NgramTable::open(
            &file,
            offset,
            u64::from(rows) + 1,
            dim as u32,
            GgmlType::Bf16,
        )
        .unwrap();
        assert!(
            !gather(&table, &[rows], &mut out),
            "truncated row read fails"
        );
        assert!(
            gather(&table, &ids, &mut out),
            "the reader recovers after failed I/O"
        );
        assert_eq!(out, expected, "failed I/O does not corrupt later rows");
        wide[333] = rows;
        assert!(
            !gather(&table, &wide, &mut values[..size]),
            "failed batched read"
        );
        assert!(
            gather(&table, &ids, &mut out),
            "the reader recovers after a batched failure"
        );
        assert_eq!(out, expected);
        drop(table);
        let _ = std::fs::remove_file(&path);
    }

    /// IQ4_NL rows decode to exact values on cold reads, cached gathers and
    /// mixed gathers, and cached rows no longer need the file
    /// (`TestQuantizedRows`).
    #[test]
    fn iq4_rows_decode_exactly_from_the_file_and_the_cache() {
        const VALUES: [i32; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        let path = temp("iq4");
        let (dim, rows, offset) = (160u32, 129u32, 4096 + 90usize);
        let mut bytes = vec![0u8; offset];
        for row in 0..rows {
            for block in 0..dim / 32 {
                // Exactly representable half scales: 0.5 and -2.
                let scale: u16 = if row % 2 == 0 { 0x3800 } else { 0xC000 };
                bytes.extend_from_slice(&scale.to_le_bytes());
                for col in 0..16 {
                    bytes.push(
                        (((row + block + col) % 16) | (((row + block + col + 7) % 16) << 4)) as u8,
                    );
                }
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let table =
            NgramTable::open(&file, offset as u64, u64::from(rows), dim, GgmlType::Iq4Nl).unwrap();
        let check = |ids: &[u32]| {
            let mut out = vec![-123.0f32; ids.len() * dim as usize + 16];
            let size = ids.len() * dim as usize;
            assert!(
                gather(&table, ids, &mut out[..size]),
                "IQ4_NL gather completes"
            );
            for (i, &r) in ids.iter().enumerate() {
                for col in 0..dim {
                    let index = (r + col / 32 + col % 16 + if col % 32 >= 16 { 7 } else { 0 }) % 16;
                    let scale = if r % 2 == 0 { 0.5 } else { -2.0 };
                    assert_eq!(
                        out[i * dim as usize + col as usize],
                        scale * VALUES[index as usize] as f32
                    );
                }
            }
            assert!(
                out[size..].iter().all(|&v| v == -123.0),
                "IQ4_NL output guard"
            );
        };
        let warm = [0, 7, 128, 7, 0, 32];
        check(&warm);
        check(&warm);
        let mixed = [128, 3, 7, 64, 3, 32, 1];
        check(&mixed);
        // Cached rows do not need the backing file after successful reads.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        check(&warm);
        check(&mixed);
        drop(table);
        let _ = std::fs::remove_file(&path);
    }

    /// Widely spaced rows share cache slots: eviction under simultaneous
    /// readers keeps exact bytes, and an incomplete row is not cached
    /// (`TestDistantRows`).
    #[test]
    fn colliding_rows_survive_cache_eviction() {
        let path = temp("distant");
        let (dim, stride, offset) = (160usize, 65_536u32, 4096 + 90usize);
        let last = 7 * stride;
        // A sparse file: only the eight rows are written.
        {
            use std::os::unix::fs::FileExt;
            let file = File::create(&path).unwrap();
            for r in 0..8u32 {
                let row: Vec<u8> = (0..dim)
                    .flat_map(|col| bf16(r as f32 + 1.25 + (col % 16) as f32 / 16.0).to_le_bytes())
                    .collect();
                file.write_all_at(&row, (offset + (r * stride) as usize * dim * 2) as u64)
                    .unwrap();
            }
        }
        let file = File::open(&path).unwrap();
        let table = NgramTable::open(
            &file,
            offset as u64,
            u64::from(last) + 2,
            dim as u32,
            GgmlType::Bf16,
        )
        .unwrap();
        let mut out = vec![-123.0f32; 80 * dim + 16];
        for round in 0..4 {
            let ids: Vec<u32> = (0..80)
                .map(|i| ((i * 3 + round) % 8) as u32 * stride)
                .collect();
            assert!(
                gather(&table, &ids, &mut out[..80 * dim]),
                "colliding gather completes"
            );
            for (i, &r) in ids.iter().enumerate() {
                for col in 0..dim {
                    let expected = (r / stride) as f32 + 1.25 + (col % 16) as f32 / 16.0;
                    assert_eq!(
                        out[i * dim + col],
                        expected,
                        "evicted row keeps exact bytes"
                    );
                }
            }
            assert!(
                !gather(&table, &[last + 1], &mut out),
                "incomplete row is not cached"
            );
        }
        assert!(
            out[80 * dim..].iter().all(|&v| v == -123.0),
            "colliding gather guard"
        );
        drop(table);
        let _ = std::fs::remove_file(&path);
    }

    /// The table stays bound to the file it was opened on after the path
    /// is replaced and the caller's descriptor closed (`TestReplacedPath`).
    #[test]
    fn the_table_reads_the_file_it_was_opened_on() {
        let path = temp("bound");
        std::fs::write(&path, 0x3f80u16.to_le_bytes()).unwrap();
        let file = File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, 0x4000u16.to_le_bytes()).unwrap();
        let table = NgramTable::open(&file, 0, 1, 1, GgmlType::Bf16).unwrap();
        drop(file);
        let mut out = [0f32; 1];
        assert!(gather(&table, &[0], &mut out) && out[0] == 1.0);
        drop(table);
        let _ = std::fs::remove_file(&path);
    }
}
