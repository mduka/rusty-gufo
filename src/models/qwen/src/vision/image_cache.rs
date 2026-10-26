//! Reuse of decoded and resized images across requests (upstream
//! `ImageCache`): a conversation resends its earlier images with
//! every turn, and decoding plus the bicubic resize dominated their queue
//! time.

use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

use gufo_core::image::{Image, decode_image};
use gufo_core::text::Result;

use super::prompt::PREPROCESSING;
use super::resize_image;

/// Backend-owned CPU preprocessing reuse, independent of continuation state.
/// Only the encoded bytes and the fixed resize policy decide the RGB pixels;
/// the encoder identity still joins each prompt's continuation identity. The
/// budget counts retained RGB and encoded bytes; entries are bounded too, as
/// every resized image holds at least 65,536 pixels. Eviction drops only the
/// cache's reference: prompts in flight keep their immutable pixels.
pub struct ImageCache {
    capacity_bytes: usize,
    inner: Mutex<Inner>,
}

/// What [`ImageCache::stats`] reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageCacheStats {
    pub entries: usize,
    pub pixel_bytes: usize,
    pub encoded_bytes: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

struct Entry {
    key: u64,
    /// Exact comparison makes hash collisions harmless; counting these bytes
    /// bounds reuse for poorly compressible uploads too.
    encoded: Arc<Vec<u8>>,
    image: Arc<Image>,
}

#[derive(Default)]
struct Inner {
    /// Most recently used first.
    entries: VecDeque<Entry>,
    stats: ImageCacheStats,
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAPACITY_BYTES)
    }
}

impl ImageCache {
    pub const DEFAULT_CAPACITY_BYTES: usize = 128 << 20;

    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// The resized RGB image of `encoded`, decoded at most once while cached.
    pub fn get(&self, encoded: &Arc<Vec<u8>>) -> Result<Arc<Image>> {
        let mut hasher = DefaultHasher::new();
        PREPROCESSING.hash(&mut hasher);
        encoded.as_slice().hash(&mut hasher);
        let key = hasher.finish();
        let matches = |entry: &Entry| entry.key == key && entry.encoded == *encoded;
        {
            let mut inner = self.lock();
            if let Some(image) = inner.promote(matches) {
                inner.stats.hits += 1;
                return Ok(image);
            }
            inner.stats.misses += 1;
        }
        // Decode outside the lock so a new upload cannot block earlier hits;
        // a concurrent miss may duplicate the work, one result is kept.
        let image = Arc::new(resize_image(&decode_image(encoded)?)?);
        let pixels = image.pixels.capacity();
        if pixels > self.capacity_bytes || encoded.capacity() > self.capacity_bytes - pixels {
            return Ok(image);
        }
        let bytes = pixels + encoded.capacity();
        let mut inner = self.lock();
        if let Some(image) = inner.promote(matches) {
            return Ok(image);
        }
        while inner.stats.pixel_bytes + inner.stats.encoded_bytes > self.capacity_bytes - bytes {
            let Some(evicted) = inner.entries.pop_back() else {
                break;
            };
            inner.stats.pixel_bytes -= evicted.image.pixels.capacity();
            inner.stats.encoded_bytes -= evicted.encoded.capacity();
            inner.stats.evictions += 1;
        }
        inner.entries.push_front(Entry {
            key,
            encoded: encoded.clone(),
            image: image.clone(),
        });
        inner.stats.pixel_bytes += pixels;
        inner.stats.encoded_bytes += encoded.capacity();
        Ok(image)
    }

    pub fn stats(&self) -> ImageCacheStats {
        let inner = self.lock();
        ImageCacheStats {
            entries: inner.entries.len(),
            ..inner.stats
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Inner {
    /// The image of the entry `matches` selects, moved to the front.
    fn promote(&mut self, matches: impl Fn(&Entry) -> bool) -> Option<Arc<Image>> {
        let at = self.entries.iter().position(matches)?;
        let entry = self.entries.remove(at)?;
        let image = entry.image.clone();
        self.entries.push_front(entry);
        Some(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64x64 PNG of one colour, `shade` in every channel.
    #[allow(clippy::unwrap_used)]
    fn png(shade: u8) -> Arc<Vec<u8>> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 64, 64);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&vec![shade; 64 * 64 * 3]).unwrap();
        }
        Arc::new(bytes)
    }

    /// Repeated bytes hit, other bytes miss, and the byte budget evicts the
    /// least recently used entry (upstream `vision_cache_test`).
    #[test]
    #[allow(clippy::unwrap_used)]
    fn reuses_resized_images_within_the_budget() {
        let (red, blue) = (png(200), png(30));
        let cache = ImageCache::default();
        let first = cache.get(&red).unwrap();
        let again = cache.get(&Arc::new(red.as_ref().clone())).unwrap();
        assert!(Arc::ptr_eq(&first, &again), "equal bytes decoded twice");
        assert_eq!(*first, resize_image(&decode_image(&red).unwrap()).unwrap());
        let other = cache.get(&blue).unwrap();
        assert!(!Arc::ptr_eq(&first, &other), "other bytes shared an entry");
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.hits, stats.misses), (2, 1, 2));

        // Room for one entry: the next image evicts the older one.
        let one = first.pixels.capacity() + red.capacity();
        let small = ImageCache::new(one + one / 2);
        small.get(&red).unwrap();
        small.get(&blue).unwrap();
        let stats = small.stats();
        assert_eq!((stats.entries, stats.evictions), (1, 1));
        // An image larger than the budget is returned, not kept.
        let none = ImageCache::new(16);
        none.get(&red).unwrap();
        assert_eq!(none.stats().entries, 0);
    }
}
