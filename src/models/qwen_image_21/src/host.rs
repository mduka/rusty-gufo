//! Host loops over large tables, split across the cores. Every element is
//! computed exactly as a serial loop would (independent elements, or
//! integer sums), so the results do not depend on the split.

/// Runs `fill(first_row, rows)` over consecutive chunks of `rows` rows of
/// `width` elements each, on up to one thread per core.
pub fn fill_rows<T: Send>(data: &mut [T], width: usize, fill: impl Fn(usize, &mut [T]) + Sync) {
    let rows = data.len().checked_div(width).unwrap_or(0);
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(rows.max(1));
    // Small tables are not worth a thread.
    if threads <= 1 || data.len() < 1 << 16 {
        fill(0, data);
        return;
    }
    let chunk = rows.div_ceil(threads) * width;
    std::thread::scope(|scope| {
        for (index, part) in data.chunks_mut(chunk).enumerate() {
            let fill = &fill;
            scope.spawn(move || fill(index * chunk / width, part));
        }
    });
}
