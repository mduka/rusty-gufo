//! The snapshot budget's view of host RAM: `MemAvailable` without the free
//! CMA pages (`TestMeminfoAvailableExcludesFreeCma`) and
//! cgroup limits that count reclaimable page cache as free (`TestCgroupAvailableCountsReclaimableCache`):
//! cargo test -p gufo-core --test host_memory

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::PathBuf;

use gufo_core::text::{cgroup_available_bytes, meminfo_available_bytes};

#[test]
fn free_cma_pages_are_not_available() {
    const KIB: u64 = 1024;
    // Ubuntu 26.04's KHO scratch reports free CMA pages with CmaTotal 0.
    let meminfo = "MemTotal:       131072000 kB\n\
                   MemAvailable:   100000000 kB\n\
                   CmaTotal:               0 kB\n\
                   CmaFree:         13034628 kB\n";
    assert_eq!(
        meminfo_available_bytes(meminfo),
        Some((100_000_000 - 13_034_628) * KIB)
    );
    assert_eq!(
        meminfo_available_bytes("MemAvailable:   8000000 kB\n"),
        Some(8_000_000 * KIB),
        "MemAvailable is used unchanged without a CmaFree line"
    );
    assert_eq!(
        meminfo_available_bytes("CmaFree:  9000 kB\nMemAvailable:  4000 kB\n"),
        Some(0),
        "free CMA beyond MemAvailable saturates at zero"
    );
    assert_eq!(
        meminfo_available_bytes("MemTotal:  4000 kB\nCmaFree:  10 kB\n"),
        None,
        "missing MemAvailable keeps the caller's fallback"
    );
}

/// A fake cgroup v2 tree under the test's temporary directory.
struct Tree(PathBuf);

impl Tree {
    fn write(&self, dir: &str, name: &str, text: &str) {
        fs::create_dir_all(self.0.join(dir)).unwrap();
        fs::write(self.0.join(dir).join(name), format!("{text}\n")).unwrap();
    }

    fn level(&self, dir: &str, max: &str, current: u64, inactive_file: Option<u64>) {
        self.write(dir, "memory.max", max);
        self.write(dir, "memory.current", &current.to_string());
        let _ = fs::remove_file(self.0.join(dir).join("memory.stat"));
        if let Some(inactive) = inactive_file {
            self.write(
                dir,
                "memory.stat",
                &format!("anon 1\ninactive_file {inactive}\nactive_file 5"),
            );
        }
    }

    fn available(&self, host: u64) -> u64 {
        cgroup_available_bytes(&self.0, "0::/pod/ctr\n", host)
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn cgroup_limits_count_reclaimable_page_cache() {
    const GIB: u64 = 1 << 30;
    let tree = Tree(std::env::temp_dir().join(format!("gufo_cgroup_test_{}", std::process::id())));
    let _ = fs::remove_dir_all(&tree.0);
    let host = 500 * GIB;
    let limit = (80 * GIB).to_string();
    let stat = |text: &str| tree.write("pod/ctr", "memory.stat", text);

    tree.level("pod/ctr", &limit, 79 * GIB, Some(32 * GIB));
    assert_eq!(
        tree.available(host),
        33 * GIB,
        "inactive file pages count as available"
    );
    assert_eq!(
        tree.available(10 * GIB),
        10 * GIB,
        "host MemAvailable still bounds the result"
    );

    tree.level("pod/ctr", &limit, 79 * GIB, None);
    assert_eq!(
        tree.available(host),
        GIB,
        "missing memory.stat keeps memory.current as used"
    );
    tree.level("pod/ctr", &limit, 79 * GIB, Some(32 * GIB));
    stat("anon 1\nactive_file 5\n");
    assert_eq!(
        tree.available(host),
        GIB,
        "memory.stat without inactive_file"
    );
    stat(&format!("inactive_file {}\nfile_dirty lots\n", 32 * GIB));
    assert_eq!(
        tree.available(host),
        GIB,
        "a malformed memory.stat value is ignored"
    );
    stat("inactive_file notanumber\n");
    assert_eq!(
        tree.available(host),
        GIB,
        "a malformed inactive_file value is ignored"
    );
    stat(&format!(
        "inactive_file {}\nfile_dirty {}\nfile_writeback {}\n",
        32 * GIB,
        3 * GIB,
        GIB
    ));
    assert_eq!(
        tree.available(host),
        29 * GIB,
        "dirty and writeback pages stay charged"
    );
    stat(&format!(
        "inactive_file {}\nfile_dirty {}\n",
        2 * GIB,
        5 * GIB
    ));
    assert_eq!(
        tree.available(host),
        GIB,
        "dirty pages beyond inactive_file reclaim nothing"
    );

    tree.level("pod/ctr", &limit, 10 * GIB, Some(32 * GIB));
    assert_eq!(
        tree.available(host),
        80 * GIB,
        "inactive file larger than current saturates"
    );

    tree.level("pod/ctr", &limit, 79 * GIB, Some(32 * GIB));
    tree.level("pod", &(40 * GIB).to_string(), 30 * GIB, Some(5 * GIB));
    assert_eq!(
        tree.available(host),
        15 * GIB,
        "a tighter parent limit wins"
    );

    tree.level("pod/ctr", "max", 79 * GIB, Some(32 * GIB));
    assert_eq!(
        tree.available(host),
        15 * GIB,
        "an unlimited level defers to its ancestors"
    );
    fs::remove_dir_all(tree.0.join("pod")).unwrap();
    tree.level("", "max", 79 * GIB, Some(32 * GIB));
    tree.level("pod/ctr", "max", 79 * GIB, Some(32 * GIB));
    assert_eq!(
        tree.available(host),
        host,
        "unlimited cgroups leave host availability unchanged"
    );
}
