//! Host facts from `/proc`, `/sys` and `uname` (`linux_sysfs.cpp`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCpuInfo {
    pub model_name: String,
    pub vendor_id: String,
    pub architecture: String,
    pub logical_cores: u32,
    pub physical_cores: u32,
    pub numa_nodes: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostMemInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub free_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SysfsGpuDevice {
    pub card_name: String,
    pub pci_id: String,
    pub pci_slot: String,
    pub driver: String,
    pub vendor_id: String,
    pub device_id: String,
    pub power_mode: String,
    /// The current core and memory clock levels (`pp_dpm_sclk`,
    /// `pp_dpm_mclk`), e.g. "sclk 2900Mhz, mclk 1000Mhz".
    pub clock_state: Option<String>,
    pub hwmon_sensors: BTreeMap<String, String>,
}

/// Reads below a sysfs and a procfs root, so tests can supply fixtures.
pub struct LinuxSysfs {
    sys_root: PathBuf,
    proc_root: PathBuf,
}

impl Default for LinuxSysfs {
    fn default() -> Self {
        Self::new(Path::new("/sys"), Path::new("/proc"))
    }
}

fn trim(text: &str) -> &str {
    text.trim_matches([' ', '\t', '\r', '\n'])
}

/// `std::from_chars` into `uint32_t`: the leading digits, or 0 when there
/// are none or they overflow.
fn parse_uint(text: &str) -> u32 {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    text[..end].parse().unwrap_or(0)
}

/// `std::istream >> std::uint64_t`: an optional sign and digits; a negative
/// value wraps as in `strtoull`, and none or an overflow fail.
fn stream_u64(text: &str) -> Option<u64> {
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    let value: u64 = digits[..end].parse().ok()?;
    Some(if negative {
        value.wrapping_neg()
    } else {
        value
    })
}

impl LinuxSysfs {
    pub fn new(sys_root: &Path, proc_root: &Path) -> Self {
        let absolute = |path: &Path| std::path::absolute(path).unwrap_or_else(|_| path.into());
        Self {
            sys_root: absolute(sys_root),
            proc_root: absolute(proc_root),
        }
    }

    /// The trimmed contents of `path`, or of `path` below the sysfs root when
    /// `path` does not exist.
    pub fn read_file(&self, path: &Path) -> Option<String> {
        let target = if path.exists() {
            path.to_path_buf()
        } else {
            self.sys_root.join(path)
        };
        let bytes = fs::read(target).ok()?;
        Some(trim(&String::from_utf8_lossy(&bytes)).to_string())
    }

    /// The kernel release of `uname`.
    pub fn kernel_release(&self) -> Option<String> {
        Some(
            rustix::system::uname()
                .release()
                .to_string_lossy()
                .into_owned(),
        )
    }

    pub fn query_cpu_info(&self) -> Option<HostCpuInfo> {
        let content = self.read_file(&self.proc_root.join("cpuinfo"))?;
        let mut info = HostCpuInfo {
            model_name: String::new(),
            vendor_id: String::new(),
            architecture: "x86_64".into(),
            logical_cores: 0,
            physical_cores: 0,
            numa_nodes: 1,
        };
        let mut processors = 0u32;
        let mut physical_ids = BTreeSet::new();
        for line in content.split('\n') {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let (key, value) = (trim(key), trim(value));
            match key {
                "model name" if info.model_name.is_empty() => info.model_name = value.into(),
                "vendor_id" if info.vendor_id.is_empty() => info.vendor_id = value.into(),
                "processor" => processors += 1,
                "siblings" if info.logical_cores == 0 => info.logical_cores = parse_uint(value),
                "cpu cores" if info.physical_cores == 0 => {
                    info.physical_cores = parse_uint(value);
                }
                "physical id" => {
                    physical_ids.insert(value.to_string());
                }
                _ => {}
            }
        }
        if info.logical_cores == 0 {
            info.logical_cores = processors;
        }
        if info.physical_cores == 0 {
            info.physical_cores = processors / 2;
        }
        info.numa_nodes = if physical_ids.is_empty() {
            1
        } else {
            physical_ids.len() as u32
        };
        Some(info)
    }

    pub fn query_mem_info(&self) -> Option<HostMemInfo> {
        let content = self.read_file(&self.proc_root.join("meminfo"))?;
        let mut info = HostMemInfo::default();
        for line in content.split('\n') {
            let Some((key, rest)) = line.split_once(':') else {
                continue;
            };
            let Some(kib) = stream_u64(trim(rest)) else {
                continue;
            };
            let bytes = kib.wrapping_mul(1024);
            match trim(key) {
                "MemTotal" => info.total_bytes = bytes,
                "MemAvailable" => info.available_bytes = bytes,
                "MemFree" => info.free_bytes = bytes,
                _ => {}
            }
        }
        Some(info)
    }

    /// The state of kernel module `name`: "loaded" when it is live,
    /// "built-in" when the kernel has it without a module, else "not
    /// loaded" or the module's transitional state.
    pub fn module_state(&self, name: &str) -> String {
        let module = self.sys_root.join("module").join(name);
        match self.read_file(&module.join("initstate")) {
            Some(state) if state == "live" => "loaded".into(),
            Some(state) => state,
            None if module.is_dir() => "built-in".into(),
            None => "not loaded".into(),
        }
    }

    /// The level marked current in a `pp_dpm_*` table, e.g. "2900Mhz".
    fn current_level(&self, path: &Path) -> Option<String> {
        let table = self.read_file(path)?;
        table.split('\n').find_map(|line| {
            let level = trim(line.strip_suffix('*')?);
            let (_, value) = level.split_once(':')?;
            Some(trim(value).to_string())
        })
    }

    /// The DRM cards (`cardN` without connector suffix) in directory order.
    pub fn query_drm_gpu_devices(&self) -> Vec<SysfsGpuDevice> {
        let mut devices = Vec::new();
        let Ok(entries) = fs::read_dir(self.sys_root.join("class/drm")) else {
            return devices;
        };
        for entry in entries {
            let Ok(entry) = entry else {
                break;
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let device = entry.path().join("device");
            if !device.exists() {
                continue;
            }
            let mut gpu = SysfsGpuDevice {
                card_name: name,
                vendor_id: self.read_file(&device.join("vendor")).unwrap_or_default(),
                device_id: self.read_file(&device.join("device")).unwrap_or_default(),
                ..SysfsGpuDevice::default()
            };
            if let Some(uevent) = self.read_file(&device.join("uevent")) {
                for line in uevent.split('\n') {
                    if let Some(driver) = line.strip_prefix("DRIVER=") {
                        gpu.driver = driver.into();
                    } else if let Some(pci_id) = line.strip_prefix("PCI_ID=") {
                        gpu.pci_id = pci_id.into();
                    } else if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                        gpu.pci_slot = slot.into();
                    }
                }
            }
            gpu.power_mode = self
                .read_file(&device.join("power_dpm_force_performance_level"))
                .unwrap_or_else(|| "unavailable (unreadable or not exposed)".into());
            let clocks: Vec<String> = [("sclk", "pp_dpm_sclk"), ("mclk", "pp_dpm_mclk")]
                .iter()
                .filter_map(|(label, file)| {
                    Some(format!(
                        "{label} {}",
                        self.current_level(&device.join(file))?
                    ))
                })
                .collect();
            gpu.clock_state = (!clocks.is_empty()).then(|| clocks.join(", "));
            if let Ok(monitors) = fs::read_dir(device.join("hwmon")) {
                for monitor in monitors {
                    let Ok(monitor) = monitor else {
                        break;
                    };
                    let Some(input) = self.read_file(&monitor.path().join("temp1_input")) else {
                        continue;
                    };
                    let millicelsius = parse_uint(&input);
                    if millicelsius > 0 {
                        // std::to_string(double) is "%f"; C++ keeps five characters.
                        let celsius = format!("{:.6}", f64::from(millicelsius) / 1000.0);
                        gpu.hwmon_sensors
                            .insert("edge_temperature_c".into(), celsius[..5].to_string());
                    }
                }
            }
            devices.push(gpu);
        }
        devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, path: &str, content: &str) {
        let path = root.join(path);
        fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        fs::write(path, content).expect("write");
    }

    #[test]
    fn cpu_memory_and_cards_parse_like_cpp() {
        let root = tempfile::tempdir().expect("tempdir");
        let (sys, proc) = (root.path().join("sys"), root.path().join("proc"));
        write(
            &proc,
            "cpuinfo",
            "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: Test CPU\n\
             physical id\t: 0\nsiblings\t: 4x\ncpu cores\t: 99999999999\n\n\
             processor\t: 1\nmodel name\t: Other\nphysical id\t: 1\n",
        );
        write(
            &proc,
            "meminfo",
            "MemTotal:       1000 kB\nMemFree:  +5 kB\nMemAvailable:   250 kB\nBogus\n",
        );
        write(&sys, "class/drm/card1/device/vendor", "0x1002\n");
        write(
            &sys,
            "class/drm/card1/device/uevent",
            "DRIVER=amdgpu\nPCI_ID=1002:1586\nPCI_SLOT_NAME=0000:c5:00.0\n",
        );
        write(
            &sys,
            "class/drm/card1/device/hwmon/hwmon3/temp1_input",
            "104250\n",
        );
        write(
            &sys,
            "class/drm/card1/device/pp_dpm_sclk",
            "0: 600Mhz \n1: 1786Mhz *\n2: 2900Mhz \n",
        );
        write(&sys, "class/drm/card1-DP-1/device/vendor", "x");
        write(&sys, "class/drm/renderD128/device/vendor", "x");
        let sysfs = LinuxSysfs::new(&sys, &proc);

        let cpu = sysfs.query_cpu_info().expect("cpuinfo");
        assert_eq!(
            (cpu.model_name.as_str(), cpu.vendor_id.as_str()),
            ("Test CPU", "AuthenticAMD")
        );
        // "4x" parses its digits; an overflowing count reads as 0, so half
        // the processors.
        assert_eq!(
            (cpu.logical_cores, cpu.physical_cores, cpu.numa_nodes),
            (4, 1, 2)
        );

        let memory = sysfs.query_mem_info().expect("meminfo");
        assert_eq!(
            (
                memory.total_bytes,
                memory.available_bytes,
                memory.free_bytes
            ),
            (1_024_000, 256_000, 5_120)
        );

        let cards = sysfs.query_drm_gpu_devices();
        assert_eq!(cards.len(), 1);
        let card = &cards[0];
        assert_eq!(
            (
                card.card_name.as_str(),
                card.driver.as_str(),
                card.pci_id.as_str()
            ),
            ("card1", "amdgpu", "1002:1586")
        );
        assert_eq!(card.pci_slot, "0000:c5:00.0");
        assert_eq!(card.power_mode, "unavailable (unreadable or not exposed)");
        // Without `pp_dpm_mclk` only the core clock is known.
        assert_eq!(card.clock_state.as_deref(), Some("sclk 1786Mhz"));
        assert_eq!(card.hwmon_sensors["edge_temperature_c"], "104.2");
        assert!(
            LinuxSysfs::new(&root.path().join("none"), &proc)
                .query_drm_gpu_devices()
                .is_empty()
        );
    }

    #[test]
    fn module_states_distinguish_loaded_built_in_and_absent() {
        let root = tempfile::tempdir().expect("tempdir");
        write(root.path(), "module/amdgpu/initstate", "live\n");
        write(root.path(), "module/ext4/parameters/x", "");
        write(root.path(), "module/going/initstate", "going\n");
        let sysfs = LinuxSysfs::new(root.path(), &root.path().join("proc"));
        assert_eq!(sysfs.module_state("amdgpu"), "loaded");
        assert_eq!(sysfs.module_state("ext4"), "built-in");
        assert_eq!(sysfs.module_state("going"), "going");
        assert_eq!(sysfs.module_state("absent"), "not loaded");
    }
}
