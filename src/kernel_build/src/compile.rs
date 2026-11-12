//! Incremental, jobserver-aware hipcc compilation.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;
use std::{env, fs, thread};

use crate::manifest::Source;

/// The only architecture Gufo builds for.
pub const OFFLOAD_ARCH: &str = "gfx1151";

/// Compiles every stale unit and returns all object paths, in order.
/// `rocm` is the prefix gufo-hip links against (`None` falls back to `hipcc`
/// on `PATH`); `$HIPCC` overrides the compiler.
pub fn compile_all(
    units: &[Source],
    root: &Path,
    obj_dir: &Path,
    rocm: Option<&Path>,
) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(obj_dir).map_err(|e| format!("{}: {e}", obj_dir.display()))?;
    let rocm = rocm.map(Path::to_path_buf);
    let hipcc = env::var("HIPCC").unwrap_or_else(|_| match &rocm {
        Some(rocm) => rocm.join("bin/hipcc").display().to_string(),
        None => "hipcc".into(),
    });
    let compiler = compiler_identity(&hipcc)?;
    let gcc = gcc_install_dir()?;
    let jobs: Vec<(Vec<String>, PathBuf)> = units
        .iter()
        .map(|unit| {
            let args = compile_args(rocm.as_deref(), &gcc, unit, root);
            let object = object_path(unit, &compiler, &args, obj_dir);
            (args, object)
        })
        .collect();
    let pending: Vec<&(Vec<String>, PathBuf)> = jobs.iter().filter(|(_, o)| is_stale(o)).collect();

    let tokens = job_tokens();
    let results = run_parallel(&pending, tokens.as_ref(), |(args, object)| {
        compile_one(&hipcc, args, object)
    });
    let failures: Vec<String> = results.into_iter().filter_map(Result::err).collect();
    if !failures.is_empty() {
        return Err(failures.join("\n"));
    }
    for (_, object) in &jobs {
        for dep in depfile_inputs(&object.with_extension("d")) {
            println!("cargo:rerun-if-changed={}", dep.display());
        }
    }
    Ok(jobs.into_iter().map(|(_, object)| object).collect())
}

/// Job slots beyond the build script's implicit one. A token is returned
/// when the box is dropped.
trait Tokens: Sync {
    fn try_acquire(&self) -> Option<Box<dyn Send + '_>>;
}

impl Tokens for jobserver::Client {
    fn try_acquire(&self) -> Option<Box<dyn Send + '_>> {
        // Unsupported (a jobserver not inherited from Cargo) means no extra slots.
        let token = jobserver::Client::try_acquire(self).ok().flatten()?;
        Some(Box::new(token))
    }
}

/// A fixed number of slots, for builds outside Cargo's jobserver.
struct Pool(Mutex<usize>);

struct PoolToken<'a>(&'a Pool);

impl Drop for PoolToken<'_> {
    fn drop(&mut self) {
        if let Ok(mut free) = self.0.0.lock() {
            *free += 1;
        }
    }
}

impl Tokens for Pool {
    fn try_acquire(&self) -> Option<Box<dyn Send + '_>> {
        let mut free = self.0.lock().ok()?;
        *free = free.checked_sub(1)?;
        Some(Box::new(PoolToken(self)))
    }
}

/// Cargo's jobserver, so several kernel crates building at once share `-j`;
/// otherwise `NUM_JOBS - 1` extra slots.
#[allow(unsafe_code)]
fn job_tokens() -> Box<dyn Tokens> {
    // SAFETY: Cargo starts the build script with its jobserver pipe open and
    // names those descriptors in the environment. A build script closes no
    // descriptor it did not open, so they still are that pipe (a file opened
    // since takes a free number), and each build script calls this once, so
    // the client is their only owner.
    if let Some(client) = unsafe { jobserver::Client::from_env() } {
        return Box::new(client);
    }
    let jobs = env::var("NUM_JOBS")
        .ok()
        .and_then(|j| j.parse().ok())
        .unwrap_or(4usize);
    Box::new(Pool(Mutex::new(jobs.saturating_sub(1))))
}

/// Runs `work` on every task and returns the results in task order.
///
/// The calling thread owns the build script's implicit job slot and works
/// through the queue itself, so progress never depends on a token. Extra
/// workers start only for tokens `try_acquire` hands out without blocking,
/// retried between tasks; each worker returns its token when the queue is
/// empty. With `-j 1` everything runs on the calling thread.
fn run_parallel<T, F>(tasks: &[T], tokens: &dyn Tokens, work: F) -> Vec<Result<(), String>>
where
    T: Sync,
    F: Fn(&T) -> Result<(), String> + Sync,
{
    let results: Vec<Mutex<Option<Result<(), String>>>> =
        tasks.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let helpers = AtomicUsize::new(0);
    let drain = || {
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(task) = tasks.get(index) else { break };
            let result = work(task);
            if let Ok(mut slot) = results[index].lock() {
                *slot = Some(result);
            }
        }
    };
    thread::scope(|scope| {
        loop {
            // Start a helper per available token while tasks outnumber workers.
            while tasks.len().saturating_sub(next.load(Ordering::Relaxed))
                > helpers.load(Ordering::Relaxed) + 1
            {
                let Some(token) = tokens.try_acquire() else {
                    break;
                };
                helpers.fetch_add(1, Ordering::Relaxed);
                let (drain, helpers) = (&drain, &helpers);
                scope.spawn(move || {
                    drain();
                    drop(token);
                    helpers.fetch_sub(1, Ordering::Relaxed);
                });
            }
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(task) = tasks.get(index) else { break };
            let result = work(task);
            if let Ok(mut slot) = results[index].lock() {
                *slot = Some(result);
            }
        }
    });
    results
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .ok()
                .flatten()
                .unwrap_or_else(|| Err("hipcc worker panicked".into()))
        })
        .collect()
}

/// Path and `--version` output: switching or upgrading ROCm rebuilds objects.
fn compiler_identity(hipcc: &str) -> Result<String, String> {
    let output = Command::new(hipcc)
        .arg("--version")
        .output()
        .map_err(|e| format!("cannot run {hipcc}: {e}"))?;
    Ok(format!(
        "{hipcc}\n{}",
        String::from_utf8_lossy(&output.stdout)
    ))
}

/// GCC installation of the linker driver (`$CC`, default `cc`). ROCm's clang
/// otherwise picks the newest GCC headers it finds, while the final link uses
/// this GCC's libstdc++; mixing them breaks the libstdc++ ABI (for example
/// C++20 atomic wait/notify symbols).
fn gcc_install_dir() -> Result<PathBuf, String> {
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let output = Command::new(&cc)
        .arg("-print-libgcc-file-name")
        .output()
        .map_err(|e| format!("cannot run {cc}: {e}"))?;
    let libgcc = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    libgcc
        .parent()
        .filter(|dir| dir.join("include").is_dir() || dir.join("libstdc++.so").exists())
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("{cc} -print-libgcc-file-name gave {}", libgcc.display()))
}

/// Every hipcc argument except the per-object depfile and output paths.
fn compile_args(rocm: Option<&Path>, gcc: &Path, unit: &Source, root: &Path) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(rocm) = rocm {
        // Searched before /usr/include, which may hold another ROCm's HIP headers.
        args.push(format!("--rocm-path={}", rocm.display()));
        args.push(format!("-isystem{}", rocm.join("include").display()));
    }
    args.push(format!("--gcc-install-dir={}", gcc.display()));
    // Same optimisation as the CMake release preset (RelWithDebInfo), so A/B
    // runs against the C++ build compare host code, not kernel flags.
    args.extend(["-x", "hip", "-O2", "-g", "-DNDEBUG", "-fPIC"].map(String::from));
    args.push(format!("-std={}", unit.std));
    args.push(format!("--offload-arch={OFFLOAD_ARCH}"));
    args.extend(["-DENGINE_ENABLE_HIP=1", "-Wall", "-Wextra"].map(String::from));
    args.extend(unit.defines.iter().map(|d| format!("-D{d}")));
    args.extend(unit.flags.iter().cloned());
    args.push(format!("-I{}", root.display()));
    args.push("-c".into());
    args.push(unit.path.display().to_string());
    args
}

/// Objects are keyed by compiler identity and the complete command line, so
/// any change of ROCm, flags or include paths rebuilds them.
fn object_path(unit: &Source, compiler: &str, args: &[String], obj_dir: &Path) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    compiler.hash(&mut hasher);
    args.hash(&mut hasher);
    let stem = unit
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unit");
    obj_dir.join(format!("{stem}-{:016x}.o", hasher.finish()))
}

fn compile_one(hipcc: &str, args: &[String], object: &Path) -> Result<(), String> {
    let output = Command::new(hipcc)
        .args(args)
        .arg("-MD")
        .arg("-MF")
        .arg(object.with_extension("d"))
        .arg("-o")
        .arg(object)
        .output()
        .map_err(|e| format!("cannot run {hipcc}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{hipcc} {} failed ({}):\n{}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn is_stale(object: &Path) -> bool {
    let Some(built) = mtime(object) else {
        return true;
    };
    let deps = depfile_inputs(&object.with_extension("d"));
    deps.is_empty() || deps.iter().any(|d| mtime(d).is_none_or(|t| t > built))
}

/// Inputs listed in a Makefile-style depfile written by `-MD -MF`.
fn depfile_inputs(depfile: &Path) -> Vec<PathBuf> {
    let Ok(text) = fs::read_to_string(depfile) else {
        return Vec::new();
    };
    parse_depfile(&text)
}

fn parse_depfile(text: &str) -> Vec<PathBuf> {
    text.replace("\\\n", " ")
        .lines()
        .filter_map(|line| line.split_once(": ").map(|(_, deps)| deps))
        .flat_map(str::split_whitespace)
        .map(PathBuf::from)
        .collect()
}

fn mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `run_parallel` with a jobserver of `tokens` extra slots and
    /// returns the peak concurrency; fails instead of hanging on a deadlock.
    fn peak_concurrency(tokens: usize, tasks: usize) -> usize {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let pool = Pool(Mutex::new(tokens));
            let (active, peak, done) = (
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
            );
            let results = run_parallel(&vec![(); tasks], &pool, |_| {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(20));
                active.fetch_sub(1, Ordering::SeqCst);
                done.fetch_add(1, Ordering::SeqCst);
                Ok(())
            });
            assert!(results.iter().all(Result::is_ok));
            assert_eq!(
                done.load(Ordering::SeqCst),
                tasks,
                "every task runs exactly once"
            );
            assert_eq!(*pool.0.lock().unwrap(), tokens, "every token is returned");
            let _ = sender.send(peak.load(Ordering::SeqCst));
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(20))
            .expect("run_parallel must not deadlock")
    }

    #[test]
    fn a_jobserver_without_tokens_runs_everything_on_the_caller() {
        // Regression: `cargo build -j 1` on a fresh target hung here.
        assert_eq!(peak_concurrency(0, 6), 1);
    }

    #[test]
    fn extra_tokens_add_workers_up_to_the_limit() {
        let peak = peak_concurrency(3, 12);
        assert!((2..=4).contains(&peak), "peak concurrency {peak}");
    }

    #[test]
    fn depfile_continuations_are_joined() {
        let deps = parse_depfile("out.o: a.hip \\\n  include/b.hpp \\\n  c.h\n");
        assert_eq!(
            deps,
            [PathBuf::from("a.hip"), "include/b.hpp".into(), "c.h".into()]
        );
    }

    #[test]
    fn object_names_change_with_compiler() {
        let unit = Source {
            path: "k.hip".into(),
            std: "c++20".into(),
            flags: vec![],
            defines: vec![],
        };
        let key = |compiler: &str| object_path(&unit, compiler, &[], Path::new("o"));
        assert_ne!(
            key("/usr/bin/hipcc\nHIP version: 7.1"),
            key("/opt/rocm/core-10.0/bin/hipcc\nHIP version: 7.15")
        );
    }

    #[test]
    fn object_names_change_with_flags() {
        let base = Source {
            path: "k.hip".into(),
            std: "c++20".into(),
            flags: vec![],
            defines: vec![],
        };
        let wave64 = Source {
            flags: vec!["-mwavefrontsize64".into()],
            ..base.clone()
        };
        let key = |unit: &Source| {
            let args = compile_args(
                Some(Path::new("/opt/rocm")),
                Path::new("gcc"),
                unit,
                Path::new("."),
            );
            object_path(unit, "hipcc", &args, Path::new("o"))
        };
        assert_ne!(key(&base), key(&wave64));
        let other_rocm = compile_args(
            Some(Path::new("/usr")),
            Path::new("gcc"),
            &base,
            Path::new("."),
        );
        assert_ne!(
            key(&base),
            object_path(&base, "hipcc", &other_rocm, Path::new("o"))
        );
    }
}
