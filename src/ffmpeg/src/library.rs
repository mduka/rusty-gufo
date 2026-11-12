//! One runtime library opened on demand (`platform::Library`).

use std::ffi::{CStr, CString, c_void};
use std::path::Path;
use std::ptr::NonNull;

use crate::Error;

/// A library handle from `dlopen`, closed on drop. Symbols taken from it
/// must not outlive it.
pub(crate) struct Library {
    handle: NonNull<c_void>,
}

// SAFETY: the handle is an opaque token of the dynamic loader, whose
// functions (`dlsym`, `dlclose`) are thread-safe; it holds no thread-bound
// state.
unsafe impl Send for Library {}
// SAFETY: as above; `&Library` only allows `dlsym`.
unsafe impl Sync for Library {}

/// The loader's message for the last failure, or an empty string.
fn last_error() -> String {
    // SAFETY: `dlerror` returns null or a NUL-terminated string that stays
    // valid until the next loader call on this thread; it is copied at once.
    let message = unsafe { libc::dlerror() };
    if message.is_null() {
        return String::new();
    }
    // SAFETY: non-null, NUL-terminated (see above).
    unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

fn open(name: &str) -> Option<NonNull<c_void>> {
    let name = CString::new(name).ok()?;
    // SAFETY: `name` is NUL-terminated and outlives the call. Loading runs
    // the library's initialisers, which is what opening it means.
    NonNull::new(unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) })
}

impl Library {
    /// Opens `name`. A relocated package supplies its library directories
    /// through the ELF loader, so an absolute build-host path that is absent
    /// falls back to its file name.
    pub(crate) fn open(name: &str) -> Result<Self, Error> {
        let path = Path::new(name);
        let handle = open(name).or_else(|| {
            path.is_absolute()
                .then(|| path.file_name().and_then(|file| file.to_str()))
                .flatten()
                .and_then(open)
        });
        handle.map(|handle| Self { handle }).ok_or_else(|| {
            Error::Library(format!(
                "cannot load runtime library {name}: {}",
                last_error()
            ))
        })
    }

    /// The address of `name`, as the function pointer type `T`.
    ///
    /// # Safety
    /// `T` must be an `extern "C"` function pointer type matching the C
    /// declaration of `name`, and the pointer must not be called after this
    /// library is dropped.
    pub(crate) unsafe fn symbol<T: Copy>(&self, name: &CStr) -> Result<T, Error> {
        const { assert!(size_of::<T>() == size_of::<*mut c_void>()) };
        // SAFETY: clears the thread's loader error before the lookup.
        let _ = unsafe { libc::dlerror() };
        // SAFETY: the handle is open and `name` is NUL-terminated.
        let address = unsafe { libc::dlsym(self.handle.as_ptr(), name.as_ptr()) };
        let error = last_error();
        let name = name.to_string_lossy();
        if !error.is_empty() {
            return Err(Error::Library(format!(
                "missing runtime symbol {name}: {error}"
            )));
        }
        if address.is_null() {
            return Err(Error::Library(format!("missing runtime symbol {name}")));
        }
        // SAFETY: a non-null symbol address of the size of `T`, which the
        // caller guarantees is the symbol's function pointer type.
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&address) })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: the handle came from `dlopen` and is closed once; symbols
        // do not outlive the library (`symbol`'s contract).
        unsafe { libc::dlclose(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_libraries_name_the_library() {
        let Err(Error::Library(message)) = Library::open("libgufo-missing-library.so.0") else {
            panic!("a missing library opened");
        };
        assert!(
            message.starts_with("cannot load runtime library libgufo-missing-library.so.0: "),
            "{message}"
        );
        // An absent build-host path falls back to the file name, which is
        // absent as well.
        let Err(Error::Library(message)) =
            Library::open("/gufo-missing-build-path/libgufo-missing-library.so.0")
        else {
            panic!("a missing library opened");
        };
        assert!(
            message.contains("libgufo-missing-library.so.0"),
            "{message}"
        );
    }

    #[test]
    fn libraries_resolve_symbols_and_refuse_missing_ones() {
        let library = Library::open("libc.so.6").expect("libc opens");
        // SAFETY: `strlen` has this signature.
        let strlen: unsafe extern "C" fn(*const libc::c_char) -> usize =
            unsafe { library.symbol(c"strlen") }.expect("strlen resolves");
        // SAFETY: a NUL-terminated string.
        assert_eq!(unsafe { strlen(c"gufo".as_ptr()) }, 4);
        // SAFETY: the type matches whatever the (absent) symbol would be.
        let missing = unsafe { library.symbol::<unsafe extern "C" fn()>(c"gufo_missing_symbol") };
        let Err(Error::Library(message)) = missing else {
            panic!("a missing symbol resolved");
        };
        assert!(
            message.starts_with("missing runtime symbol gufo_missing_symbol"),
            "{message}"
        );
    }
}
