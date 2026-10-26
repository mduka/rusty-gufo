//! The store's directory: startup indexing, reading and publishing images,
//! and making room.

use super::*;

impl Inner {
    pub(super) fn emit(
        &self,
        action: ContinuationDiskEventAction,
        reason: ContinuationDiskEventReason,
        file_bytes: usize,
        payload_bytes: usize,
        token_count: usize,
        elapsed_ms: f64,
    ) {
        self.emit_staging(
            action,
            reason,
            file_bytes,
            payload_bytes,
            token_count,
            elapsed_ms,
            0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_staging(
        &self,
        action: ContinuationDiskEventAction,
        reason: ContinuationDiskEventReason,
        file_bytes: usize,
        payload_bytes: usize,
        token_count: usize,
        elapsed_ms: f64,
        staging_used_bytes: usize,
    ) {
        let Some(sink) = &self.event_sink else {
            return;
        };
        let event = ContinuationDiskEvent {
            action,
            reason,
            file_bytes,
            payload_bytes,
            token_count,
            retained_bytes: self.retained.load(Ordering::Relaxed),
            capacity_bytes: self.options.capacity_bytes,
            staging_capacity_bytes: self.options.staging_capacity_bytes,
            staging_used_bytes,
            elapsed_ms,
        };
        // Observability must never affect request execution.
        let _ = catch_unwind(AssertUnwindSafe(|| sink(&event)));
    }

    pub(super) fn hash(&self, bytes: &[u8]) -> Result<String> {
        let digest = match &self.key_hash {
            Some(hash) => hash(bytes),
            None => sha256_hex(bytes),
        };
        if !is_lower_hex_digest(digest.as_bytes()) {
            return Err(Error::runtime(
                "continuation disk key hash must be 64 lowercase hex characters",
            ));
        }
        Ok(digest)
    }

    /// A plain file name in the directory that is a regular, owner-only,
    /// owner-readable file of the effective user; symlinks are not followed.
    pub(super) fn safe_regular_file(&self, filename: &OsStr) -> Option<Stat> {
        let bytes = filename.as_bytes();
        if bytes.is_empty() || bytes.contains(&b'/') || bytes.windows(2).any(|w| w == b"..") {
            return None;
        }
        let status =
            rustix::fs::statat(&self.directory, filename, AtFlags::SYMLINK_NOFOLLOW).ok()?;
        let safe = FileType::from_raw_mode(status.st_mode) == FileType::RegularFile
            && status.st_uid == rustix::process::geteuid().as_raw()
            && status.st_mode & 0o077 == 0
            && status.st_mode & 0o400 != 0;
        safe.then_some(status)
    }

    /// The whole file, at most the staging capacity, and its modification
    /// time; the file must not grow while it is read.
    pub(super) fn read_image(
        &self,
        filename: &OsStr,
    ) -> std::result::Result<(Vec<u8>, FileTime), ContinuationDiskEventReason> {
        let status = self
            .safe_regular_file(filename)
            .ok_or(ContinuationDiskEventReason::UnsafeFile)?;
        let size = usize::try_from(status.st_size).unwrap_or(0);
        if size == 0 || size > self.options.staging_capacity_bytes {
            return Err(if status.st_size > 0 {
                ContinuationDiskEventReason::StagingCapacity
            } else {
                ContinuationDiskEventReason::Corrupt
            });
        }
        let descriptor = rustix::fs::openat(
            &self.directory,
            filename,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .map_err(|_| ContinuationDiskEventReason::IoFailure)?;
        let mut file = File::from(descriptor);
        let mut image = vec![0; size];
        file.read_exact(&mut image)
            .map_err(|_| ContinuationDiskEventReason::IoFailure)?;
        let mut trailing = [0];
        loop {
            match file.read(&mut trailing) {
                Ok(0) => break,
                Ok(_) => return Err(ContinuationDiskEventReason::Corrupt),
                Err(error) if error.kind() == IoErrorKind::Interrupted => continue,
                Err(_) => return Err(ContinuationDiskEventReason::IoFailure),
            }
        }
        Ok((image, modification_time(&status)))
    }

    /// The header, identity and tokens of a file and its modification
    /// time, without reading or verifying the payload (upstream's
    /// metadata-only startup); restores verify the whole file. A file the
    /// staging capacity cannot hold reports `StagingCapacity`.
    pub(super) fn read_metadata(
        &self,
        filename: &OsStr,
    ) -> std::result::Result<(ParsedImage, usize, FileTime), ContinuationDiskEventReason> {
        let status = self
            .safe_regular_file(filename)
            .ok_or(ContinuationDiskEventReason::UnsafeFile)?;
        let size = usize::try_from(status.st_size).unwrap_or(0);
        if size == 0 || size > self.options.staging_capacity_bytes {
            return Err(if status.st_size > 0 {
                ContinuationDiskEventReason::StagingCapacity
            } else {
                ContinuationDiskEventReason::Corrupt
            });
        }
        if size < HEADER_BYTES {
            return Err(ContinuationDiskEventReason::Corrupt);
        }
        let descriptor = rustix::fs::openat(
            &self.directory,
            filename,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .map_err(|_| ContinuationDiskEventReason::IoFailure)?;
        let mut file = File::from(descriptor);
        let mut metadata = vec![0; HEADER_BYTES];
        file.read_exact(&mut metadata)
            .map_err(|_| ContinuationDiskEventReason::IoFailure)?;
        let header =
            parse_header(&metadata, size).map_err(|_| ContinuationDiskEventReason::Corrupt)?;
        metadata.resize(header.metadata_bytes(), 0);
        file.read_exact(&mut metadata[HEADER_BYTES..])
            .map_err(|_| ContinuationDiskEventReason::IoFailure)?;
        Ok((
            parse_metadata(&metadata, &header),
            size,
            modification_time(&status),
        ))
    }

    pub(super) fn initialize_directory(options: &ContinuationDiskStoreOptions) -> Result<OwnedFd> {
        let directory = &options.directory;
        let existed = directory
            .try_exists()
            .map_err(|_| Error::runtime("failed to inspect continuation disk directory"))?;
        if !existed {
            std::fs::create_dir_all(directory)
                .map_err(|_| Error::runtime("failed to create continuation disk directory"))?;
        }
        let status = rustix::fs::lstat(directory).map_err(|e| {
            Error::runtime(errno_message(
                "failed to inspect continuation disk directory",
                e,
            ))
        })?;
        if FileType::from_raw_mode(status.st_mode) != FileType::Directory
            || status.st_uid != rustix::process::geteuid().as_raw()
        {
            return Err(Error::runtime(
                "continuation disk directory is not a private owned directory",
            ));
        }
        rustix::fs::chmod(directory, Mode::RWXU).map_err(|e| {
            Error::runtime(errno_message(
                "failed to secure continuation disk directory",
                e,
            ))
        })?;
        rustix::fs::open(
            directory,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .map_err(|e| {
            Error::runtime(errno_message(
                "failed to open continuation disk directory",
                e,
            ))
        })
    }

    /// Removes interrupted temporary files and invalid or unsafe entries,
    /// keeps the newest of exact duplicates and indexes the rest. Only
    /// metadata is read: a payload checksum failure surfaces at its first
    /// restore, which removes the file and tries the next candidate.
    pub(super) fn index_existing_files(&self, metadata: &mut Metadata) -> Result<()> {
        let enumerate = || Error::runtime("failed to enumerate continuation disk directory");
        for item in std::fs::read_dir(&self.options.directory).map_err(|_| enumerate())? {
            let filename = item.map_err(|_| enumerate())?.file_name();
            let bytes = filename.as_bytes();
            if bytes.starts_with(TEMPORARY_PREFIX.as_bytes()) {
                // Publishers hold a shared directory lock from before their
                // temporary file exists through its rename: defer orphan
                // cleanup while one is active, without waiting for another
                // store's long write.
                if DirectoryLock::acquire(
                    &self.directory,
                    rustix::fs::FlockOperation::NonBlockingLockExclusive,
                )
                .is_some()
                {
                    self.remove_file_only(&filename);
                }
                continue;
            }
            if !bytes.ends_with(FILE_SUFFIX.as_bytes()) {
                continue;
            }

            let (parsed, file_bytes, last_access) = match self.read_metadata(&filename) {
                Ok(read) => read,
                Err(ContinuationDiskEventReason::StagingCapacity) => {
                    // A smaller RAM budget must not destroy a valid cache:
                    // the file counts for retention and LRU, but its tokens
                    // stay unverified and unindexed until a restart with
                    // enough staging.
                    let Some(status) = self.safe_regular_file(&filename) else {
                        continue;
                    };
                    let file_bytes = usize::try_from(status.st_size).unwrap_or(usize::MAX);
                    metadata.add_unindexed(Entry {
                        filename,
                        key_hash: String::new(),
                        persistence: TextRunnerPersistenceDescriptor::default(),
                        tokens: Vec::new(),
                        file_bytes,
                        payload_bytes: 0,
                        last_access: modification_time(&status),
                    });
                    self.retained.fetch_add(file_bytes, Ordering::Relaxed);
                    self.retained_entries.fetch_add(1, Ordering::Relaxed);
                    self.emit(
                        ContinuationDiskEventAction::Skipped,
                        ContinuationDiskEventReason::StagingCapacity,
                        file_bytes,
                        0,
                        0,
                        0.0,
                    );
                    continue;
                }
                Err(reason) => {
                    self.remove_file_only(&filename);
                    self.emit(ContinuationDiskEventAction::Removed, reason, 0, 0, 0, 0.0);
                    continue;
                }
            };
            let digest = self.hash_key(&parsed.persistence, &parsed.tokens)?;
            if let Some(duplicate) =
                metadata.find_exact(&digest, &parsed.persistence, &parsed.tokens)
            {
                let newer = metadata
                    .entries
                    .get(&duplicate)
                    .is_some_and(|e| e.last_access >= last_access);
                if newer {
                    self.remove_file_only(&filename);
                    self.emit(
                        ContinuationDiskEventAction::Removed,
                        ContinuationDiskEventReason::ExactReplacement,
                        file_bytes,
                        parsed.payload_bytes,
                        parsed.tokens.len(),
                        0.0,
                    );
                    continue;
                }
                self.remove_entry(
                    metadata,
                    duplicate,
                    ContinuationDiskEventReason::ExactReplacement,
                );
            }
            metadata.add(Entry {
                filename,
                key_hash: digest,
                persistence: parsed.persistence,
                tokens: parsed.tokens,
                file_bytes,
                payload_bytes: parsed.payload_bytes,
                last_access,
            });
            self.retained.fetch_add(file_bytes, Ordering::Relaxed);
            self.retained_entries.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Unlinks and forgets an entry; an unlink failure other than a missing
    /// file keeps it and returns false.
    pub(super) fn remove_entry(
        &self,
        metadata: &mut Metadata,
        id: EntryId,
        reason: ContinuationDiskEventReason,
    ) -> bool {
        let Some(entry) = metadata.entries.get(&id) else {
            return true;
        };
        if let Err(error) = rustix::fs::unlinkat(
            &self.directory,
            entry.filename.as_os_str(),
            AtFlags::empty(),
        ) && error != Errno::NOENT
        {
            self.emit(
                ContinuationDiskEventAction::Skipped,
                ContinuationDiskEventReason::IoFailure,
                entry.file_bytes,
                entry.payload_bytes,
                entry.tokens.len(),
                0.0,
            );
            return false;
        }
        let Some(entry) = metadata.remove(id) else {
            return true;
        };
        self.retained.fetch_sub(entry.file_bytes, Ordering::Relaxed);
        self.retained_entries.fetch_sub(1, Ordering::Relaxed);
        self.emit(
            ContinuationDiskEventAction::Removed,
            reason,
            entry.file_bytes,
            entry.payload_bytes,
            entry.tokens.len(),
            0.0,
        );
        true
    }

    pub(super) fn make_capacity(&self, metadata: &mut Metadata, file_bytes: usize) -> bool {
        if file_bytes > self.options.capacity_bytes {
            return false;
        }
        while self.retained.load(Ordering::Relaxed) > self.options.capacity_bytes - file_bytes {
            let Some(victim) = metadata.least_recently_used() else {
                return false;
            };
            if !self.remove_entry(metadata, victim, ContinuationDiskEventReason::Lru) {
                return false;
            }
        }
        true
    }

    pub(super) fn new_final_filename(&self, digest: &str, counter: &mut u64) -> Result<String> {
        for _ in 0..64 {
            let filename = format!("{digest}-{}{FILE_SUFFIX}", unique_suffix(counter));
            if matches!(
                rustix::fs::statat(
                    &self.directory,
                    filename.as_str(),
                    AtFlags::SYMLINK_NOFOLLOW
                ),
                Err(Errno::NOENT)
            ) {
                return Ok(filename);
            }
        }
        Err(Error::runtime(
            "failed to allocate a continuation disk filename",
        ))
    }

    /// Writes a private temporary file, fsyncs it, renames it into place and
    /// fsyncs the directory.
    pub(super) fn publish_image<R: TextModelRunner>(
        &self,
        final_filename: &str,
        header: &[u8],
        runner: &R,
        snapshot: &R::Snapshot,
        payload_bytes: usize,
        counter: &mut u64,
    ) -> bool {
        // `save` serialises this store's publishers and startup indexing
        // finishes before its worker starts, so no two owners in one store
        // share the directory's open file description.
        let Some(_publication) =
            DirectoryLock::acquire(&self.directory, rustix::fs::FlockOperation::LockShared)
        else {
            return false;
        };
        let temporary = format!("{TEMPORARY_PREFIX}{}", unique_suffix(counter));
        let Ok(descriptor) = rustix::fs::openat(
            &self.directory,
            temporary.as_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::RUSR | Mode::WUSR,
        ) else {
            return false;
        };
        let mut file = File::from(descriptor);
        let valid = write_image(&mut file, header, runner, snapshot, payload_bytes).is_ok()
            && file.sync_all().is_ok();
        // Unlike C++, a failing close() after a successful fsync is not
        // observable: `File` ignores close errors on drop.
        drop(file);
        if !valid {
            self.remove_file_only(OsStr::new(&temporary));
            return false;
        }
        if rustix::fs::renameat(
            &self.directory,
            temporary.as_str(),
            &self.directory,
            final_filename,
        )
        .is_err()
        {
            self.remove_file_only(OsStr::new(&temporary));
            return false;
        }
        if rustix::fs::fsync(&self.directory).is_err() {
            self.remove_file_only(OsStr::new(final_filename));
            let _ = rustix::fs::fsync(&self.directory);
            return false;
        }
        true
    }
}
