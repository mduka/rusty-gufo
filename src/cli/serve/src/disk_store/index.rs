//! The in-memory index: the prefix tree over input tokens and the entry
//! metadata.

use super::*;

impl PrefixNode {
    pub(super) fn is_empty(&self) -> bool {
        self.entry.is_none() && self.children.is_empty()
    }

    pub(super) fn insert(&mut self, tokens: &[Token], value: EntryId) {
        let Some(&first) = tokens.first() else {
            self.entry = Some(value);
            return;
        };
        let child = match self.children.entry(first) {
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(PrefixNode {
                    edge: tokens.to_vec(),
                    entry: Some(value),
                    children: BTreeMap::new(),
                });
                return;
            }
            std::collections::btree_map::Entry::Occupied(occupied) => occupied.into_mut(),
        };
        let shared = common_prefix(&child.edge, tokens);
        if shared < child.edge.len() {
            // Split the edge: a branch keeps the shared tokens and adopts
            // the old child with the remainder.
            let mut old = std::mem::take(child);
            let remainder = old.edge.split_off(shared);
            child.edge = std::mem::replace(&mut old.edge, remainder);
            child.children.insert(old.edge[0], old);
        }
        child.insert(&tokens[shared..], value);
    }

    pub(super) fn erase(&mut self, tokens: &[Token]) {
        let Some(&first) = tokens.first() else {
            self.entry = None;
            return;
        };
        let Some(child) = self.children.get_mut(&first) else {
            return;
        };
        let length = child.edge.len();
        if tokens.len() < length || child.edge[..] != tokens[..length] {
            return;
        }
        child.erase(&tokens[length..]);
        if child.is_empty() {
            self.children.remove(&first);
        }
    }

    /// The longest stored prefix of `prompt` and, when asked, the lengths
    /// where the prompt leaves stored paths without an entry there: at a
    /// branching node, where no child continues it, or inside an edge.
    pub(super) fn find(&self, prompt: &[Token], boundaries: bool) -> PrefixMatch {
        let mut result = PrefixMatch::default();
        let mut node = self;
        let mut offset = 0;
        loop {
            if node.entry.is_some() {
                result.longest = node.entry;
            }
            if offset == prompt.len() {
                break;
            }
            let next = node.children.get(&prompt[offset]);
            if boundaries
                && node.entry.is_none()
                && offset != 0
                && (node.children.len() > 1 || (!node.children.is_empty() && next.is_none()))
            {
                result.shared_boundaries.push(offset);
            }
            let Some(child) = next else {
                break;
            };
            let shared = common_prefix(&child.edge, &prompt[offset..]);
            offset += shared;
            if shared != child.edge.len() {
                if boundaries && offset != 0 && offset < prompt.len() {
                    result.shared_boundaries.push(offset);
                }
                break;
            }
            node = child;
        }
        result
    }
}

impl Metadata {
    pub(super) fn find_exact(
        &self,
        digest: &str,
        persistence: &TextRunnerPersistenceDescriptor,
        tokens: &[Token],
    ) -> Option<EntryId> {
        self.index.get(digest)?.iter().copied().find(|id| {
            self.entries
                .get(id)
                .is_some_and(|e| e.persistence == *persistence && e.tokens == tokens)
        })
    }

    pub(super) fn find_longest(
        &self,
        persistence: &TextRunnerPersistenceDescriptor,
        prompt: &[Token],
    ) -> Option<EntryId> {
        let key = (
            persistence.payload_version,
            persistence.compatibility_identity.clone(),
        );
        self.prefixes.get(&key)?.find(prompt, false).longest
    }

    /// The longest stored prefix of `prompt` saved with the input identity
    /// at its length (`FindLongestInputCandidate`).
    pub(super) fn find_longest_input<R: TextModelRunner + ?Sized>(
        &self,
        runner: &R,
        prompt: &[Token],
        input: InputIdentity<'_>,
    ) -> Option<EntryId> {
        let mut best: Option<(EntryId, usize)> = None;
        visit_input_ranges(prompt.len(), input, |begin, end, identity| {
            let Some(persistence) = persistence_for_input(runner, identity) else {
                return;
            };
            let Some(id) = self.find_longest(&persistence, &prompt[..end]) else {
                return;
            };
            let Some(tokens) = self.entries.get(&id).map(|e| e.tokens.len()) else {
                return;
            };
            if tokens >= begin && best.is_none_or(|(_, longest)| tokens > longest) {
                best = Some((id, tokens));
            }
        });
        best.map(|(id, _)| id)
    }

    pub(super) fn add(&mut self, entry: Entry) -> EntryId {
        let id = self.next_id;
        self.next_id += 1;
        self.index
            .entry(entry.key_hash.clone())
            .or_default()
            .push(id);
        let key = (
            entry.persistence.payload_version,
            entry.persistence.compatibility_identity.clone(),
        );
        self.prefixes
            .entry(key)
            .or_default()
            .insert(&entry.tokens, id);
        self.entries.insert(id, entry);
        id
    }

    /// A file that counts for retention and LRU but is never looked up.
    pub(super) fn add_unindexed(&mut self, entry: Entry) -> EntryId {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, entry);
        id
    }

    pub(super) fn remove(&mut self, id: EntryId) -> Option<Entry> {
        let entry = self.entries.remove(&id)?;
        if entry.key_hash.is_empty() {
            return Some(entry);
        }
        let key = (
            entry.persistence.payload_version,
            entry.persistence.compatibility_identity.clone(),
        );
        if let Some(root) = self.prefixes.get_mut(&key) {
            root.erase(&entry.tokens);
            if root.is_empty() {
                self.prefixes.remove(&key);
            }
        }
        if let Some(ids) = self.index.get_mut(&entry.key_hash) {
            ids.retain(|&other| other != id);
            if ids.is_empty() {
                self.index.remove(&entry.key_hash);
            }
        }
        Some(entry)
    }

    /// The oldest entry; equal times fall back to the file name.
    pub(super) fn least_recently_used(&self) -> Option<EntryId> {
        self.entries
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.last_access
                    .cmp(&b.last_access)
                    .then_with(|| a.filename.cmp(&b.filename))
            })
            .map(|(&id, _)| id)
    }
}
