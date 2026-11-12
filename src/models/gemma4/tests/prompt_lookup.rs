//! The agreement rule of the extension's prompt-lookup copy
//! (`prompt_lookup.hpp`, `Find(..., agreed)`), which Gemma 4 uses through
//! `gufo_core::prompt_lookup::PromptLookup::find_agreed`.

use gufo_core::prompt_lookup::{Match, PromptLookup};

#[test]
fn agreement_counts_the_shared_continuation() {
    // The key 1 2 3 occurs after positions 3 and 9; the most recent
    // occurrence proposes 10 11 99 ..., the earlier one 10 11 12 ...
    let committed = [1, 2, 3, 10, 11, 12, 1, 2, 3, 10, 11, 99, 1, 2, 3];
    let mut lookup = PromptLookup::<i32>::default();
    lookup.extend(&committed, false);
    let plain = lookup.find_accepted(&committed, &[], 3, |_| true, false);
    let (found, agreed) = lookup.find_agreed(&committed, &[], 3, |_| true, false);
    assert_eq!(found, plain, "the agreement does not change the match");
    assert_eq!(
        found,
        Match {
            start: 9,
            length: 3
        }
    );
    assert_eq!(agreed, Some(2), "both continuations agree on 10 11");

    // A rejected source still competes.
    let (found, agreed) = lookup.find_agreed(&committed, &[], 3, |start| start != 3, false);
    assert_eq!((found.start, agreed), (9, Some(2)));

    // Without a competitor the whole rest of the source agrees.
    let alone = [7, 8, 9, 4, 5, 7, 8, 9];
    let mut lookup = PromptLookup::<i32>::default();
    lookup.extend(&alone, false);
    let (found, agreed) = lookup.find_agreed(&alone, &[], 3, |_| true, false);
    assert_eq!(
        found,
        Match {
            start: 3,
            length: 3
        }
    );
    assert_eq!(agreed, Some(alone.len() - 3));

    // No match, no agreement.
    let (found, agreed) = lookup.find_agreed(&alone, &[], 4, |_| true, false);
    assert_eq!((found.length, agreed), (0, None));
}
