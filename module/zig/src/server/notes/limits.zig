//! Notes size limits, each refused with a clear error (docs/notes-store.md, Limits).
const std = @import("std");

/// A journal larger than this is refused at startup: it is read whole, and every erase rewrites it whole.
pub const journal_bytes: u64 = 64 << 20;
/// A write is refused when the journal would pass this: the rest is room for start lines and erase records.
pub const journal_write_bytes: u64 = journal_bytes - (1 << 20);
/// A snapshot larger than this is refused at startup and at erase (both read it whole).
pub const snapshot_bytes: u64 = 256 << 20;
/// Most notes a store holds, including snapshot notes, journal notes and retained history.
pub const notes: usize = 100_000;

pub const journal_full = std.fmt.comptimePrint("the notes journal is at its size limit ({d} MiB of writes); erase notes before writing more", .{journal_write_bytes >> 20});
pub const notes_full = std.fmt.comptimePrint("the notes store holds the most notes it takes ({d}, snapshot and journal together); erase notes before writing more", .{notes});
pub const notes_over = std.fmt.comptimePrint("the notes store (snapshot and journal together) holds more than {d} notes; refused", .{notes});
pub const journal_too_large = std.fmt.comptimePrint("the notes journal is over its {d} MiB read limit; refused", .{journal_bytes >> 20});
pub const snapshot_too_large = std.fmt.comptimePrint("the snapshot is over its {d} MiB read limit; refused", .{snapshot_bytes >> 20});
