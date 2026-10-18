//! --notes: one owner's notes in the server, looked up per chat request and written over HTTP; built copy-on-write.
const std = @import("std");
const model_text = @import("model_text.zig");
const clock = @import("clock.zig");
const index = @import("notes/index.zig");
const sa = @import("notes/sa.zig");
const lookup = @import("notes/lookup.zig");
const model_mod = @import("notes/model.zig");
const write = @import("notes/write.zig");
const journal_mod = @import("notes/journal.zig");
const line_mod = @import("notes/line.zig");
const zero = @import("notes/zero_alloc.zig");
const records_mod = @import("notes_records.zig");
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The n-gram index a store builds: LLWZ's posting lists (the default), or a suffix array (--notes-index sa).
pub const IndexKind = index.Kind;

/// How a snapshot's tokenizer was checked: the tokenizer.json SHA-256 it names, or its vocabulary size only.
pub const Check = enum { sha256, vocabulary_only };

/// The most of tokenizer.json the startup check reads (the grammar compiler reads it under the same limit).
pub const tokenizer_limit: usize = 256 << 20;

/// One build of the index, held by lookups; `garbage`: lines a write replaced, freed with it (oldest first).
pub const Version = struct { idx: *index.Index, readers: std.atomic.Value(u32) = .init(0), garbage: std.ArrayList([]u8) = .empty };

/// A lookup's scratch in the pool, with the erase epoch it was made in (an older one is freed, not reused).
const Kept = struct { sc: lookup.Scratch, epoch: u64 };

pub const Store = struct {
    zeroing: zero.Zeroing,
    gpa: Allocator, // zeroing.allocator(): every note's bytes are zeroed when freed
    io: std.Io,
    text: model_text.Text,
    model: *Model, // the writer's (under `writer`); an erase replaces it, so lookups never read it
    fixed_settings: index.Settings, // the snapshot's, the same in every model: what a lookup may read without a lock
    lock: std.Io.Mutex = .init, // current, retired, free and epoch; held only to swap or take
    current: *Version,
    retired: std.ArrayList(*Version) = .empty,
    free: std.ArrayList(*Kept) = .empty,
    epoch: u64 = 0,
    writer: std.Io.Mutex = .init, // one write at a time: the model, the journal and the records
    log: records_mod.Log,
    journal: ?journal_mod.Journal = null,
    read_only: []const u8 = "", // why writes are refused ("" when they are not)
    snapshot_path: ?[]u8 = null,
    snapshot_bytes: ?[]u8 = null, // a store loaded from bytes keeps them for an erase's rebuild
    snapshot_sha: [64]u8,
    next_ver: u64 = 1,
    load_ms: f64 = 0,
    check: Check = .vocabulary_only,
    today: ?i64 = null, // the day writes are told on, as an ordinal (tests); null: the local date
    replayed: usize = 0,
    truncated: u64 = 0,
    auto_write: bool = true, // a "notes": true chat request's last user message is written (--no-notes-write: not)
    kind: IndexKind = .postings, // the n-gram index each version builds
    long: index.Long = .{}, // sa: matches longer than n_max (off: the lookup LLWZ's tests check)
    side_max: u32 = 1024, // sa: a write's version keeps its base while at most this many notes are past it
    merges: u64 = 0, // sa: bases built by writes and merges since startup (the load's not counted)
    defer_erase_merge: bool = false, // tests only: an erase that leaves the old base and model to a later merge
    parked: std.ArrayList(*Model) = .empty, // the models such an erase left

    /// Loads the snapshot at `path` for a model whose tokenizer is `text` (its tokenizer.json lies in `model_dir`).
    pub fn open(gpa: Allocator, io: std.Io, path: []const u8, text: model_text.Text, model_dir: []const u8, problem: *[]const u8) !*Store {
        return openLimited(gpa, io, path, text, model_dir, tokenizer_limit, problem);
    }

    /// open() building the n-gram index of `kind`.
    pub fn openIndexed(gpa: Allocator, io: std.Io, path: []const u8, text: model_text.Text, model_dir: []const u8, kind: IndexKind, problem: *[]const u8) !*Store {
        return openWith(gpa, io, path, text, model_dir, tokenizer_limit, kind, problem);
    }

    /// open() with the tokenizer.json read limit given; a named hash passes only when the model's is read and equal.
    pub fn openLimited(gpa: Allocator, io: std.Io, path: []const u8, text: model_text.Text, model_dir: []const u8, limit: usize, problem: *[]const u8) !*Store {
        return openWith(gpa, io, path, text, model_dir, limit, .postings, problem);
    }

    fn openWith(gpa: Allocator, io: std.Io, path: []const u8, text: model_text.Text, model_dir: []const u8, limit: usize, kind: IndexKind, problem: *[]const u8) !*Store {
        const t0 = clock.nowNs(io);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch |e| {
            problem.* = "the snapshot file cannot be read";
            return e;
        };
        defer gpa.free(bytes);
        const journal_path = try std.mem.concat(gpa, u8, &.{ path, ".journal" });
        defer gpa.free(journal_path);
        const s = try loadWith(gpa, io, bytes, text, journal_path, kind, problem);
        errdefer s.deinit();
        s.snapshot_path = try s.gpa.dupe(u8, path);
        if (s.snapshot_bytes) |b| s.gpa.free(b);
        s.snapshot_bytes = null;
        if (s.model.tokenizer_sha256.len > 0) {
            const sha = try tokenizerSha(gpa, io, model_dir, limit, problem);
            if (!std.mem.eql(u8, &sha, s.model.tokenizer_sha256)) {
                problem.* = "the snapshot names a tokenizer.json SHA-256 that is not this model's: it was exported with another tokenizer";
                return error.TokenizerMismatch;
            }
            s.check = .sha256;
        }
        s.load_ms = clock.seconds(clock.nowNs(io) - t0) * 1e3;
        return s;
    }

    /// The startup log line: what was loaded, and how its tokenizer was checked.
    pub fn startupLine(s: *const Store, buf: []u8, path: []const u8) []const u8 {
        const ix = s.current.idx;
        const checked = switch (s.check) {
            .sha256 => "tokenizer checked: the model's tokenizer.json SHA-256 equals the snapshot's",
            .vocabulary_only => "tokenizer checked by vocabulary size only (vocabulary-only compatibility): the snapshot names no tokenizer.json SHA-256",
        };
        var w: [240]u8 = undefined;
        const writes = if (s.read_only.len > 0) std.fmt.bufPrint(&w, "writes off ({s})", .{s.read_only}) catch "writes off" else std.fmt.bufPrint(&w, "{d} writes replayed from the journal; a chat request's user statements {s}", .{ s.replayed, if (s.auto_write) "are written through the intake (--no-notes-write: not)" else "are not written (--no-notes-write)" }) catch "";
        var k: [96]u8 = undefined;
        const kind = if (s.kind == .sa) std.fmt.bufPrint(&k, "; suffix-array index (--notes-index sa, a merge past {d} written notes)", .{s.side_max}) catch "" else "";
        return std.fmt.bufPrint(buf, "notes: {d} notes ({d} current) from {s}, cap {d}, loaded in {d:.1} ms{s}; {s}; {s}; a chat request sends \"notes\": true to have them looked up", .{ ix.notes(), ix.n_cur, path, ix.settings.cap, s.load_ms, kind, checked, writes }) catch buf;
    }

    /// A snapshot's JSON text as a store, checked against the model's vocabulary size; writes are off (no journal).
    pub fn load(gpa: Allocator, io: std.Io, bytes: []const u8, text: model_text.Text, problem: *[]const u8) !*Store {
        return loadJournaled(gpa, io, bytes, text, null, problem);
    }

    /// load() with the journal at `journal_path` replayed over the snapshot; null keeps writes off.
    pub fn loadJournaled(parent: Allocator, io: std.Io, bytes: []const u8, text: model_text.Text, journal_path: ?[]const u8, problem: *[]const u8) !*Store {
        return loadWith(parent, io, bytes, text, journal_path, .postings, problem);
    }

    /// load() building the n-gram index of `kind`.
    pub fn loadIndexed(gpa: Allocator, io: std.Io, bytes: []const u8, text: model_text.Text, kind: IndexKind, problem: *[]const u8) !*Store {
        return loadWith(gpa, io, bytes, text, null, kind, problem);
    }

    fn loadWith(parent: Allocator, io: std.Io, bytes: []const u8, text: model_text.Text, journal_path: ?[]const u8, kind: IndexKind, problem: *[]const u8) !*Store {
        const s = try parent.create(Store);
        errdefer parent.destroy(s);
        s.zeroing = .{ .child = parent };
        const gpa = s.zeroing.allocator();
        var sha: [Sha256.digest_length]u8 = undefined;
        Sha256.hash(bytes, &sha, .{});
        const m = model_mod.parse(gpa, bytes) catch |e| {
            problem.* = switch (e) {
                error.Unsupported => "the snapshot needs what this lookup does not do (n_max over 4, or a stand-in tokenizer)",
                error.OutOfMemory => "out of memory",
                error.BadSnapshot => "the file is not a valid lw-notes-snapshot/1 snapshot",
            };
            return e;
        };
        errdefer m.deinit();
        if (m.vkey.len != text.vocabSize()) {
            problem.* = "the snapshot's vocabulary is not the model tokenizer's size";
            return error.BadSnapshot;
        }
        s.* = .{ .zeroing = s.zeroing, .gpa = gpa, .io = io, .text = text, .model = m, .fixed_settings = m.settings, .current = undefined, .log = .{ .gpa = gpa }, .snapshot_sha = std.fmt.bytesToHex(sha, .lower), .kind = kind };
        errdefer s.log.deinit();
        if (journal_path) |jp| {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            var recs: std.ArrayList(write.Record) = .empty;
            s.journal = try journal_mod.Journal.open(gpa, arena.allocator(), jp, s.snapshot_sha, &recs, problem);
            errdefer s.journal.?.deinit();
            s.truncated = s.journal.?.truncated;
            for (recs.items) |r| try s.replay(r, problem);
            s.replayed = recs.items.len;
            try m.trim(); // the lists the writes grew, at their lengths
            s.log.next_source = @max(s.log.next_source, s.journal.?.next_source);
            s.snapshot_bytes = try gpa.dupe(u8, bytes);
        } else s.read_only = "the store has no journal";
        errdefer if (s.snapshot_bytes) |b| gpa.free(b);
        const idx = index.Index.buildKind(gpa, m, kind) catch |e| {
            if (e == error.Unsupported) problem.* = "the suffix-array index needs a vocabulary whose keys are all real and whose ids of one key agree on being a word";
            return e;
        };
        errdefer idx.deinit();
        try @import("notes/alias.zig").populate(idx, m, text);
        const v = try gpa.create(Version);
        v.* = .{ .idx = idx };
        s.current = v;
        return s;
    }

    /// One journal record applied to the model at startup (the index is built once, after the last).
    fn replay(s: *Store, r: write.Record, problem: *[]const u8) !void {
        var arena = std.heap.ArenaAllocator.init(s.gpa);
        defer arena.deinit();
        var out: write.Outcome = undefined;
        var undo: write.Undo = .{};
        defer undo.deinit(s.gpa);
        write.apply(s.model, s.text, arena.allocator(), r, &out, &undo) catch |e| {
            problem.* = "a write in the notes journal cannot be applied";
            return e;
        };
        write.commit(s.model, &undo);
        try s.log.add(r, out, s.model);
    }

    pub fn deinit(s: *Store) void {
        const parent = s.zeroing.child;
        for (s.free.items) |k| {
            k.sc.deinit();
            s.gpa.destroy(k);
        }
        s.free.deinit(s.gpa);
        for (s.retired.items) |v| freeVersion(s.gpa, v);
        s.retired.deinit(s.gpa);
        freeVersion(s.gpa, s.current);
        for (s.parked.items) |m| m.deinit();
        s.parked.deinit(s.gpa);
        s.log.deinit();
        if (s.journal) |*j| j.deinit();
        if (s.snapshot_path) |p| s.gpa.free(p);
        if (s.snapshot_bytes) |b| s.gpa.free(b);
        s.model.deinit();
        parent.destroy(s);
    }

    fn freeVersion(g: Allocator, v: *Version) void {
        v.idx.deinit();
        for (v.garbage.items) |l| g.free(l);
        v.garbage.deinit(g);
        g.destroy(v);
    }

    /// The index lookups read now, held until release().
    pub fn acquire(s: *Store) *Version {
        s.lock.lockUncancelable(s.io);
        defer s.lock.unlock(s.io);
        const v = s.current;
        _ = v.readers.fetchAdd(1, .acquire);
        return v;
    }

    pub fn release(_: *Store, v: *Version) void {
        _ = v.readers.fetchSub(1, .release);
    }

    /// The snapshot's settings (the same in every version).
    pub fn settings(s: *const Store) index.Settings {
        return s.fixed_settings;
    }

    pub fn take(s: *Store, idx: *const index.Index) Allocator.Error!*Kept {
        s.lock.lockUncancelable(s.io);
        const kept = s.free.pop();
        const epoch = s.epoch;
        s.lock.unlock(s.io);
        if (kept) |k| {
            if (k.epoch == epoch and k.sc.slot.len >= idx.notes() and (idx.kind != .sa or k.sc.seen.len >= idx.notes())) return k;
            k.sc.deinit(); // made before an erase, or for fewer notes
            s.gpa.destroy(k);
        }
        const k = try s.gpa.create(Kept);
        errdefer s.gpa.destroy(k);
        k.* = .{ .sc = try lookup.Scratch.init(s.gpa, idx), .epoch = epoch };
        return k;
    }

    pub fn give(s: *Store, k: *Kept) void {
        s.lock.lockUncancelable(s.io);
        const keep = k.epoch == s.epoch and if (s.free.append(s.gpa, k)) |_| true else |_| false;
        s.lock.unlock(s.io);
        if (!keep) {
            k.sc.deinit();
            s.gpa.destroy(k);
        }
    }

    /// NoteStore.retrieve(owner, question, cap, past, exclude) from the question's ids, fit() counted by `text`.
    pub fn retrieve(s: *Store, a: Allocator, text: model_text.Text, qids: []const u32, past: bool, cap: u32, exclude: []const u32) Failure!Found {
        return s.retrieveWith(a, text, qids, past, cap, exclude, null);
    }

    /// retrieve() with the snapshot's hop setting replaced for this lookup (null: as the snapshot says).
    pub fn retrieveWith(s: *Store, a: Allocator, text: model_text.Text, qids: []const u32, past: bool, cap: u32, exclude: []const u32, hop: ?index.Hop) Failure!Found {
        return s.retrieveSubjects(a, text, qids, past, cap, exclude, hop, false);
    }

    pub fn retrieveSubjects(s: *Store, a: Allocator, text: model_text.Text, qids: []const u32, past: bool, cap: u32, exclude: []const u32, hop: ?index.Hop, resolve_subjects: bool) Failure!Found {
        const v = s.acquire();
        defer s.release(v);
        var view: index.Index = undefined; // a shallow copy: the same notes and maps, read only
        const idx = if (hop != null or resolve_subjects) blk: {
            view = v.idx.*;
            if (hop) |h| view.settings.hop = h;
            view.settings.resolve_subjects = resolve_subjects;
            break :blk &view;
        } else v.idx;
        const k = try s.take(idx);
        defer s.give(k);
        const sc = &k.sc;
        const io = s.io;
        const t0 = clock.nowNs(io);
        const packed_notes = try lookup.lookup(idx, sc, qids, past, cap, exclude, s.long);
        const chosen = try a.dupe(u32, packed_notes.chosen);
        var shown = try lookup.render(idx, sc, chosen);
        const t1 = clock.nowNs(io);
        // fit(): the whole block counted by the model's tokenizer; the lowest ranked note goes while it is over the cap
        var n = chosen.len;
        var tokens: usize = 0;
        while (n > 0) {
            // Oversized retry encodes must not accumulate in the request arena.
            var fit_arena = std.heap.ArenaAllocator.init(s.gpa);
            defer fit_arena.deinit();
            tokens = (text.encode(fit_arena.allocator(), shown.block, false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.Tokenizer,
            }).len;
            if (tokens <= cap) break;
            n -= 1;
            tokens = 0;
            shown = try lookup.render(idx, sc, chosen[0..n]);
        }
        const t2 = clock.nowNs(io);
        return .{
            .ids = try a.dupe(u32, shown.ids),
            .block = try a.dupe(u8, shown.block),
            .tokens = tokens,
            .pack_tokens = packed_notes.tokens,
            .dropped = chosen.len - n,
            .candidates = packed_notes.candidates,
            .long = packed_notes.long,
            .lookup_ns = t1 - t0,
            .fit_ns = t2 - t1,
            .in_store = idx.notes(),
            .ver = idx.ver,
        };
    }

    /// Frees retired versions no lookup holds, oldest first; with `all`, waits until none holds any. Never allocates.
    pub fn reclaim(s: *Store, all: bool) void {
        while (true) {
            s.lock.lockUncancelable(s.io);
            var found: ?*Version = null;
            if (s.retired.items.len > 0 and s.retired.items[0].readers.load(.acquire) == 0) found = s.retired.orderedRemove(0);
            const left = s.retired.items.len;
            s.lock.unlock(s.io);
            if (found) |v| {
                freeVersion(s.gpa, v);
                continue;
            }
            if (!all or left == 0) return;
            std.Io.sleep(s.io, .fromMilliseconds(1), .awake) catch {};
        }
    }

    /// Makes `idx` the version lookups read from now; the old one is retired (its room was reserved).
    pub fn swap(s: *Store, v: *Version, erase: bool) void {
        s.lock.lockUncancelable(s.io);
        defer s.lock.unlock(s.io);
        s.retired.appendAssumeCapacity(s.current);
        s.current = v;
        if (erase) s.epoch += 1;
    }

    /// Frees the pool's scratches made before the last erase (they may hold an erased note's text). Never allocates.
    pub fn sweep(s: *Store) void {
        while (true) {
            s.lock.lockUncancelable(s.io);
            var found: ?*Kept = null;
            for (s.free.items, 0..) |k, i| if (k.epoch != s.epoch) {
                found = s.free.swapRemove(i);
                break;
            };
            s.lock.unlock(s.io);
            const k = found orelse return;
            k.sc.deinit();
            s.gpa.destroy(k);
        }
    }

    /// A new numbered version; sa: on the current base while side_max or fewer notes follow it, else a merge.
    pub fn build(s: *Store) !*Version {
        var reuse: ?*sa.Base = null;
        if (s.kind == .sa) if (s.current.idx.base) |b| {
            if (s.model.notes.items.len - b.notes <= s.side_max) reuse = b;
        };
        return s.buildOn(reuse);
    }

    fn buildOn(s: *Store, reuse: ?*sa.Base) !*Version {
        const idx = try index.Index.buildWith(s.gpa, s.model, s.kind, reuse);
        errdefer idx.deinit();
        try @import("notes/alias.zig").populate(idx, s.model, s.text);
        const v = try s.gpa.create(Version);
        v.* = .{ .idx = idx };
        idx.ver = s.next_ver;
        if (s.kind == .sa and reuse == null) s.merges += 1;
        return v;
    }

    /// sa: the side buffer merged now, and what a deferred erase left freed; postings: the index built again.
    pub fn merge(s: *Store) !void {
        s.writer.lockUncancelable(s.io);
        defer s.writer.unlock(s.io);
        try s.retired.ensureUnusedCapacity(s.gpa, 1);
        const v = try s.buildOn(null);
        s.swap(v, false);
        s.next_ver += 1;
        if (s.parked.items.len > 0) { // what a deferred erase left: freed as an erase frees it
            s.reclaim(true);
            for (s.parked.items) |m| m.deinit();
            s.parked.clearRetainingCapacity();
            s.sweep();
            s.text.forget();
        }
        s.reclaim(false);
    }

    /// The tick a write made now is told at: after every write so far, and inside today (NoteStore._apply).
    pub fn toldNow(s: *const Store) i64 {
        const day = s.today orelse localOrdinal(s.io);
        return @max(s.model.told_last + 100, day * line_mod.tpd);
    }
};

/// Today's date where the server runs, as an ordinal (Python's date.today().toordinal()).
fn localOrdinal(io: std.Io) i64 {
    const Tm = extern struct { sec: c_int, min: c_int, hour: c_int, mday: c_int, mon: c_int, year: c_int, wday: c_int, yday: c_int, isdst: c_int, gmtoff: c_long, zone: ?[*:0]const u8 };
    const ext = struct {
        extern "c" fn localtime_r(t: *const i64, out: *Tm) ?*Tm;
    };
    const now: i64 = @intCast(@divFloor(std.Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_s));
    var tm: Tm = undefined;
    if (ext.localtime_r(&now, &tm) == null) return @divFloor(now, 86400) + 719163;
    return line_mod.ordinalOf(@as(i64, tm.year) + 1900, @intCast(tm.mon + 1), @intCast(tm.mday));
}

pub const Failure = error{ OutOfMemory, Tokenizer };

/// What retrieve() found.
pub const Found = struct {
    ids: []const u32,
    block: []const u8, // "" when nothing is retrieved: then no system message goes in
    tokens: usize, // the block's tokens by the model's tokenizer (fit()'s count)
    pack_tokens: u32, // pack()'s sum of the parts
    dropped: usize, // notes fit() took out
    candidates: u32,
    long: u32 = 0, // sa with longer matches on: the n-grams over n_max weighted
    lookup_ns: i96,
    fit_ns: i96,
    in_store: usize = 0, // notes in the version it read
    ver: u64 = 0, // that version's number
};

/// The SHA-256 of `model_dir`/tokenizer.json in hex; a file that cannot be read is an error naming why, never a pass.
fn tokenizerSha(gpa: Allocator, io: std.Io, model_dir: []const u8, limit: usize, problem: *[]const u8) ![64]u8 {
    if (model_dir.len == 0) {
        problem.* = "the snapshot names a tokenizer.json SHA-256, and the server has no model folder to read tokenizer.json from";
        return error.NoModelDir;
    }
    const path = try std.fs.path.join(gpa, &.{ model_dir, "tokenizer.json" });
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(limit)) catch |e| {
        problem.* = switch (e) {
            error.FileNotFound => "the snapshot names a tokenizer.json SHA-256, and the model folder has no tokenizer.json to check it against",
            error.StreamTooLong => "the snapshot names a tokenizer.json SHA-256, and the model's tokenizer.json is over the read limit (256 MiB), so it cannot be checked",
            error.OutOfMemory => "out of memory reading the model's tokenizer.json",
            else => "the snapshot names a tokenizer.json SHA-256, and the model's tokenizer.json cannot be read to check it",
        };
        return e;
    };
    defer gpa.free(bytes);
    var out: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}
