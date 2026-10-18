//! The notes' writes over HTTP: add, change, take back, erase, list; each on disk before it is seen or answered.
const std = @import("std");
const store_mod = @import("notes_store.zig");
const records_mod = @import("notes_records.zig");
const model_mod = @import("notes/model.zig");
const write = @import("notes/write.zig");
const intake = @import("notes/intake.zig");
const index = @import("notes/index.zig");
const Store = store_mod.Store;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;

/// Why a write was refused: the HTTP status and the message.
pub const Refusal = struct { status: u16 = 400, message: []const u8 = "" };

pub const Error = error{ Refused, OutOfMemory };

fn refuse(r: *Refusal, status: u16, message: []const u8) Error {
    r.* = .{ .status = status, .message = message };
    return error.Refused;
}

/// A note as a listing or a write's reply shows it.
pub const View = struct {
    id: u32,
    version: u32,
    state: index.State,
    kind: model_mod.Kind,
    subject: []const u8,
    text: []const u8,
    line: []const u8,
    source: ?[]const u8,
    told: i64,
    by: u32,
};

pub fn view(m: *const Model, a: Allocator, place: u32) Allocator.Error!View {
    const n = m.notes.items[place];
    return .{ .id = n.id, .version = n.version, .state = n.state, .kind = n.kind, .subject = try a.dupe(u8, m.entities.items[n.entity]), .text = try a.dupe(u8, n.text()), .line = try a.dupe(u8, n.line()), .source = if (n.source()) |s| try a.dupe(u8, s) else null, .told = n.told, .by = n.by };
}

/// What a write did: its verdict, its source, the note it wrote (or is the same as), and the notes it retired.
pub const Written = struct {
    verdict: intake.Verdict,
    source: []const u8,
    told: i64,
    note: ?View,
    retired: []const u32,
    same_as: []const u32,
    ver: u64 = 0, // the index version it swapped in (0: none, as for a write that learns nothing)
};

/// What a write asks: a new note, a change of note `target`, or a take-back of it.
pub const Ask = struct {
    op: write.Op,
    text: []const u8,
    subject: ?[]const u8 = null,
    target: u32 = 0,
};

/// The reply to a write whose append and rollback both failed before its commit line was down: a restart drops it.
const rollback_dropped = "the write could not be stored on disk, and the journal could not be put back as it was: nothing was applied in memory; the journal may hold this write, but not its commit line, so a restart drops it. Writes are refused until the server restarts; lookups go on";

/// The same, once its commit line was written but not synced: the disk state is unknown.
const rollback_doubt = "the write's commit line reached the journal but could not be synced, and the journal could not be put back as it was: nothing was applied in memory, but the disk state is unknown and a restart may apply this write. Writes are refused until the server restarts; lookups go on";

/// The reply to an erase whose new journal was renamed in but whose folder could not be synced: applied, not certain.
const erase_unsynced = "the journal was rewritten without the note and renamed in, but its folder could not be synced, so the disk state is unknown: after a power cut either the old journal (with the note) or the new one (without it) may be there, each whole. The erase was applied in memory. Writes are refused until the server restarts; after that, list the notes and erase it again if it is back";

/// The reply to every write after one of those.
const needs_recovery = "the notes journal needs recovery, restart the server: an earlier write or erase failed and the journal's state on disk is not known (lookups go on)";

fn poisoned(s: *const Store) bool {
    return if (s.journal) |j| j.poisoned else false;
}

/// One write, whole or not at all: intake, the model, a new index version, the journal synced, then the swap.
pub fn perform(s: *Store, a: Allocator, ask: Ask, why: *Refusal) Error!Written {
    s.writer.lockUncancelable(s.io);
    defer s.writer.unlock(s.io);
    if (s.read_only.len > 0) return refuse(why, 503, s.read_only);
    if (poisoned(s)) return refuse(why, 503, needs_recovery);
    const m = s.model;
    const source = try std.fmt.allocPrint(a, "w{d}", .{s.log.next_source});
    var r: write.Record = .{ .op = ask.op, .source = source, .told = s.toldNow(), .subject = ask.subject orelse "me", .attribute = "", .text = ask.text };
    if (ask.op == .add) {
        r.attribute = try std.fmt.allocPrint(a, "note:{s}", .{source});
    } else {
        const p = m.placeOf(ask.target) orelse return refuse(why, 404, "no note has this id");
        const t = m.notes.items[p];
        if (t.state != .current) return refuse(why, 409, if (t.state == .superseded) "this note is no longer current (a later note replaced it): change or take back the current one" else "this note was taken back already");
        if (ask.op == .take_back and t.kind == .tombstone) return refuse(why, 409, "this note is a take-back; a take-back cannot be taken back (change it to say something new)");
        if (ask.op == .take_back and p < m.base_notes and m.takeback == .explicit)
            return refuse(why, 409, "under the snapshot's \"explicit\" take-back wording a take-back's label depends on whether the attribute holds several values, which the snapshot does not record");
        if (ask.subject) |sub| if (!std.mem.eql(u8, sub, m.entities.items[t.entity])) return refuse(why, 400, "a change keeps the subject of the note it changes");
        r.subject = m.entities.items[t.entity];
        r.attribute = m.attributes.items[t.attribute];
        r.target = if (p < m.base_notes) .{ .note = t.id } else .{ .source = t.source().? };
    }
    try s.retired.ensureUnusedCapacity(s.gpa, 1);
    var out: write.Outcome = undefined;
    var undo: write.Undo = .{};
    defer undo.deinit(s.gpa);
    write.apply(m, s.text, a, r, &out, &undo) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Tokenizer => refuse(why, 400, "the tokenizer cannot encode this note"),
    };
    var applied = true;
    defer if (applied) write.revert(m, &undo);
    const v: ?*store_mod.Version = if (out.verdict.learn()) (s.build() catch return error.OutOfMemory) else null;
    errdefer if (v) |x| {
        x.idx.deinit();
        s.gpa.destroy(x);
    };
    const shown: ?u32 = if (out.note != 0) out.note else if (out.same_as.len > 0) out.same_as[0] else null;
    var reply: Written = .{ .verdict = out.verdict, .source = source, .told = r.told, .note = if (shown) |id| try view(m, a, m.placeOf(id).?) else null, .retired = out.retired, .same_as = out.same_as };
    try s.current.garbage.ensureUnusedCapacity(s.gpa, undo.gone.items.len); // the replaced lines, kept by the version it retires
    var entry = try s.log.prepare(r, out, m);
    errdefer records_mod.freeEntry(s.gpa, &entry);
    s.journal.?.append(a, r) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JournalIo => refuse(why, 500, "the write could not be stored on disk; nothing changed"),
        error.JournalPoisoned => refuse(why, 500, if (s.journal.?.doubt) rollback_doubt else rollback_dropped),
    };
    // stored: from here the write stands, and nothing below can fail
    applied = false;
    write.commitKeeping(m, &undo, &s.current.garbage);
    s.log.push(entry);
    if (v) |x| {
        s.swap(x, false);
        reply.ver = x.idx.ver;
        s.next_ver += 1;
    }
    s.reclaim(false);
    return reply;
}

/// What an erase did: the writes it took out, the ids that moved (old, new), and what it could not reach.
pub const Erased = struct {
    removed: []const []const u8,
    moved: []const [2]u32,
};

/// Erase = never taught: the store rebuilt from the snapshot and the journal without the note's write (LLWZ 4.5).
pub fn erase(s: *Store, a: Allocator, id: u32, why: *Refusal) Error!Erased {
    s.writer.lockUncancelable(s.io);
    defer s.writer.unlock(s.io);
    if (s.read_only.len > 0) return refuse(why, 503, s.read_only);
    if (poisoned(s)) return refuse(why, 503, needs_recovery);
    const old = s.model;
    const p = old.placeOf(id) orelse return refuse(why, 404, "no note has this id");
    if (p < old.base_notes) return refuse(why, 409, "this note comes from the snapshot: erase it in LLWZ (NoteStore.delete) and export again");
    const x: write.Target = .{ .source = old.notes.items[p].source().? };
    var gone: std.ArrayList(usize) = .empty;
    try records_mod.erased(&s.log, x, &gone, a);
    const bytes = snapshotBytes(s, a) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => refuse(why, 500, "the snapshot file cannot be read again, or changed since startup; nothing was erased"),
    };
    const fresh = model_mod.parse(s.gpa, bytes) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => refuse(why, 500, "the snapshot no longer parses; nothing was erased"),
    };
    var fresh_owned = true;
    defer if (fresh_owned) fresh.deinit();
    var log: records_mod.Log = .{ .gpa = s.gpa, .next_source = s.log.next_source };
    var log_owned = true;
    defer if (log_owned) log.deinit();
    var kept: std.ArrayList(write.Record) = .empty;
    var removed: std.ArrayList([]const u8) = .empty;
    for (s.log.items.items, 0..) |e, k| {
        if (std.mem.indexOfScalar(usize, gone.items, k) != null) {
            try removed.append(a, try a.dupe(u8, e.r.source));
            continue;
        }
        try kept.append(a, e.r);
        var out: write.Outcome = undefined;
        var undo: write.Undo = .{};
        defer undo.deinit(s.gpa);
        write.apply(fresh, s.text, a, e.r, &out, &undo) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.Tokenizer => refuse(why, 500, "a kept write cannot be applied again; nothing was erased"),
        };
        write.commit(fresh, &undo);
        try log.add(e.r, out, fresh);
    }
    try fresh.trim(); // as at startup: the lists the kept writes grew, at their lengths
    const idx = index.Index.buildKind(s.gpa, fresh, s.kind) catch return error.OutOfMemory;
    var idx_owned = true;
    defer if (idx_owned) idx.deinit();
    @import("notes/alias.zig").populate(idx, fresh, s.text) catch return error.OutOfMemory;
    const v = try s.gpa.create(store_mod.Version);
    var v_owned = true; // not an errdefer: an erase applied but not certain on disk returns a refusal after the swap
    defer if (v_owned) s.gpa.destroy(v);
    v.* = .{ .idx = idx };
    idx.ver = s.next_ver;
    try s.retired.ensureUnusedCapacity(s.gpa, 1);
    var moved: std.ArrayList([2]u32) = .empty;
    for (old.notes.items[old.base_notes..]) |*n| if (fresh.placeOfSource(n.source().?)) |q| {
        if (fresh.notes.items[q].id != n.id) try moved.append(a, .{ n.id, fresh.notes.items[q].id });
    };
    var unsynced = false;
    s.journal.?.rewrite(a, kept.items, log.next_source) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.JournalIo => return refuse(why, 500, "the journal could not be rewritten without the note; nothing was erased"),
        error.RenamedUnsynced => unsynced = true, // the file at the journal's name lacks it now: so must the memory
    };
    // the journal no longer holds it: swap, wait for every lookup of the old versions, free them all (zeroed)
    idx_owned = false;
    v_owned = false;
    fresh_owned = false;
    log_owned = false;
    s.swap(v, true);
    s.next_ver += 1;
    s.model = fresh;
    var old_log = s.log;
    s.log = log;
    old_log.deinit();
    if (s.defer_erase_merge) {
        // the negative control of the erase tests: the note hidden now, its bytes left to a later merge
        s.parked.append(s.gpa, old) catch old.deinit();
        if (unsynced) return refuse(why, 500, erase_unsynced); // as below: applied, the disk state unknown
        return .{ .removed = removed.items, .moved = moved.items };
    }
    s.reclaim(true);
    old.deinit();
    s.sweep();
    s.text.forget(); // the tokenizer's kept encodes: a block or a prompt that quoted it
    if (unsynced) return refuse(why, 500, erase_unsynced); // the journal is poisoned: writes stop until a restart
    return .{ .removed = removed.items, .moved = moved.items };
}

/// The snapshot as at startup: its file read again (and checked against the journal's hash), or the kept bytes.
fn snapshotBytes(s: *Store, a: Allocator) ![]const u8 {
    if (s.snapshot_bytes) |b| return b;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(s.io, s.snapshot_path.?, a, .unlimited);
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sha, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(sha, .lower), &s.snapshot_sha)) return error.SnapshotChanged;
    return bytes;
}

/// The notes now: current ones, and with `history` the superseded and withdrawn ones too, by id.
pub fn list(s: *Store, a: Allocator, history: bool) Allocator.Error![]View {
    s.writer.lockUncancelable(s.io);
    defer s.writer.unlock(s.io);
    var out: std.ArrayList(View) = .empty;
    for (s.model.notes.items, 0..) |n, k| if (history or n.state == .current) try out.append(a, try view(s.model, a, @intCast(k)));
    return out.items;
}
