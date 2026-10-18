//! A failed append: rolled back, nothing changed; else writes stop and a restart cuts the tail (each index kind).
const std = @import("std");
const builtin = @import("builtin");
const rig = @import("notes_rig_test.zig");
const ops = @import("notes_ops.zig");
const notes_route = @import("notes_route.zig");
const journal_mod = @import("notes/journal.zig");
const testing = std.testing;

fn add(r: *rig.Rig, a: std.mem.Allocator, text: []const u8, why: *ops.Refusal) ops.Error!ops.Written {
    return ops.perform(r.store(), a, .{ .op = .add, .text = text }, why);
}

fn blockOf(r: *rig.Rig, a: std.mem.Allocator, q: []const u8) ![]const u8 {
    const t = r.text.text();
    return (try r.store().retrieve(a, t, try t.encode(a, q, false), false, 512, &.{})).block;
}

/// The rig of index kind `k` after two good writes, with the journal's faults hooked in.
fn setup(a: std.mem.Allocator, faults: *journal_mod.Faults, k: rig.Kind) !*rig.Rig {
    const r = try rig.Rig.initWith(testing.allocator, k);
    errdefer r.deinit();
    var why: ops.Refusal = .{};
    _ = try add(r, a, "My favourite colour is green.", &why);
    _ = try add(r, a, "My sister lives in Oslo.", &why);
    r.store().journal.?.faults = faults;
    return r;
}

/// Runs `case` once per index kind of rig.kinds, naming the kind when it fails.
fn eachKind(comptime case: anytype, args: anytype) !void {
    for (rig.kinds) |k| {
        errdefer std.debug.print("index {s}, side buffer {d}\n", .{ @tagName(k.kind), k.side_max });
        try @call(.auto, case, .{k} ++ args);
    }
}

test "(a) a write that fails while its rollback succeeds changes nothing, and the next write is stored" {
    try eachKind(rolledBack, .{});
}

fn rolledBack(k: rig.Kind) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var faults: journal_mod.Faults = .{};
    const r = try setup(a, &faults, k);
    defer r.deinit();
    const before = try r.journal(a);
    const notes = r.store().model.notes.items.len;
    faults.write = faults.writes;
    const got = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"I live in Helsinki now.\"}");
    try testing.expectEqual(@as(u16, 500), got.status);
    try testing.expectEqualStrings("the write could not be stored on disk; nothing changed", rig.message(got));
    try testing.expect(!r.store().journal.?.poisoned);
    try testing.expectEqualStrings(before, try r.journal(a)); // half the record was written, then cut away
    try testing.expectEqual(notes, r.store().model.notes.items.len);
    var why: ops.Refusal = .{};
    _ = try add(r, a, "I live in Helsinki now.", &why);
    try r.restart();
    try testing.expectEqual(@as(usize, 3), r.store().replayed);
}

/// (b) and (c): the write fails and so does its rollback's truncate, or its rollback's sync.
fn poisonedBy(k: rig.Kind, rollback_truncate_fails: bool) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var faults: journal_mod.Faults = .{};
    const r = try setup(a, &faults, k);
    defer r.deinit();
    const q = "What is my favourite colour?";
    const block = try blockOf(r, a, q);
    const notes = r.store().model.notes.items.len;
    const listed = (try rig.request(r.srv, a, "GET", "/v1/notes?history=1", null)).body;
    faults.write = faults.writes;
    if (rollback_truncate_fails) faults.truncate = faults.truncates else faults.sync = faults.syncs;
    const got = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"My favourite colour is purple.\"}");
    try testing.expectEqual(@as(u16, 500), got.status);
    try testing.expect(std.mem.indexOf(u8, rig.message(got), "nothing was applied in memory; the journal may hold this write, but not its commit line, so a restart drops it") != null);
    try testing.expect(r.store().journal.?.poisoned);
    // every further write is refused, the chat's own write too
    const next = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"I like tea.\"}");
    try testing.expectEqual(@as(u16, 503), next.status);
    try testing.expect(std.mem.startsWith(u8, rig.message(next), "the notes journal needs recovery, restart the server"));
    const erase = try rig.request(r.srv, a, "DELETE", "/v1/notes/30?erase=true", null);
    try testing.expectEqual(@as(u16, 503), erase.status);
    const chat = try rig.request(r.srv, a, "POST", "/v1/chat/completions", "{\"model\":\"m\",\"notes\":true,\"messages\":[{\"role\":\"user\",\"content\":\"I like tea.\"}]}");
    try testing.expectEqual(@as(u16, 200), chat.status);
    try testing.expect(std.mem.startsWith(u8, chat.value.get("notes").?.get("write").?.strField("failed").?, "the notes journal needs recovery"));
    // lookups go on, from the last good version: the same block and the same notes as before the failed write
    try testing.expectEqualStrings(block, try blockOf(r, a, q));
    try testing.expectEqual(notes, r.store().model.notes.items.len);
    try testing.expectEqualStrings(listed, (try rig.request(r.srv, a, "GET", "/v1/notes?history=1", null)).body);
}

test "(b) a write that fails and whose rollback truncate fails: poisoned, writes refused, lookups served" {
    try eachKind(poisonedBy, .{true});
}

test "(c) a write that fails and whose rollback sync fails: the same" {
    try eachKind(poisonedBy, .{false});
}

test "(d) a torn trailing record on disk: the restart cuts it and replays to the last whole record" {
    try eachKind(tornTail, .{});
}

fn tornTail(k: rig.Kind) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var faults: journal_mod.Faults = .{};
    const r = try setup(a, &faults, k);
    defer r.deinit();
    const whole = try r.journal(a);
    faults.write = faults.writes;
    faults.truncate = faults.truncates;
    var why: ops.Refusal = .{};
    try testing.expectError(error.Refused, add(r, a, "My favourite colour is purple.", &why));
    const torn = try r.journal(a);
    try testing.expect(torn.len > whole.len and torn[torn.len - 1] != '\n'); // half a record, no newline
    try testing.expect(std.mem.startsWith(u8, torn, whole));
    try r.restart();
    try testing.expectEqual(@as(usize, 2), r.store().replayed);
    try testing.expect(r.store().truncated == torn.len - whole.len);
    try testing.expectEqualStrings(whole, try r.journal(a));
    for (r.store().model.notes.items) |*n| try testing.expect(std.mem.indexOf(u8, n.text(), "purple") == null);
    try testing.expect(!r.store().journal.?.poisoned);
    _ = try add(r, a, "My favourite colour is purple.", &why); // and writes go on
    try r.restart();
    try testing.expectEqual(@as(usize, 3), r.store().replayed);
}

/// The rig after two good writes, a third whose append fails as `set` arranges, then a restart.
fn failThenRestart(a: std.mem.Allocator, k: rig.Kind, set: *const fn (*journal_mod.Faults) void) !struct { r: *rig.Rig, reply: []const u8, whole: []const u8, left: []const u8 } {
    var faults: journal_mod.Faults = .{};
    const r = try setup(a, &faults, k);
    errdefer r.deinit();
    const whole = try r.journal(a);
    set(&faults);
    const got = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"My favourite colour is purple.\"}");
    try testing.expectEqual(@as(u16, 500), got.status);
    try testing.expect(r.store().journal.?.poisoned);
    const left = try r.journal(a);
    try r.restart();
    return .{ .r = r, .reply = rig.message(got), .whole = whole, .left = left };
}

fn syncFailsThenTruncate(f: *journal_mod.Faults) void {
    f.sync = f.syncs; // the write's own sync
    f.truncate = f.truncates;
}

test "(e) a whole write whose sync fails and whose rollback truncate fails is never applied by the restart" {
    try eachKind(neverApplied, .{});
}

fn neverApplied(k: rig.Kind) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try failThenRestart(a, k, syncFailsThenTruncate);
    defer got.r.deinit();
    try testing.expect(std.mem.indexOf(u8, got.reply, "so a restart drops it") != null);
    // on disk: the whole write line, newline and all, but no commit line after it
    try testing.expect(std.mem.startsWith(u8, got.left, got.whole) and got.left.len > got.whole.len);
    try testing.expect(got.left[got.left.len - 1] == '\n' and std.mem.indexOf(u8, got.left[got.whole.len..], "commit") == null);
    try testing.expectEqual(@as(usize, 2), got.r.store().replayed);
    try testing.expectEqual(@as(u64, got.left.len - got.whole.len), got.r.store().truncated);
    for (got.r.store().model.notes.items) |*n| try testing.expect(std.mem.indexOf(u8, n.text(), "purple") == null);
    try testing.expectEqualStrings(got.whole, try got.r.journal(a));
}

fn commitSyncFailsThenTruncate(f: *journal_mod.Faults) void {
    f.sync = f.syncs + 1; // the commit line's sync
    f.truncate = f.truncates;
}

test "(f) a commit line written whose sync fails and whose rollback fails: the reply says a restart may apply it" {
    try eachKind(mayApply, .{});
}

fn mayApply(k: rig.Kind) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try failThenRestart(a, k, commitSyncFailsThenTruncate);
    defer got.r.deinit();
    try testing.expect(std.mem.indexOf(u8, got.reply, "the disk state is unknown and a restart may apply this write") != null);
    // here the page cache kept the commit line, so the restart applies the write (the documented case)
    try testing.expectEqual(@as(usize, 3), got.r.store().replayed);
}

test "(g) a format-1 journal (no commit lines) is replayed whole and rewritten as format 2 under the lock" {
    try eachKind(formatOneUpgraded, .{});
}

fn formatOneUpgraded(k: rig.Kind) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var faults: journal_mod.Faults = .{};
    const r = try setup(a, &faults, k);
    defer r.deinit();
    const v2 = try r.journal(a);
    r.srv.notes.?.deinit();
    r.srv.notes = null;
    try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json.journal", .data = try formatOne(a, v2) });
    try r.open();
    try testing.expectEqual(@as(usize, 2), r.store().replayed);
    try testing.expect(r.store().journal.?.upgraded);
    try testing.expectEqualStrings(v2, try r.journal(a));
    var why: ops.Refusal = .{};
    _ = try add(r, a, "I like tea.", &why);
    try r.restart();
    try testing.expectEqual(@as(usize, 3), r.store().replayed);
}

/// The rig's folder at `mode`: 0o300 lets files be made, renamed and removed in it, but not the folder be opened.
fn folderMode(r: *rig.Rig, a: std.mem.Allocator, mode: std.c.mode_t) !void {
    const dir = try a.dupeSentinel(u8, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{r.tmp.sub_path}), 0);
    try testing.expectEqual(@as(c_int, 0), std.c.chmod(dir, mode));
}

/// The folder made one that cannot be opened for its sync; false where that does not stop an open (root).
fn folderUnopenable(r: *rig.Rig, a: std.mem.Allocator) !bool {
    try folderMode(r, a, 0o300);
    const dir = try a.dupeSentinel(u8, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{r.tmp.sub_path}), 0);
    const fd = std.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true });
    if (fd < 0) return true;
    _ = std.c.close(fd);
    try folderMode(r, a, 0o700);
    return false;
}

const FolderFault = enum { hook, fullfsync_eio, unopenable };

/// Arranges the next folder sync to fail: by the hook, by its F_FULLFSYNC (the call after the file's), or for real.
fn failFolder(r: *rig.Rig, a: std.mem.Allocator, faults: *journal_mod.Faults, how: FolderFault) !bool {
    switch (how) {
        .hook => faults.dir = faults.dirs,
        .fullfsync_eio => {
            if (!builtin.os.tag.isDarwin()) return false;
            faults.full = faults.fulls + 1; // the file's own F_FULLFSYNC first, then the folder's
            faults.full_errno = .IO;
        },
        .unopenable => return folderUnopenable(r, a),
    }
    return true;
}

/// (h): the first write, whose new journal's folder cannot be synced: refused, no journal left, nothing changed.
fn createFolderFails(k: rig.Kind, how: FolderFault) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var faults: journal_mod.Faults = .{};
    const r = try rig.Rig.initWith(testing.allocator, k);
    defer r.deinit();
    defer folderMode(r, a, 0o700) catch {};
    r.store().journal.?.faults = &faults;
    if (!try failFolder(r, a, &faults, how)) return error.SkipZigTest;
    const notes = r.store().model.notes.items.len;
    const got = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"I live in Helsinki now.\"}");
    try folderMode(r, a, 0o700);
    try testing.expectEqual(@as(u16, 500), got.status);
    try testing.expectEqualStrings("the write could not be stored on disk; nothing changed", rig.message(got));
    try testing.expectError(error.FileNotFound, r.journal(a)); // the header-only file is removed again
    try testing.expectEqual(notes, r.store().model.notes.items.len);
    try testing.expect(!r.store().journal.?.poisoned);
    const next = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"I live in Helsinki now.\"}");
    try testing.expectEqual(@as(u16, 201), next.status);
    try r.restart();
    try testing.expectEqual(@as(usize, 1), r.store().replayed);
}

test "(h) the journal's folder sync fails at its create (by the hook): the first write refused, nothing changed" {
    try eachKind(createFolderFails, .{FolderFault.hook});
}

test "(h) the folder's F_FULLFSYNC fails with EIO at the create: the same" {
    try eachKind(createFolderFails, .{FolderFault.fullfsync_eio});
}

test "(h) the folder cannot be opened for its sync at the create: the same" {
    try eachKind(createFolderFails, .{FolderFault.unopenable});
}

fn idOf(r: *rig.Rig, source: []const u8) u32 {
    for (r.store().model.notes.items) |*n| if (n.source()) |s| if (std.mem.eql(u8, s, source)) return n.id;
    unreachable;
}

fn holds(r: *rig.Rig, text: []const u8) bool {
    for (r.store().model.notes.items) |*n| if (std.mem.indexOf(u8, n.text(), text) != null) return true;
    return false;
}

/// (i): an erase whose new journal is renamed in but whose folder cannot be synced: applied, the disk state unknown.
fn eraseFolderFails(k: rig.Kind, how: FolderFault) !void {
    return eraseFolderFailsAs(k, how, false);
}

/// eraseFolderFails(); `deferred`: the old model and base left to a later merge (the erase tests' control).
fn eraseFolderFailsAs(k: rig.Kind, how: FolderFault, deferred: bool) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var faults: journal_mod.Faults = .{};
    const r = try setup(a, &faults, k);
    defer r.deinit();
    defer folderMode(r, a, 0o700) catch {};
    const before = try r.journal(a);
    try testing.expect(std.mem.indexOf(u8, before, "Oslo") != null and holds(r, "Oslo"));
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{idOf(r, "w2")});
    r.store().defer_erase_merge = deferred;
    if (!try failFolder(r, a, &faults, how)) return error.SkipZigTest;
    const got = try rig.request(r.srv, a, "DELETE", path, null);
    try folderMode(r, a, 0o700);
    try testing.expectEqual(@as(u16, 500), got.status);
    const said = rig.message(got);
    try testing.expect(std.mem.indexOf(u8, said, "the disk state is unknown: after a power cut either the old journal (with the note) or the new one (without it) may be there") != null);
    try testing.expect(std.mem.indexOf(u8, said, "The erase was applied in memory") != null);
    try testing.expect(std.mem.indexOf(u8, said, "nothing") == null); // never "nothing changed" or "nothing was erased"
    try testing.expect(r.store().journal.?.poisoned);
    // the file at the journal's name is the new one, and the memory agrees with it
    const after = try r.journal(a);
    try testing.expect(std.mem.indexOf(u8, after, "Oslo") == null and std.mem.indexOf(u8, after, "green") != null);
    try testing.expect(!holds(r, "Oslo") and holds(r, "green"));
    try testing.expect(std.mem.indexOf(u8, (try rig.request(r.srv, a, "GET", "/v1/notes?history=1", null)).body, "Oslo") == null);
    const next = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"I like tea.\"}");
    try testing.expectEqual(@as(u16, 503), next.status);
    try testing.expect(std.mem.startsWith(u8, rig.message(next), "the notes journal needs recovery, restart the server"));
    // no power cut: the restart reads the new file, and writes go on
    try r.restart();
    try testing.expectEqual(@as(usize, 1), r.store().replayed);
    try testing.expect(!holds(r, "Oslo") and !r.store().journal.?.poisoned);
    var why: ops.Refusal = .{};
    _ = try add(r, a, "I like tea.", &why);
}

test "(i) an erase's folder sync fails after its rename (by the hook): applied, poisoned, the disk state unknown" {
    try eachKind(eraseFolderFails, .{FolderFault.hook});
}

test "(i) an erase's folder F_FULLFSYNC fails with EIO after the rename: the same" {
    try eachKind(eraseFolderFails, .{FolderFault.fullfsync_eio});
}

test "(i) an erase's folder cannot be opened for its sync after the rename: the same" {
    try eachKind(eraseFolderFails, .{FolderFault.unopenable});
}

test "(i) an erase left to a later merge whose folder sync fails after the rename: the same reply" {
    try eraseFolderFailsAs(.{ .kind = .sa }, .hook, true);
}

/// A format-1 journal of the rig's two writes: the header's format, and no commit lines.
fn formatOne(a: std.mem.Allocator, v2: []const u8) ![]u8 {
    var v1: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, v2, '\n');
    while (lines.next()) |l| {
        if (l.len == 0 or std.mem.startsWith(u8, l, "{\"commit\"")) continue;
        try v1.appendSlice(a, try std.mem.replaceOwned(u8, a, l, "tf-notes-journal/2", "tf-notes-journal/1"));
        try v1.append(a, '\n');
    }
    return v1.items;
}

/// The store opened with `faults` from the start: the problem that stopped it.
fn openFails(r: *rig.Rig, faults: *journal_mod.Faults) ![]const u8 {
    journal_mod.faults_at_open = faults;
    defer journal_mod.faults_at_open = null;
    var problem: []const u8 = "";
    const s = notes_route.Store.openIndexed(r.gpa, testing.io, r.snap, r.text.text(), "", r.kind, &problem) catch |e| {
        try testing.expectEqual(error.JournalIo, e);
        return problem;
    };
    s.deinit();
    return error.TestUnexpectedResult;
}

test "(i) the format-1 upgrade at startup: a failure before the rename changes nothing, one after it stops the start" {
    try eachKind(upgradeFails, .{});
}

fn upgradeFails(k: rig.Kind) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var unused: journal_mod.Faults = .{};
    const r = try setup(a, &unused, k);
    defer r.deinit();
    const v2 = try r.journal(a);
    const v1 = try formatOne(a, v2);
    r.srv.notes.?.deinit();
    r.srv.notes = null;
    try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json.journal", .data = v1 });
    // the new file's sync fails, before the rename: the format-1 file stays as it was
    var before: journal_mod.Faults = .{ .sync = 0 };
    try testing.expectEqualStrings("the notes journal (format 1) could not be rewritten as format 2; nothing changed", try openFails(r, &before));
    try testing.expectEqualStrings(v1, try r.journal(a));
    // the folder's sync fails, after the rename: the start stops, and says either file may be there
    var after: journal_mod.Faults = .{ .dir = 0 };
    const problem = try openFails(r, &after);
    try testing.expect(std.mem.indexOf(u8, problem, "renamed in, but its folder could not be synced: after a power cut either file may be there, each whole") != null);
    try testing.expect(std.mem.indexOf(u8, problem, "nothing changed") == null);
    try testing.expectEqualStrings(v2, try r.journal(a)); // the new file is at the journal's name
    try r.open(); // the next start reads it as format 2
    try testing.expectEqual(@as(usize, 2), r.store().replayed);
    try testing.expect(!r.store().journal.?.upgraded);
}

/// The rig after two writes, the third's first F_FULLFSYNC failing with `e`: the reply, and the faults after it.
fn fullSyncFails(a: std.mem.Allocator, k: rig.Kind, e: std.c.E) !struct { r: *rig.Rig, got: rig.Reply, before: []const u8, fallbacks: u32 } {
    var faults: journal_mod.Faults = .{};
    const r = try setup(a, &faults, k);
    errdefer r.deinit();
    const before = try r.journal(a);
    faults.full = faults.fulls;
    faults.full_errno = e;
    const got = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"I live in Helsinki now.\"}");
    r.store().journal.?.faults = null;
    return .{ .r = r, .got = got, .before = before, .fallbacks = faults.fallbacks };
}

test "(j) F_FULLFSYNC failing with EIO, ENODEV or EBADF is a failed sync: no plain fsync stands in, nothing changed" {
    if (!builtin.os.tag.isDarwin()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (rig.kinds) |k| {
        for ([_]std.c.E{ .IO, .NODEV, .BADF }) |e| {
            errdefer std.debug.print("index {s}, side buffer {d}, errno {s}\n", .{ @tagName(k.kind), k.side_max, @tagName(e) });
            const x = try fullSyncFails(a, k, e);
            defer x.r.deinit();
            try testing.expectEqual(@as(u16, 500), x.got.status);
            try testing.expectEqualStrings("the write could not be stored on disk; nothing changed", rig.message(x.got));
            try testing.expectEqual(@as(u32, 0), x.fallbacks);
            try testing.expectEqualStrings(x.before, try x.r.journal(a));
            try testing.expect(!x.r.store().journal.?.poisoned and !holds(x.r, "Helsinki"));
        }
    }
}

test "(k) F_FULLFSYNC unsupported (ENOTTY, ENOTSUP, EINVAL): plain fsync instead, the write stored" {
    if (!builtin.os.tag.isDarwin()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (rig.kinds) |k| {
        for ([_]std.c.E{ .NOTTY, .OPNOTSUPP, .INVAL }) |e| {
            errdefer std.debug.print("index {s}, side buffer {d}, errno {s}\n", .{ @tagName(k.kind), k.side_max, @tagName(e) });
            const x = try fullSyncFails(a, k, e);
            defer x.r.deinit();
            try testing.expectEqual(@as(u16, 201), x.got.status);
            try testing.expectEqual(@as(u32, 1), x.fallbacks);
            try x.r.restart();
            try testing.expectEqual(@as(usize, 3), x.r.store().replayed);
            try testing.expect(holds(x.r, "Helsinki"));
        }
    }
}
