//! /v1/notes writes against LLWZ Python: the fixture's write script over HTTP, then every case and the notes compared.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const rig = @import("notes_rig_test.zig");
const past = @import("notes/past.zig");
const Bytes = @import("notes_test.zig").Bytes;
const testing = std.testing;
const Value = json.Value;

fn int(v: ?Value) ?i64 {
    const x = v orelse return null;
    return if (x == .null) null else x.int64();
}

fn sameInts(want: ?Value, got: ?Value) bool {
    const w = (want orelse return false).items() orelse return false;
    const g = (got orelse return false).items() orelse return false;
    if (w.len != g.len) return false;
    for (w, g) |x, y| if (x.int64() != y.int64()) return false;
    return true;
}

/// The note id a step's target names now: a snapshot note's id, or the note a write made (looked up by source).
fn targetId(r: *rig.Rig, t: Value) ?u32 {
    if (int(t.get("note"))) |n| return @intCast(n);
    const src = t.strField("write") orelse return null;
    const m = r.store().model;
    const p = m.placeOfSource(src) orelse return null;
    return m.notes.items[p].id;
}

/// One write step over HTTP (the three change forms in turn); the differences from Python's reply, counted.
fn step(r: *rig.Rig, a: std.mem.Allocator, s: Value, k: usize) !usize {
    const op = s.strField("op").?;
    const want = s.get("want").?;
    const text = s.strField("text");
    const quoted = if (text) |t| try json.quote(a, t, .{}) else "";
    const target = if (s.get("target")) |t| targetId(r, t) else null;
    if (s.get("target") != null and target == null) return 1; // a control's left-out write: its note is not there
    if (std.mem.eql(u8, op, "erase")) {
        const id = target.?;
        if (int(want.get("note")).? != id) return 1;
        const got = try rig.request(r.srv, a, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id}), null);
        if (got.status != 200) return 1;
        var bad: usize = 0;
        const removed = got.value.get("removed").?.array;
        const wr = want.get("removed").?.array;
        if (removed.len != wr.len) bad += 1 else for (removed, wr) |x, y| if (!std.mem.eql(u8, x.string, y.string)) {
            bad += 1;
        };
        const moved = got.value.get("renumbered").?.array;
        const wm = want.get("moved").?.array;
        if (moved.len != wm.len) bad += 1 else for (moved, wm) |x, y| if (int(x.get("from")) != y.array[0].int64() or int(x.get("to")) != y.array[1].int64()) {
            bad += 1;
        };
        return bad;
    }
    var got: rig.Reply = undefined;
    if (std.mem.eql(u8, op, "add")) {
        const subject = if (s.strField("subject")) |x| try std.fmt.allocPrint(a, ",\"subject\":{s}", .{try json.quote(a, x, .{})}) else "";
        got = try rig.request(r.srv, a, "POST", "/v1/notes", try std.fmt.allocPrint(a, "{{\"text\":{s}{s}}}", .{ quoted, subject }));
    } else {
        const id = target.?;
        const path = try std.fmt.allocPrint(a, "/v1/notes/{d}", .{id});
        if (std.mem.eql(u8, op, "change")) {
            got = switch (k % 3) {
                0 => try rig.request(r.srv, a, "POST", path, try std.fmt.allocPrint(a, "{{\"text\":{s}}}", .{quoted})),
                1 => try rig.request(r.srv, a, "PUT", path, try std.fmt.allocPrint(a, "{{\"text\":{s}}}", .{quoted})),
                else => try rig.request(r.srv, a, "POST", "/v1/notes", try std.fmt.allocPrint(a, "{{\"text\":{s},\"replaces\":{d}}}", .{ quoted, id })),
            };
        } else got = try rig.request(r.srv, a, "DELETE", path, if (text != null) try std.fmt.allocPrint(a, "{{\"text\":{s}}}", .{quoted}) else null);
    }
    const v = got.value;
    var bad: usize = 0;
    const verdict = want.strField("verdict").?;
    if (!std.mem.eql(u8, v.strField("verdict") orelse "", verdict)) bad += 1;
    const learn = std.mem.eql(u8, verdict, "NEW") or std.mem.eql(u8, verdict, "REPLACES");
    if (got.status != @as(u16, if (learn) 201 else 200)) bad += 1;
    if (int(v.get("id")) != int(want.get("id"))) bad += 1;
    if (int(v.get("version")) != int(want.get("version"))) bad += 1;
    if (!std.mem.eql(u8, v.strField("state") orelse "", want.strField("state") orelse "")) bad += 1;
    if (!std.mem.eql(u8, v.strField("source").?, want.strField("source").?)) bad += 1;
    if (int(v.get("told")) != int(want.get("told"))) bad += 1;
    if (!sameInts(want.get("retired"), v.get(if (std.mem.eql(u8, op, "take_back")) "tombstoned" else "superseded"))) bad += 1;
    if (!sameInts(want.get("same_as"), v.get("same_as"))) bad += 1;
    return bad;
}

/// A checkpoint: every fixture case looked up as Python's retrieve() answered it, and the notes as Python lists them.
fn check(r: *rig.Rig, a: std.mem.Allocator, cp: Value) !usize {
    const text = r.text.text();
    const cases = (try json.parse(a, rig.cases_json)).ok.array;
    const want = cp.get("cases").?.array;
    var bad: usize = 0;
    for (cases, want) |c, w| {
        const q = c.strField("question").?;
        const read = if (c.field("past")) |g| g.bool else past.isPast(q);
        var ex: std.ArrayList(u32) = .empty;
        for (c.get("exclude").?.array) |x| try ex.append(a, @intCast(x.int64().?));
        const cap: u32 = if (c.field("cap")) |x| @intCast(x.int64().?) else r.store().settings().cap;
        const got = try r.store().retrieve(a, text, try text.encode(a, q, false), read, cap, ex.items);
        var same = got.tokens == @as(usize, @intCast(int(w.get("tokens")).?)) and got.candidates == int(w.get("candidates")).?;
        const ids = w.get("ids").?.array;
        same = same and ids.len == got.ids.len;
        if (same) for (ids, got.ids) |x, y| {
            same = same and x.int64().? == y;
        };
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(got.block, &sha, .{});
        if (w.field("block_sha256")) |h| {
            same = same and std.mem.eql(u8, h.string, &std.fmt.bytesToHex(sha, .lower));
        } else same = same and got.block.len == 0;
        if (!same) bad += 1;
    }
    const listed = try rig.request(r.srv, a, "GET", "/v1/notes?history=1", null);
    const notes = listed.value.get("data").?.array;
    const wn = cp.get("notes").?.array;
    if (notes.len != wn.len) return bad + 1;
    for (notes, wn) |n, x| {
        var same = int(n.get("id")) == int(x.get("id")) and int(n.get("version")) == int(x.get("version"));
        for ([_][]const u8{ "state", "subject", "text", "line" }) |f| same = same and std.mem.eql(u8, n.strField(f).?, x.strField(f).?);
        same = same and std.mem.eql(u8, n.strField("kind").?, if (std.mem.eql(u8, x.strField("kind").?, "fact")) "fact" else "take_back");
        const src = x.strField("source");
        same = same and (if (src) |s| std.mem.eql(u8, n.strField("source") orelse "", s) else n.get("source").? == .null);
        if (!same) bad += 1;
    }
    return bad;
}

/// The script run on a fresh rig, `skip` (a step index) left out; differences per checkpoint name.
fn run(a: std.mem.Allocator, skip: ?usize, restart: bool) !Ran {
    return runKind(a, skip, restart, .postings, 1024);
}

const Ran = struct { replies: usize, checks: [3]usize, after_restart: usize, merges: u64 = 0, side_at_checks: [3]usize = .{ 0, 0, 0 } };

/// run() on a store with the n-gram index of `kind` (sa: a merge once the side buffer passes `side_max` notes).
fn runKind(a: std.mem.Allocator, skip: ?usize, restart: bool, kind: @import("notes_store.zig").IndexKind, side_max: u32) !Ran {
    const r = try rig.Rig.initKind(testing.allocator, kind, side_max);
    defer r.deinit();
    const doc = (try json.parse(a, rig.writes_json)).ok;
    var replies: usize = 0;
    var checks: [3]usize = .{ 0, 0, 0 };
    var side: [3]usize = .{ 0, 0, 0 };
    var last: Value = .null;
    for (doc.get("steps").?.array, 0..) |s, k| {
        if (s.strField("check")) |name| {
            checks[name[0] - 'A'] = try check(r, a, s);
            const ix = r.store().current.idx;
            side[name[0] - 'A'] = if (ix.base) |b| ix.notes() - b.notes else 0;
            last = s;
            continue;
        }
        if (skip == k) continue;
        replies += try step(r, a, s, k);
    }
    var after: usize = 0;
    const merges = r.store().merges;
    if (restart) {
        try r.restart();
        after = try check(r, a, last);
    }
    return .{ .replies = replies, .checks = checks, .after_restart = after, .merges = merges, .side_at_checks = side };
}

test "the suffix-array index: the fixture's writes give Python's replies, blocks and notes with the side buffer never merged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const got = try runKind(arena.allocator(), null, true, .sa, 1_000_000);
    try testing.expectEqual(@as(usize, 0), got.replies);
    try testing.expectEqual([3]usize{ 0, 0, 0 }, got.checks);
    try testing.expectEqual(@as(usize, 0), got.after_restart);
    try testing.expect(got.side_at_checks[0] > 0 and got.side_at_checks[1] > 0); // the lookups read the side buffer
}

test "the suffix-array index: the same with a merge at every write, and with one every third note" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    for ([_]u32{ 0, 3 }) |side_max| {
        const got = try runKind(arena.allocator(), null, true, .sa, side_max);
        try testing.expectEqual(@as(usize, 0), got.replies);
        try testing.expectEqual([3]usize{ 0, 0, 0 }, got.checks);
        try testing.expectEqual(@as(usize, 0), got.after_restart);
        try testing.expect(got.merges > 0);
        if (side_max == 0) try testing.expectEqual([3]usize{ 0, 0, 0 }, got.side_at_checks);
    }
}

test "negative control: the suffix-array index with one take-back left out differs from Python at the next checkpoint" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const got = try runKind(arena.allocator(), 7, false, .sa, 1_000_000);
    try testing.expectEqual(@as(usize, 0), got.checks[0]);
    try testing.expect(got.checks[1] > 0);
}

test "the fixture's writes over HTTP give LLWZ Python's replies, blocks and notes, and the same after a restart" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const got = try run(arena.allocator(), null, true);
    try testing.expectEqual(@as(usize, 0), got.replies);
    try testing.expectEqual([3]usize{ 0, 0, 0 }, got.checks);
    try testing.expectEqual(@as(usize, 0), got.after_restart);
}

test "negative control: the same script with one take-back left out differs from Python at the next checkpoint" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const got = try run(arena.allocator(), 7, false); // step 7: the take-back of note 28 ("I like pears.")
    try testing.expectEqual(@as(usize, 0), got.checks[0]);
    try testing.expect(got.checks[1] > 0);
    try testing.expect(got.replies > 0); // the later writes' ids move
}

test "norm() gives what LLWZ's intake.norm gives, ASCII letters lowered, for the fixture's texts" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, rig.writes_json)).ok;
    const intake = @import("notes/intake.zig");
    for (doc.get("norm").?.array) |pair| try testing.expectEqualStrings(pair.array[1].string, try intake.norm(a, pair.array[0].string));
}
