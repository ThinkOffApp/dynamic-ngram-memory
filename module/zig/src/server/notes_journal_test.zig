//! The journal: writes survive a restart, a torn last line is cut, and a bad, foreign or held journal is refused.
const std = @import("std");
const server = @import("root.zig");
const rig = @import("notes_rig_test.zig");
const ops = @import("notes_ops.zig");
const notes_route = @import("notes_route.zig");
const testing = std.testing;

fn add(r: *rig.Rig, a: std.mem.Allocator, text: []const u8) !ops.Written {
    var why: ops.Refusal = .{};
    return ops.perform(r.store(), a, .{ .op = .add, .text = text }, &why);
}

fn blockOf(r: *rig.Rig, a: std.mem.Allocator, q: []const u8) ![]const u8 {
    const t = r.text.text();
    return (try r.store().retrieve(a, t, try t.encode(a, q, false), false, 512, &.{})).block;
}

test "writes are on disk before the reply, and a restart gives the same notes and the same blocks" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectError(error.FileNotFound, r.journal(a)); // no write yet: no journal
    _ = try add(r, a, "My favourite colour is green.");
    const after_one = try r.journal(a);
    try testing.expect(std.mem.indexOf(u8, after_one, "My favourite colour is green.") != null);
    var why: ops.Refusal = .{};
    _ = try ops.perform(r.store(), a, .{ .op = .take_back, .text = "I take that back.", .target = 28 }, &why);
    const q = [_][]const u8{ "What is my favourite colour?", "Do I like pears?", "Did I like pears before?" };
    var before: [q.len][]const u8 = undefined;
    for (q, &before) |x, *b| b.* = try blockOf(r, a, x);
    try r.restart();
    try testing.expectEqual(@as(usize, 2), r.store().replayed);
    for (q, before) |x, b| try testing.expectEqualStrings(b, try blockOf(r, a, x));
    try testing.expectEqual(@as(usize, 31), r.store().model.notes.items.len);
}

test "a torn last line is cut at startup, and the writes before it stand" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try add(r, a, "My favourite colour is green.");
    _ = try add(r, a, "I like long walks.");
    const whole = try r.journal(a);
    r.srv.notes.?.deinit();
    r.srv.notes = null;
    const torn = try std.mem.concat(a, u8, &.{ whole, "{\"op\":\"add\",\"source\":\"w3\",\"told\":7399" });
    try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json.journal", .data = torn });
    try r.open();
    try testing.expectEqual(@as(usize, 2), r.store().replayed);
    try testing.expect(r.store().truncated > 0);
    try testing.expectEqualStrings(whole, try r.journal(a));
    const third = try add(r, a, "I live in Oslo.");
    try testing.expectEqualStrings("w3", third.source);
    try r.restart();
    try testing.expectEqual(@as(usize, 3), r.store().replayed);
}

test "a journal with a bad line, one for another snapshot, or one another server holds is refused" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try add(r, a, "My favourite colour is green.");
    const whole = try r.journal(a);
    // another server on the same files: the journal is locked
    var problem: []const u8 = "";
    try testing.expectError(error.Locked, notes_route.Store.open(testing.allocator, testing.io, r.snap, r.text.text(), "", &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "another process") != null);
    r.srv.notes.?.deinit();
    r.srv.notes = null;
    const bad = try std.mem.concat(a, u8, &.{ whole, "not json\n" });
    try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json.journal", .data = bad });
    try testing.expectError(error.BadJournal, notes_route.Store.open(testing.allocator, testing.io, r.snap, r.text.text(), "", &problem));
    const other = try std.mem.replaceOwned(u8, a, whole, "\"snapshot_sha256\":\"", "\"snapshot_sha256\":\"0");
    try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json.journal", .data = other });
    try testing.expectError(error.OtherSnapshot, notes_route.Store.open(testing.allocator, testing.io, r.snap, r.text.text(), "", &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "another snapshot") != null);
    try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json.journal", .data = whole });
    try r.open();
    try testing.expectEqual(@as(usize, 1), r.store().replayed);
}

test "a write the journal can neither store nor roll back is a 500 that applies nothing and stops writes" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try add(r, a, "My favourite colour is green.");
    const notes = r.store().model.notes.items.len;
    const journal = try r.journal(a);
    // the journal's descriptor swapped for one that can be neither written nor truncated (the rollback fails too)
    const ro = std.c.open("/dev/null", .{ .ACCMODE = .RDONLY });
    try testing.expect(ro >= 0);
    const fd = r.store().journal.?.fd;
    try testing.expect(std.c.dup2(ro, fd) >= 0);
    _ = std.c.close(ro);
    const got = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"I like long walks.\"}");
    try testing.expectEqual(@as(u16, 500), got.status);
    try testing.expect(std.mem.indexOf(u8, rig.message(got), "nothing was applied in memory") != null);
    try testing.expect(r.store().journal.?.poisoned);
    try testing.expectEqual(@as(u16, 503), (try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"I like tea.\"}")).status);
    try testing.expectEqual(notes, r.store().model.notes.items.len);
    try testing.expectEqual(@as(u64, 1), r.store().current.idx.ver); // still the first write's version
    try testing.expectEqualStrings(journal, try r.journal(a));
    const listed = try rig.request(r.srv, a, "GET", "/v1/notes", null);
    for (listed.value.get("data").?.array) |n| try testing.expect(!std.mem.eql(u8, n.strField("text").?, "I like long walks."));
}
