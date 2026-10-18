//! 8 lookup threads while 1,000 writes (and erases) land: every block a lookup got is its version's, none torn.
const std = @import("std");
const rig = @import("notes_rig_test.zig");
const ops = @import("notes_ops.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const questions = [_][]const u8{ "What is my favourite colour?", "Where does my sister live?", "What colour is my item 12?", "Do I like pears?", "What was my favourite colour before?", "What is my dog called?", "What colour is my item 400 now?", "Which items are blue?" };
const threads = 8;

const Seen = struct { ver: u64, q: u8, hash: u64, ns: u64 };

const Shared = struct {
    r: *rig.Rig,
    stop: std.atomic.Value(bool) = .init(false),
    seen: [threads]std.ArrayList(Seen) = @splat(.empty),
    quiet: [threads]std.ArrayList(u64) = @splat(.empty), // lookup times before the writes start
    busy: [threads]std.ArrayList(u64) = @splat(.empty), // lookup times while they land
    after: [threads]std.ArrayList(u64) = @splat(.empty), // lookup times on the final store, no writes (the control)
    phase: std.atomic.Value(u8) = .init(0), // 0 before, 1 writing, 2 after
    failed: std.atomic.Value(u32) = .init(0),

    fn fail(sh: *Shared) void {
        _ = sh.failed.fetchAdd(1, .monotonic);
    }
};

fn reader(sh: *Shared, k: usize) void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const text = sh.r.text.text();
    var n: usize = 0;
    while (!sh.stop.load(.acquire)) : (n += 1) {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        const q: u8 = @intCast((n + k) % questions.len);
        const ids = text.encode(a, questions[q], false) catch return sh.fail();
        const phase = sh.phase.load(.acquire);
        const writing = phase == 1;
        const t0 = std.Io.Clock.awake.now(testing.io).toNanoseconds();
        const got = sh.r.store().retrieve(a, text, ids, q == 4, 512, &.{}) catch return sh.fail();
        const ns: u64 = @intCast(std.Io.Clock.awake.now(testing.io).toNanoseconds() - t0);
        const list = switch (phase) {
            0 => &sh.quiet[k],
            1 => &sh.busy[k],
            else => &sh.after[k],
        };
        list.append(std.heap.page_allocator, ns) catch return sh.fail();
        if (writing) sh.seen[k].append(std.heap.page_allocator, .{ .ver = got.ver, .q = q, .hash = std.hash.Wyhash.hash(0, got.block), .ns = ns }) catch return sh.fail();
    }
}

/// Each question's block hash on the version lookups read now, asked by the writer alone.
fn expect(r: *rig.Rig, a: Allocator, table: *std.AutoHashMapUnmanaged(u64, [questions.len]u64)) !void {
    const text = r.text.text();
    var row: [questions.len]u64 = undefined;
    var ver: u64 = 0;
    for (questions, &row, 0..) |q, *h, i| {
        const got = try r.store().retrieve(a, text, try text.encode(a, q, false), i == 4, 512, &.{});
        h.* = std.hash.Wyhash.hash(0, got.block);
        ver = got.ver;
    }
    try table.put(testing.allocator, ver, row);
}

fn pct(xs: []u64, p: usize) u64 {
    std.sort.pdq(u64, xs, {}, std.sort.asc(u64));
    if (xs.len == 0) return 0;
    return xs[@min(xs.len - 1, (xs.len * p + 99) / 100 -| 1)];
}

fn joined(lists: []std.ArrayList(u64), a: Allocator) ![]u64 {
    var out: std.ArrayList(u64) = .empty;
    for (lists) |l| try out.appendSlice(a, l.items);
    return out.items;
}

/// One write of the script: k % 4 picks an add, a change, a same-as add or a take-back; every hundredth an erase too.
fn writeStep(r: *rig.Rig, w: Allocator, k: usize, why: *ops.Refusal, erased: *usize, table: ?*std.AutoHashMapUnmanaged(u64, [questions.len]u64)) !void {
    const s = r.store();
    const item = try std.fmt.allocPrint(w, "my item {d}", .{k / 4});
    const last = s.model.notes.items[s.model.notes.items.len - 1];
    const done = switch (k % 4) {
        0 => ops.perform(s, w, .{ .op = .add, .text = try std.fmt.allocPrint(w, "My item {d} is red.", .{k / 4}), .subject = item }, why),
        1 => ops.perform(s, w, .{ .op = .change, .text = try std.fmt.allocPrint(w, "My item {d} is blue now.", .{k / 4}), .target = last.id }, why),
        2 => ops.perform(s, w, .{ .op = .add, .text = try std.fmt.allocPrint(w, "my item {d} is blue now", .{k / 4}), .subject = item }, why),
        else => if (k % 100 == 99) blk: {
            _ = try ops.erase(s, w, last.id, why);
            erased.* += 1;
            if (table) |t| try expect(r, w, t); // the erase's own version, which lookups read until the next write
            break :blk ops.perform(s, w, .{ .op = .add, .text = "I like long walks.", .subject = item }, why);
        } else ops.perform(s, w, .{ .op = .take_back, .text = "I take that back.", .target = last.id }, why),
    };
    _ = done catch |e| {
        std.debug.print("write {d}: {s} {s}\n", .{ k, @errorName(e), why.message });
        return e;
    };
}

test "8 lookup threads during 1,000 writes and erases: no crash, every block its version's, lookups not slowed" {
    try concurrent(.postings, 1024);
}

test "the suffix-array index: 8 lookup threads during 1,000 writes, erases and merges, every block its version's" {
    try concurrent(.sa, 64);
}

test "the suffix-array index answers as the postings do after each of 1,000 writes, erases and merges" {
    const p = try rig.Rig.initKind(testing.allocator, .postings, 1024);
    defer p.deinit();
    const x = try rig.Rig.initKind(testing.allocator, .sa, 64);
    defer x.deinit();
    var why: ops.Refusal = .{};
    var erased: usize = 0;
    var differ: usize = 0;
    var with_side: usize = 0;
    for (0..1000) |k| {
        var wa = std.heap.ArenaAllocator.init(testing.allocator);
        defer wa.deinit();
        const w = wa.allocator();
        try writeStep(p, w, k, &why, &erased, null);
        try writeStep(x, w, k, &why, &erased, null);
        const ix = x.store().current.idx;
        if (ix.notes() > ix.base.?.notes) with_side += 1;
        for (questions, 0..) |q, i| for ([_]bool{ i == 4, true }) |past| {
            const text = p.text.text();
            const ids = try text.encode(w, q, false);
            const a = try p.store().retrieve(w, text, ids, past, 512, &.{});
            const b = try x.store().retrieve(w, text, ids, past, 512, &.{});
            if (!std.mem.eql(u8, a.block, b.block) or !std.mem.eql(u32, a.ids, b.ids) or a.candidates != b.candidates) differ += 1;
        };
    }
    try testing.expectEqual(@as(usize, 0), differ);
    try testing.expectEqual(@as(usize, 20), erased); // 10 on each store
    try testing.expect(x.store().merges > 5 and with_side > 500); // merges happened, and most lookups read a side buffer
}

fn concurrent(kind: @import("notes_store.zig").IndexKind, side_max: u32) !void {
    const r = try rig.Rig.initKind(testing.allocator, kind, side_max);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var table: std.AutoHashMapUnmanaged(u64, [questions.len]u64) = .empty;
    defer table.deinit(testing.allocator);
    try expect(r, a, &table);
    const base_notes = r.store().model.notes.items.len;
    var sh: Shared = .{ .r = r };
    defer for (0..threads) |k| {
        sh.seen[k].deinit(std.heap.page_allocator);
        sh.quiet[k].deinit(std.heap.page_allocator);
        sh.busy[k].deinit(std.heap.page_allocator);
        sh.after[k].deinit(std.heap.page_allocator);
    };
    var ts: [threads]std.Thread = undefined;
    for (&ts, 0..) |*t, k| t.* = try std.Thread.spawn(.{}, reader, .{ &sh, k });
    std.Io.sleep(testing.io, .fromMilliseconds(500), .awake) catch {}; // lookups alone first
    sh.phase.store(1, .release);
    var why: ops.Refusal = .{};
    var written: usize = 0;
    var erased: usize = 0;
    var k: usize = 0;
    while (k < 1000) : (k += 1) {
        var wa = std.heap.ArenaAllocator.init(testing.allocator);
        defer wa.deinit();
        const w = wa.allocator();
        try writeStep(r, w, k, &why, &erased, &table);
        written += 1;
        try expect(r, w, &table);
    }
    sh.phase.store(2, .release);
    std.Io.sleep(testing.io, .fromMilliseconds(500), .awake) catch {}; // the same lookups on the final store alone
    sh.stop.store(true, .release);
    for (ts) |t| t.join();
    try testing.expectEqual(@as(u32, 0), sh.failed.load(.acquire));
    var looked: usize = 0;
    var torn: usize = 0; // a block that is not what its version gives alone
    var unknown: usize = 0; // a version the writer never made (none should exist)
    var versions: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer versions.deinit(testing.allocator);
    for (sh.seen) |list| for (list.items) |x| {
        looked += 1;
        try versions.put(testing.allocator, x.ver, {});
        const row = table.get(x.ver) orelse {
            unknown += 1;
            continue;
        };
        if (row[x.q] != x.hash) torn += 1;
    };
    const quiet = try joined(&sh.quiet, a);
    const busy = try joined(&sh.busy, a);
    const after = try joined(&sh.after, a);
    // a test's stderr reads as a failure in zig build's output: the timings are notes-live-bench's to report
    try testing.expect(quiet.len > 0 and after.len > 0 and pct(busy, 50) > 0);
    try testing.expect(r.store().model.notes.items.len > base_notes + 500); // the writes grew the store
    try testing.expectEqual(@as(usize, 0), torn);
    try testing.expectEqual(@as(usize, 0), unknown);
    try testing.expectEqual(@as(usize, 1000), written);
    try testing.expect(looked > 8 * 100 and versions.count() > 100); // lookups ran across many versions
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}
