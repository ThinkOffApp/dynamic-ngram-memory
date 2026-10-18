//! Every write path under a failing allocator: each allocation fails in turn; the store is then as it was, no leak.
const std = @import("std");
const rig = @import("notes_rig_test.zig");
const ops = @import("notes_ops.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// The store as a reader sees it: every note's id, version, state and line, the journal's bytes, the writes kept.
fn digest(r: *rig.Rig, a: Allocator) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (r.store().model.notes.items) |n| try out.print(a, "{d}|{d}|{s}|{s}\n", .{ n.id, n.version, @tagName(n.state), n.line() });
    try out.print(a, "log {d} next {d} ver {d}\n", .{ r.store().log.items.items.len, r.store().log.next_source, r.store().current.idx.ver });
    const j = r.journal(a) catch "";
    try out.appendSlice(a, j);
    return out.items;
}

const Op = enum { add, add_entity, already_known, change_snapshot, change_written, take_back_snapshot, take_back_written, erase };

/// The writes an op needs made first, then the op itself; error.OutOfMemory when an allocation failed.
fn setup(r: *rig.Rig, a: Allocator, op: Op) !void {
    var why: ops.Refusal = .{};
    const s = r.store();
    switch (op) {
        .already_known, .change_written, .take_back_written => _ = try ops.perform(s, a, .{ .op = .add, .text = "My favourite colour is green." }, &why),
        .erase => {
            _ = try ops.perform(s, a, .{ .op = .add, .text = "My favourite colour is green." }, &why);
            _ = try ops.perform(s, a, .{ .op = .add, .text = "my favourite colour is green" }, &why);
            _ = try ops.perform(s, a, .{ .op = .add, .text = "I like long walks." }, &why);
            _ = try ops.perform(s, a, .{ .op = .take_back, .text = "Green is not my favourite colour anymore.", .target = 30 }, &why);
        },
        else => {},
    }
}

fn run(r: *rig.Rig, a: Allocator, op: Op) !void {
    var why: ops.Refusal = .{};
    const s = r.store();
    switch (op) {
        .add, .already_known => _ = try ops.perform(s, a, .{ .op = .add, .text = "My favourite colour is green." }, &why),
        .add_entity => _ = try ops.perform(s, a, .{ .op = .add, .text = "My friend Ana lives in Oslo.", .subject = "my friend Ana" }, &why),
        .change_snapshot => _ = try ops.perform(s, a, .{ .op = .change, .text = "My favourite colour is red now.", .target = 26 }, &why),
        .change_written => _ = try ops.perform(s, a, .{ .op = .change, .text = "My favourite colour is red now.", .target = 30 }, &why),
        .take_back_snapshot => _ = try ops.perform(s, a, .{ .op = .take_back, .text = "I take that back.", .target = 28 }, &why),
        .take_back_written => _ = try ops.perform(s, a, .{ .op = .take_back, .text = "I take that back.", .target = 30 }, &why),
        .erase => _ = try ops.erase(s, a, 30, &why),
    }
}

test "every write path, with each of its allocations failing in turn, changes nothing and leaks nothing" {
    for (std.enums.values(Op)) |op| {
        var points: usize = 0;
        var failed: usize = 0;
        while (true) : (points += 1) {
            var failing = std.testing.FailingAllocator.init(testing.allocator, .{});
            var arena = std.heap.ArenaAllocator.init(testing.allocator); // the request's memory, freed with it
            defer arena.deinit();
            const a = arena.allocator();
            const r = try rig.Rig.init(failing.allocator());
            try setup(r, a, op);
            const before = try digest(r, a);
            failing.fail_index = failing.alloc_index + points;
            const result = run(r, a, op);
            failing.fail_index = std.math.maxInt(usize);
            const induced = failing.has_induced_failure;
            if (result) |_| {} else |e| {
                try testing.expectEqual(error.OutOfMemory, e);
                failed += 1;
                errdefer std.debug.print("{s}: allocation {d} failed and the store changed\n", .{ @tagName(op), points });
                try testing.expectEqualStrings(before, try digest(r, a));
                try run(r, a, op); // and the same write then works
            }
            r.deinit();
            try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            if (!induced) break;
        }
        errdefer std.debug.print("{s}: {d} points\n", .{ @tagName(op), points });
        try testing.expect(failed >= 5); // the note's strings, its line and costs, the version's index, the log entry
    }
}
