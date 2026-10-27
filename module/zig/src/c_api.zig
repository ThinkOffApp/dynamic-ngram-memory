//! Caller serializes calls per handle and keeps the tokenizer callback/context alive until close.
//! Returned JSON is handle-owned until its next call. Failures return null; last_error is thread-local.
const std = @import("std");
const Storage = @import("server/notes/storage.zig").Storage;
const Text = @import("server/model_text.zig");
const ops = @import("server/notes/operations.zig");
const erase = @import("standalone_erase.zig");
const a = std.heap.c_allocator;
pub const Encode = *const fn (?*anyopaque, [*]const u8, usize, ?[*]u32, usize) callconv(.c) isize;
threadlocal var error_text: [256:0]u8 = @splat(0);

const Handle = struct {
    store: ?*Storage = null,
    arena: std.heap.ArenaAllocator,
    callback: Encode,
    context: ?*anyopaque,
    vocabulary: u32,
    io_backend: std.Io.Threaded = .init_single_threaded,
    fn reset(h: *Handle) std.mem.Allocator {
        _ = h.arena.reset(.free_all);
        error_text[0] = 0;
        return h.arena.allocator();
    }
};

fn fail(e: anyerror) void {
    const name = @errorName(e);
    const n = @min(name.len, error_text.len - 1);
    @memcpy(error_text[0..n], name[0..n]);
    error_text[n] = 0;
}

fn encode(ctx: *anyopaque, allocator: std.mem.Allocator, bytes: []const u8, _: bool) Text.Error![]u32 {
    const h: *Handle = @ptrCast(@alignCast(ctx));
    const count = h.callback(h.context, bytes.ptr, bytes.len, null, 0);
    if (count < 0 or count > 1 << 24) return error.Template;
    const ids = try allocator.alloc(u32, @intCast(count));
    errdefer allocator.free(ids);
    if (h.callback(h.context, bytes.ptr, bytes.len, ids.ptr, ids.len) != count) return error.Template;
    for (ids) |id| if (id >= h.vocabulary) return error.Template;
    return ids;
}

pub export fn np_last_error() [*:0]const u8 {
    return &error_text;
}

pub export fn np_open(snapshot: [*]const u8, snapshot_len: usize, path: [*]const u8, path_len: usize, vocabulary: u32, callback: Encode, context: ?*anyopaque) ?*Handle {
    error_text[0] = 0;
    const h = a.create(Handle) catch |e| {
        fail(e);
        return null;
    };
    h.* = .{ .arena = .init(a), .callback = callback, .context = context, .vocabulary = vocabulary };
    var problem: []const u8 = "";
    h.store = Storage.load(a, snapshot[0..snapshot_len], .{
        .ctx = h,
        .encode_fn = encode,
        .vocabulary_size = vocabulary,
    }, path[0..path_len], &problem) catch |e| {
        fail(e);
        h.arena.deinit();
        a.destroy(h);
        return null;
    };
    h.store.?.beginRun(h.arena.allocator(), .{}) catch |e| {
        fail(e);
        h.store.?.deinit();
        h.arena.deinit();
        a.destroy(h);
        return null;
    };
    return h;
}

pub export fn np_close(h: ?*Handle) void {
    const handle = h orelse return;
    handle.store.?.deinit();
    handle.arena.deinit();
    a.destroy(handle);
}

fn output(allocator: std.mem.Allocator, value: anytype) ![*:0]const u8 {
    const bytes = try std.json.Stringify.valueAlloc(allocator, value, .{});
    return (try allocator.dupeSentinel(u8, bytes, 0)).ptr;
}

pub export fn np_write(h: *Handle, bytes: [*]const u8, len: usize, told: i64) ?[*:0]const u8 {
    const temp = h.reset();
    if (len == 0 or len > 16 << 10) {
        fail(error.InvalidTextSize);
        return null;
    }
    var refusal: ops.Refusal = .{};
    var prepared = ops.prepare(h.store.?, temp, .{ .op = .add, .text = bytes[0..len] }, told, &refusal) catch |e| {
        fail(e);
        return null;
    };
    defer prepared.deinit();
    const result = output(temp, prepared.reply) catch |e| {
        fail(e);
        return null;
    };
    prepared.commit(temp, &refusal) catch |e| {
        fail(e);
        return null;
    };
    return result;
}

pub export fn np_list(h: *Handle) ?[*:0]const u8 {
    const temp = h.reset();
    const m = h.store.?.model;
    const rows = temp.alloc(ops.View, m.notes.items.len) catch |e| {
        fail(e);
        return null;
    };
    for (rows, 0..) |*row, k| row.* = ops.view(m, temp, @intCast(k)) catch |e| {
        fail(e);
        return null;
    };
    return output(temp, rows) catch |e| {
        fail(e);
        return null;
    };
}

pub export fn np_erase(h: *Handle, id: u32) ?[*:0]const u8 {
    const temp = h.reset();
    var prepared = erase.prepare(h.store.?, temp, id) catch |e| {
        fail(e);
        return null;
    };
    defer prepared.deinit();
    const result = output(temp, .{ .erased = id, .complete = false, .remaining = "upstream" }) catch |e| {
        fail(e);
        return null;
    };
    prepared.commit(temp) catch |e| {
        fail(e);
        return null;
    };
    prepared.finish(temp) catch |e| {
        fail(e);
        return null;
    };
    return result;
}

pub export fn np_lookup(h: *Handle, question: [*]const u8, len: usize, cap: u32, past: bool, excluded: ?[*]const u32, excluded_len: usize) ?[*:0]const u8 {
    const temp = h.reset();
    if (len > 1 << 20 or excluded_len > 100_000 or (excluded == null and excluded_len != 0)) {
        fail(error.InvalidLookup);
        return null;
    }
    const ids = encode(h, temp, question[0..len], false) catch |e| {
        fail(e);
        return null;
    };
    const omit = if (excluded) |p| p[0..excluded_len] else &.{};
    const result = @import("server/notes/retrieve.zig").retrieve(h.store.?, h.io_backend.io(), temp, h.store.?.text, ids, past, cap, omit, .{}) catch |e| {
        fail(e);
        return null;
    };
    return output(temp, result) catch |e| {
        fail(e);
        return null;
    };
}
