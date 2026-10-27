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
threadlocal var error_status: u16 = 500;

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
        error_status = 500;
        return h.arena.allocator();
    }
};

fn fail(e: anyerror) void {
    error_status = switch (e) {
        error.NotesFull, error.JournalFull, error.TooManyNotes, error.JournalTooLarge => 507,
        error.JournalPoisoned, error.ReadOnly, error.MissingRun => 503,
        error.InvalidTextSize, error.InvalidLookup, error.Template, error.Tokenizer, error.InvalidOperation => 400,
        error.NoNote => 404,
        error.SnapshotNote => 409,
        else => 500,
    };
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

pub export fn np_last_status() u16 {
    return error_status;
}

pub export fn np_open(snapshot: [*]const u8, snapshot_len: usize, path: [*]const u8, path_len: usize, vocabulary: u32, callback: Encode, context: ?*anyopaque) ?*Handle {
    error_text[0] = 0;
    error_status = 500;
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
    return np_operate(h, 0, 0, null, 0, bytes, len, told);
}

pub export fn np_operate(h: *Handle, operation: u8, target: u32, subject: ?[*]const u8, subject_len: usize, bytes: [*]const u8, len: usize, told: i64) ?[*:0]const u8 {
    const temp = h.reset();
    if (len == 0 or len > 16 << 10 or !std.unicode.utf8ValidateSlice(bytes[0..len])) {
        fail(error.InvalidTextSize);
        return null;
    }
    const op: @import("server/notes/write.zig").Op = switch (operation) {
        0 => .add,
        1 => .change,
        2 => .take_back,
        else => {
            fail(error.InvalidOperation);
            return null;
        },
    };
    if (subject_len > 200 or (subject == null and subject_len != 0)) {
        fail(error.InvalidOperation);
        return null;
    }
    const sub: ?[]const u8 = if (subject) |p| p[0..subject_len] else null;
    if (sub) |value| {
        if (value.len == 0 or !std.unicode.utf8ValidateSlice(value) or value[0] == ' ' or value[value.len - 1] == ' ') {
            fail(error.InvalidOperation);
            return null;
        }
        for (value) |byte| if (byte < 0x20 or byte == 0x7f) {
            fail(error.InvalidOperation);
            return null;
        };
    }
    var refusal: ops.Refusal = .{};
    var prepared = ops.prepare(h.store.?, temp, .{ .op = op, .target = target, .subject = sub, .text = bytes[0..len] }, told, &refusal) catch |e| {
        fail(e);
        if (e == error.Refused) error_status = refusal.status;
        return null;
    };
    defer prepared.deinit();
    const result = output(temp, prepared.reply) catch |e| {
        fail(e);
        return null;
    };
    prepared.commit(temp, &refusal) catch |e| {
        fail(e);
        if (e == error.Refused) error_status = refusal.status;
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

pub export fn np_get(h: *Handle, id: u32) ?[*:0]const u8 {
    const temp = h.reset();
    const m = h.store.?.model;
    const place = m.placeOf(id) orelse { fail(error.NoNote); return null; };
    const row = ops.view(m, temp, place) catch |e| { fail(e); return null; };
    return output(temp, row) catch |e| { fail(e); return null; };
}

pub export fn np_metadata(h: *Handle) ?[*:0]const u8 {
    const temp = h.reset();
    const Row = struct { id: u32, source: ?[]const u8, told: i64 };
    const notes = h.store.?.model.notes.items;
    const rows = temp.alloc(Row, notes.len) catch |e| { fail(e); return null; };
    for (notes, rows) |n, *row| row.* = .{ .id = n.id, .source = n.source(), .told = n.told };
    return output(temp, rows) catch |e| { fail(e); return null; };
}

// Attribute identity is durable in the reviewed journal. The adapter prefixes structured slots with "schema:";
// raw lesson text remains unmodified and can be parsed on reopen, without a non-atomic metadata sidecar.
pub export fn np_add_attribute(h: *Handle, attribute: [*]const u8, attribute_len: usize, subject: [*]const u8, subject_len: usize, bytes: [*]const u8, len: usize, told: i64) ?[*:0]const u8 {
    const temp = h.reset();
    if (attribute_len == 0 or attribute_len > 200 or subject_len == 0 or subject_len > 200 or len == 0 or len > 16 << 10) {
        fail(error.InvalidOperation);
        return null;
    }
    if (!std.unicode.utf8ValidateSlice(attribute[0..attribute_len]) or !std.unicode.utf8ValidateSlice(subject[0..subject_len]) or !std.unicode.utf8ValidateSlice(bytes[0..len])) {
        fail(error.InvalidOperation);
        return null;
    }
    const source = std.fmt.allocPrint(temp, "w{d}", .{h.store.?.log.next_source}) catch |e| {
        fail(e);
        return null;
    };
    var prepared = h.store.?.prepare(temp, .{
        .op = .add,
        .source = source,
        .subject = subject[0..subject_len],
        .attribute = attribute[0..attribute_len],
        .text = bytes[0..len],
        .told = told,
    }) catch |e| {
        fail(e);
        return null;
    };
    defer prepared.deinit();
    const out = prepared.out;
    const shown = if (out.note != 0) out.note else if (out.same_as.len > 0) out.same_as[0] else null;
    const reply: ops.Written = .{
        .verdict = out.verdict,
        .source = source,
        .told = told,
        .note = if (shown) |id| ops.view(h.store.?.model, temp, h.store.?.model.placeOf(id).?) catch |e| {
            fail(e);
            return null;
        } else null,
        .retired = out.retired,
        .same_as = out.same_as,
    };
    const result = output(temp, reply) catch |e| {
        fail(e);
        return null;
    };
    prepared.commit(temp) catch |e| {
        fail(e);
        return null;
    };
    return result;
}

pub export fn np_records(h: *Handle) ?[*:0]const u8 {
    return recordRows(h, false);
}

pub export fn np_structured_records(h: *Handle) ?[*:0]const u8 {
    return recordRows(h, true);
}

pub export fn np_last_told(h: *Handle) i64 {
    var last: i64 = 0;
    for (h.store.?.log.items.items) |entry| last = @max(last, entry.r.told);
    return last;
}

fn recordRows(h: *Handle, structured: bool) ?[*:0]const u8 {
    const temp = h.reset();
    const Record = struct { source: []const u8, text: []const u8, subject: []const u8, attribute: []const u8, told: i64 };
    const entries = h.store.?.log.items.items;
    var count: usize = 0;
    for (entries) |entry| {
        if (!structured or std.mem.startsWith(u8, entry.r.attribute, "schema:")) count += 1;
    }
    const rows = temp.alloc(Record, count) catch |e| {
        fail(e);
        return null;
    };
    var k: usize = 0;
    for (entries) |entry| {
        if (structured and !std.mem.startsWith(u8, entry.r.attribute, "schema:")) continue;
        rows[k] = .{
        .source = entry.r.source,
        .text = entry.r.text,
        .subject = entry.r.subject,
        .attribute = entry.r.attribute,
        .told = entry.r.told,
        };
        k += 1;
    }
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
    const result = output(temp, .{ .erased = id, .removed = prepared.removed, .complete = false, .remaining = "upstream" }) catch |e| {
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
