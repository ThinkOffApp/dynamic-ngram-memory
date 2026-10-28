//! Adapter-only v4 wire codec. No journal writes or replay side effects.
const std = @import("std");
const json = @import("json");
const types = @import("intake_record_v4.zig");
const Allocator = std.mem.Allocator;
pub const Error = Allocator.Error || error{BadRecord};

fn quote(a: Allocator, out: *std.ArrayList(u8), text: []const u8) Allocator.Error!void {
    const encoded = try json.quote(a, text, .{ .ascii = false });
    defer a.free(encoded);
    try out.appendSlice(a, encoded);
}

fn list(a: Allocator, out: *std.ArrayList(u8), items: []const []const u8) Allocator.Error!void {
    try out.append(a, '[');
    for (items, 0..) |item, i| {
        if (i != 0) try out.append(a, ',');
        try quote(a, out, item);
    }
    try out.append(a, ']');
}

/// Contextual source/owner validation must precede encoding in the writer.
pub fn encode(a: Allocator, r: types.Record) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.print(a, "{{\"format\":\"{s}\",\"type\":\"{s}\",\"source\":", .{ types.format, @tagName(r.effect) });
    try quote(a, &out, r.source);
    try out.print(a, ",\"told\":{d},\"verdict\":\"{s}\",\"subject\":", .{ r.told, @tagName(r.verdict) });
    try quote(a, &out, r.subject);
    try out.appendSlice(a, ",\"attribute\":");
    try quote(a, &out, r.attribute);
    try out.appendSlice(a, ",\"text\":");
    try quote(a, &out, r.text);
    try out.appendSlice(a, ",\"depends\":");
    try list(a, &out, r.depends);
    switch (r.effect) {
        .learned => |learned| {
            try out.print(a, ",\"id\":{d},\"retire\":", .{learned.id});
            try list(a, &out, learned.retire);
        },
        .noop => {},
    }
    try out.appendSlice(a, "}\n");
    return out.toOwnedSlice(a);
}

fn sources(a: Allocator, value: json.Value) Error![]const []const u8 {
    const items = value.items() orelse return error.BadRecord;
    if (items.len > types.max_references) return error.BadRecord;
    const result = try a.alloc([]const u8, items.len);
    errdefer a.free(result);
    for (items, result) |item, *slot| {
        slot.* = item.str() orelse return error.BadRecord;
        _ = types.Source.parse(slot.*) catch return error.BadRecord;
    }
    return result;
}

/// All strings borrow `v`; arrays belong to `a`. Use a replay arena so the
/// complete parsed record remains alive until copied into prepared state.
/// This is syntax/shape validation only; validateWithAllocator checks references.
pub fn decode(a: Allocator, v: json.Value) Error!types.Record {
    if (!std.mem.eql(u8, v.strField("format") orelse return error.BadRecord, types.format)) return error.BadRecord;
    const tag = v.strField("type") orelse return error.BadRecord;
    const learned = std.mem.eql(u8, tag, "learned");
    if (!learned and !std.mem.eql(u8, tag, "noop")) return error.BadRecord;
    if (v != .object or v.object.count() != (if (learned) @as(usize, 11) else 9)) return error.BadRecord;
    const source = v.strField("source") orelse return error.BadRecord;
    const parsed_source = types.Source.parse(source) catch return error.BadRecord;
    if (parsed_source.kind != .write) return error.BadRecord;
    const verdict = std.meta.stringToEnum(types.Verdict, v.strField("verdict") orelse return error.BadRecord) orelse return error.BadRecord;
    const subject = v.strField("subject") orelse return error.BadRecord;
    const attribute = v.strField("attribute") orelse return error.BadRecord;
    const text = v.strField("text") orelse return error.BadRecord;
    if (subject.len == 0 or attribute.len == 0 or text.len == 0) return error.BadRecord;
    const depends = try sources(a, v.get("depends") orelse return error.BadRecord);
    errdefer a.free(depends);
    var effect: types.Effect = .noop;
    if (learned) {
        if (verdict != .NEW and verdict != .REPLACES) return error.BadRecord;
        const id = std.math.cast(u32, (v.get("id") orelse return error.BadRecord).int64() orelse return error.BadRecord) orelse return error.BadRecord;
        if (id == 0) return error.BadRecord;
        const retire = try sources(a, v.get("retire") orelse return error.BadRecord);
        errdefer a.free(retire);
        if ((verdict == .REPLACES) != (retire.len != 0)) return error.BadRecord;
        effect = .{ .learned = .{ .id = id, .retire = retire } };
    } else if ((verdict != .ALREADY_KNOWN and verdict != .IMPLIED) or (verdict == .ALREADY_KNOWN and depends.len == 0)) return error.BadRecord;
    errdefer if (effect == .learned) a.free(effect.learned.retire);
    return .{
        .source = source,
        .told = (v.get("told") orelse return error.BadRecord).int64() orelse return error.BadRecord,
        .subject = subject,
        .attribute = attribute,
        .text = text,
        .verdict = verdict,
        .depends = depends,
        .effect = effect,
    };
}

test "v4 learned and raw no-op round trip with explicit source lists" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]types.Record{
        .{ .source = "w3", .told = 3, .subject = "alice", .attribute = "likes", .text = "I like \"pears\".\nNow!", .verdict = .REPLACES, .effect = .{ .learned = .{ .id = 3, .retire = &.{ "w1", "w2" } } } },
        .{ .source = "w4", .told = 4, .subject = "alice", .attribute = "likes", .text = "Still pears — kyllä!", .verdict = .ALREADY_KNOWN, .depends = &.{"w3"}, .effect = .noop },
    }) |record| {
        const encoded = try encode(a, record);
        const parsed = (try json.parse(a, encoded)).ok;
        const decoded = try decode(a, parsed);
        try std.testing.expectEqualStrings(record.text, decoded.text);
        try std.testing.expectEqualStrings(encoded, try encode(a, decoded));
        // The existing v3 decoder cannot silently interpret either v4 record.
        try std.testing.expect(@import("server/notes/journal.zig").decode(parsed) == null);
    }
}

test "v4 codec refuses unknown version, incomplete shape, non-array and invalid id" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const record: types.Record = .{ .source = "w3", .told = 3, .subject = "alice", .attribute = "likes", .text = "pears", .verdict = .REPLACES, .effect = .{ .learned = .{ .id = 3, .retire = &.{ "w1", "w2" } } } };
    const encoded = try encode(a, record);
    for ([_][]const u8{ "format", "retire", "id", "told" }) |field| {
        const parsed = (try json.parse(a, encoded)).ok;
        try parsed.object.put(a, field, .null);
        try std.testing.expectError(error.BadRecord, decode(a, parsed));
    }
    const parsed = (try json.parse(a, encoded)).ok;
    _ = parsed.object.swapRemove("retire");
    try std.testing.expectError(error.BadRecord, decode(a, parsed));
}
