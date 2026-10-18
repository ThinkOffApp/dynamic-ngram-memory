//! The decode-time note lookup's request fields and its report: the reply's notes.decode field and the log line.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const notes_route = @import("notes_route.zig");
const notes_decode = @import("notes_decode.zig");
const Live = notes_decode.Live;
const Injection = notes_decode.Injection;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Where new notes go: (a) into the first block, (b) a block before the assistant turn, (c) one before the question.
pub const Place = enum { first, turn, user };

/// A request's notes_decode (default on), notes_decode_place, notes_decode_names and notes_decode_dry (checks only).
pub const Ask = struct { on: bool = true, place: Place = default_place, dry: bool = false, names: bool = default_names };

/// The defaults, measured (NOTES-DECODE.md): a hit's notes before the question, a name the reply writes a check point.
pub const default_place: Place = .user;
pub const default_names = true;
/// The first lookup's defaults: the subject+value hop on (the snapshot's own hop with false), the block first.
pub const default_value_hop = true;
pub const default_before_user = false;

/// A request's notes_value_hop and notes_position ("first" or "user"): how the first lookup is made and placed.
pub fn shapeAsked(cx: *Cx, body: Value) errors.Refused!notes_route.Shape {
    const on = (try notes_route.asked(cx, body)) != null;
    const hop = body.field("notes_value_hop");
    const position = body.field("notes_position");
    const subjects = body.field("notes_subjects");
    if (!on) {
        if (hop != null or position != null or subjects != null) return cx.refuse("notes_value_hop, notes_position and notes_subjects need \"notes\": true");
        return .{};
    }
    var value_hop = default_value_hop;
    if (hop) |v| {
        if (v != .bool) return cx.refuse("notes_value_hop must be a boolean");
        value_hop = v.bool;
    }
    var before = default_before_user;
    if (position) |v| {
        const s = v.str() orelse "";
        before = if (std.mem.eql(u8, s, "user")) true else if (std.mem.eql(u8, s, "first")) false else return cx.refuse("notes_position must be \"first\" or \"user\"");
    }
    var resolve_subjects = false;
    if (subjects) |v| {
        if (v != .bool) return cx.refuse("notes_subjects must be a boolean");
        resolve_subjects = v.bool;
    }
    return .{ .hop = if (value_hop) .subject_value else null, .before_user = before, .resolve_subjects = resolve_subjects };
}

/// The decode fields of a request; `notes_on`: it asked for notes, which they need.
pub fn asked(cx: *Cx, body: Value, notes_on: bool) errors.Refused!Ask {
    var ask: Ask = .{};
    const on = body.field("notes_decode");
    const place = body.field("notes_decode_place");
    const dry = body.field("notes_decode_dry");
    const names = body.field("notes_decode_names");
    if (!notes_on) {
        if (on != null or place != null or dry != null or names != null) return cx.refuse("notes_decode, notes_decode_place, notes_decode_names and notes_decode_dry need \"notes\": true");
        return ask;
    }
    if (names) |v| {
        if (v != .bool) return cx.refuse("notes_decode_names must be a boolean");
        ask.names = v.bool;
    }
    if (on) |v| {
        if (v != .bool) return cx.refuse("notes_decode must be a boolean");
        ask.on = v.bool;
    }
    if (dry) |v| {
        if (v != .bool) return cx.refuse("notes_decode_dry must be a boolean");
        ask.dry = v.bool;
    }
    if (place) |v| {
        const s = v.str() orelse "";
        ask.place = std.meta.stringToEnum(Place, s) orelse return cx.refuse("notes_decode_place must be \"first\", \"turn\" or \"user\"");
    }
    return ask;
}

/// Whether a reply piece (one token's text) ends a sentence: a full stop, ! or ? before closing marks, or a line break.
pub fn sentenceEnd(piece: []const u8) bool {
    if (std.mem.indexOfScalar(u8, piece, '\n') != null) return true;
    var end = piece.len;
    while (end > 0 and std.mem.indexOfScalar(u8, "\"')]*_ \t", piece[end - 1]) != null) end -= 1;
    const t = piece[0..end];
    if (t.len > 0 and std.mem.indexOfScalar(u8, ".!?", t[t.len - 1]) != null) return true;
    for ([_][]const u8{ "\u{3002}", "\u{ff01}", "\u{ff1f}", "\u{2026}" }) |m| if (std.mem.endsWith(u8, t, m)) return true;
    return false;
}

/// The reply's notes.decode field.
pub fn value(l: *const Live, a: Allocator) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "on", .{ .bool = l.on });
    if (!l.on) {
        if (l.stopped) |s| try o.put(a, "stopped", .{ .string = s });
        return .{ .object = o };
    }
    try o.put(a, "place", .{ .string = @tagName(l.place) });
    try o.put(a, "names", .{ .bool = l.names });
    try o.put(a, "value_hop", .{ .bool = l.shape.hop != null });
    try o.put(a, "position", .{ .string = if (l.shape.before_user) "user" else "first" });
    if (l.dry) try o.put(a, "dry", .{ .bool = true });
    try o.put(a, "notes_decode.min_share", .{ .float = notes_decode.min_share });
    try o.put(a, "checks", try json.intValue(a, l.n_checks));
    try o.put(a, "capped", try json.intValue(a, l.capped));
    try o.put(a, "check_us", .{ .float = micro(l.check_ns) });
    try o.put(a, "best_share", .{ .float = l.best.share });
    try o.put(a, "best_id", if (l.best.best > 0) try json.intValue(a, l.best.best) else .null);
    const list = try a.alloc(Value, l.injections.items.len);
    for (l.injections.items, list) |*inj, *slot| slot.* = try injectionValue(a, inj);
    try o.put(a, "injected", .{ .array = list });
    try o.put(a, "ids", try intsValue(a, l.held.items));
    try o.put(a, "stopped", if (l.stopped) |s| .{ .string = s } else .null);
    if (!l.echo) return .{ .object = o };
    const checks = try a.alloc(Value, l.checks.items.len);
    for (l.checks.items, checks) |c, *slot| {
        const x = try json.newObject(a);
        try x.put(a, "at", try json.intValue(a, c.at));
        try x.put(a, "best_id", if (c.best > 0) try json.intValue(a, c.best) else .null);
        try x.put(a, "share", .{ .float = c.share });
        try x.put(a, "ids", try intsValue(a, c.ids));
        const ln = try a.alloc(Value, c.named.len);
        for (c.named, ln) |b, *v| v.* = .{ .bool = b };
        try x.put(a, "named", .{ .array = ln });
        const sh = try a.alloc(Value, c.shares.len);
        for (c.shares, sh) |f, *v| v.* = .{ .float = f };
        try x.put(a, "shares", .{ .array = sh });
        slot.* = .{ .object = x };
    }
    try o.put(a, "check_list", .{ .array = checks });
    try o.put(a, "block", .{ .string = if (l.place == .first) l.block else l.late_block });
    return .{ .object = o };
}

fn micro(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

fn intsValue(a: Allocator, xs: []const u32) Allocator.Error!Value {
    const out = try a.alloc(Value, xs.len);
    for (xs, out) |x, *slot| slot.* = try json.intValue(a, x);
    return .{ .array = out };
}

fn injectionValue(a: Allocator, inj: *const Injection) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "ids", try intsValue(a, inj.ids));
    const shares = try a.alloc(Value, inj.shares.len);
    for (inj.shares, shares) |s, *slot| slot.* = .{ .float = s };
    try o.put(a, "shares", .{ .array = shares });
    try o.put(a, "at_token", try json.intValue(a, inj.at));
    try o.put(a, "place", .{ .string = @tagName(inj.place) });
    try o.put(a, "block_tokens", try json.intValue(a, inj.block_tokens));
    try o.put(a, "lookup_us", .{ .float = micro(inj.lookup_ns) });
    try o.put(a, "prompt_tokens", try json.intValue(a, inj.prompt));
    try o.put(a, "cached_tokens", if (inj.cached) |c| try json.intValue(a, c) else .null);
    try o.put(a, "pause_ms", if (inj.pause_ns) |p| .{ .float = micro(p) / 1e3 } else .null);
    try o.put(a, "build_us", .{ .float = micro(inj.build_ns) });
    try o.put(a, "drain_ms", .{ .float = micro(inj.drain_ns) / 1e3 });
    try o.put(a, "wait_ms", .{ .float = micro(inj.wait_ns) / 1e3 });
    return .{ .object = o };
}

/// The server's line for one injection: which notes, after which reply token, and the pause it made.
pub fn logLine(id: []const u8, inj: *const Injection) void {
    var buf: [200]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (inj.ids, 0..) |n, k| w.print("{s}{d}", .{ if (k > 0) "," else "", n }) catch break;
    const cached = inj.cached orelse 0;
    log.line("notes decode {s}: [{s}] put in after reply token {d} ({s}), pause {d:.1} ms (lookup {d:.1} us, render {d:.1} us, cancel {d:.1} ms, prefill and first token {d:.1} ms); prompt {d} tokens, {d} cached, {d} prefilled; notes {d} tokens", .{
        id,                        w.buffered(),             inj.at,     @tagName(inj.place), micro(inj.pause_ns orelse 0) / 1e3, micro(inj.lookup_ns), micro(inj.build_ns),
        micro(inj.drain_ns) / 1e3, micro(inj.wait_ns) / 1e3, inj.prompt, cached,              inj.prompt -| cached,               inj.block_tokens,
    });
}
