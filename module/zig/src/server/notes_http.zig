//! /v1/notes: the note store's writes and listing over HTTP, behind the /v1 key check like every /v1 route.
const std = @import("std");
const json = @import("json.zig");
const http_body = @import("http_body.zig");
const ops = @import("notes_ops.zig");
const write = @import("notes/write.zig");
const index = @import("notes/index.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The most a notes request body may hold, and a note's text, and a subject.
pub const body_limit: usize = 1 << 20;
pub const text_limit: usize = 16 << 10;
pub const subject_limit: usize = 200;

/// A path this file answers: /v1/notes, or /v1/notes/ID (null id), or not ours (null).
pub const Route = struct { id: ?u32, bad_id: bool = false };

pub fn route(path: []const u8) ?Route {
    if (std.mem.eql(u8, path, "/v1/notes")) return .{ .id = null };
    if (!std.mem.startsWith(u8, path, "/v1/notes/")) return null;
    const tail = path["/v1/notes/".len..];
    for (tail) |c| if (!std.ascii.isDigit(c)) return .{ .id = null, .bad_id = true };
    const id = std.fmt.parseInt(u32, tail, 10) catch return .{ .id = null, .bad_id = true };
    return .{ .id = id };
}

/// One request to /v1/notes or /v1/notes/ID, by its method.
pub fn handle(srv: *Server, conn: *Conn, a: Allocator, r: Route) void {
    reply(srv, conn, a, r) catch |e| switch (e) {
        error.OutOfMemory => fail(conn, a, 500, "out of memory"),
    };
}

fn reply(srv: *Server, conn: *Conn, a: Allocator, r: Route) Allocator.Error!void {
    const m = conn.method;
    const get = std.mem.eql(u8, m, "GET");
    const body = if (get) "" else switch (try http_body.read(conn, a, body_limit)) {
        .ok => |b| b,
        .refused => |why| return fail(conn, a, if (std.mem.startsWith(u8, why, "request body exceeds")) 413 else 400, why),
    };
    const store = srv.notes orelse return fail(conn, a, 404, "notes are off: this server was started without --notes");
    if (r.bad_id) return fail(conn, a, 404, "a note id is a whole number: /v1/notes/ID");
    var why: ops.Refusal = .{};
    if (get) {
        if (r.id) |id| {
            const all = try ops.list(store, a, true);
            for (all) |v| if (v.id == id) return send(conn, a, 200, try viewValue(a, v));
            return fail(conn, a, 404, "no note has this id");
        }
        const history = flag(conn.path, "history");
        const list = try ops.list(store, a, history);
        const o = try json.newObject(a);
        try o.put(a, "object", .{ .string = "list" });
        const data = try a.alloc(Value, list.len);
        for (list, data) |v, *slot| slot.* = try viewValue(a, v);
        try o.put(a, "data", .{ .array = data });
        try o.put(a, "history", .{ .bool = history });
        return send(conn, a, 200, .{ .object = o });
    }
    const post = std.mem.eql(u8, m, "POST") or std.mem.eql(u8, m, "PUT");
    const delete = std.mem.eql(u8, m, "DELETE");
    if (!post and !delete or post and std.mem.eql(u8, m, "PUT") and r.id == null) return fail(conn, a, 405, "use GET or POST on /v1/notes, and GET, POST, PUT or DELETE on /v1/notes/ID");
    if (delete and r.id == null) return fail(conn, a, 405, "DELETE takes a note: /v1/notes/ID");
    const v = if (body.len == 0 and delete) Value.null else switch (try json.parse(a, body)) {
        .ok => |x| x,
        .err => |e| return fail(conn, a, 400, try std.fmt.allocPrint(a, "the request body is not JSON: {s}", .{e})),
    };
    if (v != .null and v != .object) return fail(conn, a, 400, "the request body must be a JSON object");
    if (delete and flag(conn.path, "erase")) {
        if (v != .null and v.object.count() > 0) return fail(conn, a, 400, "an erase takes no body");
        const done = ops.erase(store, a, r.id.?, &why) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.Refused => fail(conn, a, why.status, why.message),
        };
        return send(conn, a, 200, try erasedValue(srv, a, r.id.?, done));
    }
    const allowed: []const []const u8 = if (delete) &.{"text"} else if (r.id != null) &.{"text"} else &.{ "text", "subject", "replaces" };
    if (v == .object) for (v.object.keys()) |k| {
        for (allowed) |name| {
            if (std.mem.eql(u8, k, name)) break;
        } else return fail(conn, a, 400, try std.fmt.allocPrint(a, "unknown field {s}; the fields here are {s}", .{ k, if (allowed.len == 1) "text" else "text, subject and replaces" }));
    };
    var ask: ops.Ask = .{ .op = if (delete) .take_back else if (r.id != null) .change else .add, .text = write.default_takeback };
    if (r.id) |id| ask.target = id;
    if (v.field("text")) |t| {
        ask.text = t.str() orelse return fail(conn, a, 400, "text must be a string");
    } else if (!delete) return fail(conn, a, 400, "text is required: the note's sentence, as the user said it");
    if (std.mem.trim(u8, ask.text, " \t\r\n").len == 0) return fail(conn, a, 400, "text must not be empty");
    if (ask.text.len > text_limit) return fail(conn, a, 413, "text is over 16 KiB: a note is one thing the user said");
    if (!std.unicode.utf8ValidateSlice(ask.text)) return fail(conn, a, 400, "text must be valid Unicode (no lone surrogates)");
    if (v.field("subject")) |sub| {
        const name = sub.str() orelse return fail(conn, a, 400, "subject must be a string");
        if (!goodSubject(name)) return fail(conn, a, 400, "subject must be 1 to 200 bytes of text with no control characters and no space at either end");
        ask.subject = name;
    }
    if (v.field("replaces")) |x| {
        const n = if (x == .int) x.int64() else null;
        ask.target = std.math.cast(u32, n orelse -1) orelse return fail(conn, a, 400, "replaces must be a note id");
        ask.op = .change;
    }
    const done = ops.perform(store, a, ask, &why) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Refused => fail(conn, a, why.status, why.message),
    };
    return send(conn, a, if (done.verdict.learn()) 201 else 200, try writtenValue(a, ask.op, done));
}

fn goodSubject(s: []const u8) bool {
    if (s.len == 0 or s.len > subject_limit or !std.unicode.utf8ValidateSlice(s)) return false;
    for (s) |c| if (c < 0x20 or c == 0x7f) return false;
    return s[0] != ' ' and s[s.len - 1] != ' ';
}

/// Whether the query string sets `name` to 1 or true.
fn flag(path: []const u8, name: []const u8) bool {
    const q = std.mem.indexOfScalar(u8, path, '?') orelse return false;
    var it = std.mem.splitScalar(u8, path[q + 1 ..], '&');
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse kv.len;
        if (!std.mem.eql(u8, kv[0..eq], name)) continue;
        const val = if (eq < kv.len) kv[eq + 1 ..] else "1";
        return std.mem.eql(u8, val, "1") or std.ascii.eqlIgnoreCase(val, "true");
    }
    return false;
}

fn stateName(s: index.State) []const u8 {
    return switch (s) {
        .current => "current",
        .superseded => "superseded",
        .tombstoned => "tombstoned",
    };
}

fn ids(a: Allocator, xs: []const u32) Allocator.Error!Value {
    const out = try a.alloc(Value, xs.len);
    for (xs, out) |x, *slot| slot.* = try json.intValue(a, x);
    return .{ .array = out };
}

pub fn viewValue(a: Allocator, v: ops.View) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "id", try json.intValue(a, v.id));
    try o.put(a, "version", try json.intValue(a, v.version));
    try o.put(a, "state", .{ .string = stateName(v.state) });
    try o.put(a, "kind", .{ .string = if (v.kind == .fact) "fact" else "take_back" });
    try o.put(a, "subject", .{ .string = v.subject });
    try o.put(a, "text", .{ .string = v.text });
    try o.put(a, "line", .{ .string = v.line });
    try o.put(a, "source", if (v.source) |s| .{ .string = s } else .null);
    try o.put(a, "told", try json.intValue(a, v.told));
    try o.put(a, "superseded_by", if (v.state == .superseded) try json.intValue(a, v.by) else .null);
    try o.put(a, "tombstoned_by", if (v.state == .tombstoned) try json.intValue(a, v.by) else .null);
    return .{ .object = o };
}

fn writtenValue(a: Allocator, op: write.Op, w: ops.Written) Allocator.Error!Value {
    const o = try json.newObject(a);
    if (w.note) |n| {
        try o.put(a, "id", try json.intValue(a, n.id));
        try o.put(a, "version", try json.intValue(a, n.version));
        try o.put(a, "state", .{ .string = stateName(n.state) });
    } else {
        try o.put(a, "id", .null);
        try o.put(a, "version", .null);
        try o.put(a, "state", .null);
    }
    try o.put(a, "verdict", .{ .string = @tagName(w.verdict) });
    try o.put(a, "written", .{ .bool = w.verdict.learn() });
    try o.put(a, "source", .{ .string = w.source });
    try o.put(a, "told", try json.intValue(a, w.told));
    try o.put(a, if (op == .take_back) "tombstoned" else "superseded", try ids(a, w.retired));
    try o.put(a, "same_as", try ids(a, w.same_as));
    try o.put(a, "note", if (w.note) |n| try viewValue(a, n) else .null);
    return .{ .object = o };
}

fn erasedValue(srv: *Server, a: Allocator, id: u32, e: ops.Erased) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "erased", try json.intValue(a, id));
    const removed = try a.alloc(Value, e.removed.len);
    for (e.removed, removed) |s, *slot| slot.* = .{ .string = s };
    try o.put(a, "removed", .{ .array = removed });
    const moved = try a.alloc(Value, e.moved.len);
    for (e.moved, moved) |p, *slot| {
        const x = try json.newObject(a);
        try x.put(a, "from", try json.intValue(a, p[0]));
        try x.put(a, "to", try json.intValue(a, p[1]));
        slot.* = .{ .object = x };
    }
    try o.put(a, "renumbered", .{ .array = moved });
    // the store's own structures and the journal hold it no more; the engine's prompt cache is out of its reach
    const cached = srv.info.prompt_cache;
    try o.put(a, "complete", .{ .bool = !cached });
    var remaining: []Value = &.{};
    if (cached) {
        remaining = try a.alloc(Value, 1);
        remaining[0] = .{ .string = "the engine's prompt cache may still hold prompts that carried this note's block; serve with --prompt-cache-gib 0 for an erase that reaches every cache" };
    }
    try o.put(a, "remaining", .{ .array = remaining });
    return .{ .object = o };
}

fn send(conn: *Conn, a: Allocator, code: u16, v: Value) void {
    const text = json.stringify(a, v, .{}) catch return fail(conn, a, 500, "out of memory");
    conn.sendJson(code, text);
}

fn fail(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const kind = switch (code) {
        404 => "not_found_error",
        409 => "conflict_error",
        413 => "request_too_large",
        503 => "unavailable_error",
        500 => "server_error",
        else => "invalid_request_error",
    };
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}, \"type\": \"{s}\", \"code\": {d}}}}}", .{ json.quote(a, message, .{}) catch "\"\"", kind, code }) catch return;
    conn.sendJson(code, body);
}
