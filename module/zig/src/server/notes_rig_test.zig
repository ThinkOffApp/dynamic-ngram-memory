//! The live-notes tests' rig: the byte-tokenized teach store in a temporary folder, a server, and HTTP requests.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const routes = @import("routes.zig");
const notes_route = @import("notes_route.zig");
const notes_store = @import("notes_store.zig");
const line_mod = @import("notes/line.zig");
const Bytes = @import("notes_test.zig").Bytes;
const Conn = @import("http_conn.zig").Conn;
const testing = std.testing;

pub const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");
pub const writes_json = @embedFile("notes/testdata/l6_bytes_writes.json");
pub const cases_json = @embedFile("notes/testdata/l6_bytes_cases.json");

/// Answers "ok" to every prompt.
pub const Echo = struct {
    pub fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(_: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{'o'} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The day the fixture's writes are told on (2026-10-16).
pub const today = line_mod.ordinalOf(2026, 10, 16);

/// An index kind and side buffer a rig is built with.
pub const Kind = struct { kind: notes_store.IndexKind = .postings, side_max: u32 = 1024 };

/// postings; sa with its side buffer (a few writes never merge); sa merging at every write (a new base each time).
pub const kinds = [_]Kind{ .{}, .{ .kind = .sa }, .{ .kind = .sa, .side_max = 0 } };

pub const Rig = struct {
    tmp: testing.TmpDir,
    snap: []u8, // the snapshot's path; the journal is snap + ".journal"
    text: Bytes = .{},
    engine: Echo = .{},
    srv: *server.Server,
    gpa: std.mem.Allocator,
    kind: notes_store.IndexKind = .postings,
    side_max: u32 = 1024,

    pub fn init(gpa: std.mem.Allocator) !*Rig {
        return initKind(gpa, .postings, 1024);
    }

    /// initKind() with `k`'s index kind and side buffer.
    pub fn initWith(gpa: std.mem.Allocator, k: Kind) !*Rig {
        return initKind(gpa, k.kind, k.side_max);
    }

    /// The rig with the n-gram index of `kind` (sa: a new base once the side buffer passes `side_max` notes).
    pub fn initKind(gpa: std.mem.Allocator, kind: notes_store.IndexKind, side_max: u32) !*Rig {
        const r = try testing.allocator.create(Rig);
        errdefer testing.allocator.destroy(r);
        r.* = .{ .tmp = testing.tmpDir(.{}), .snap = undefined, .srv = undefined, .gpa = gpa, .kind = kind, .side_max = side_max };
        errdefer r.tmp.cleanup();
        try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json", .data = snapshot });
        r.snap = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/snap.json", .{r.tmp.sub_path});
        errdefer testing.allocator.free(r.snap);
        r.srv = try server.Server.init(testing.allocator, testing.io, r.engine.engine(), r.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
        errdefer r.srv.deinit();
        try r.open();
        return r;
    }

    /// The store opened from the snapshot and its journal, as a server start does.
    pub fn open(r: *Rig) !void {
        var problem: []const u8 = "";
        const s = notes_route.Store.openIndexed(r.gpa, testing.io, r.snap, r.text.text(), "", r.kind, &problem) catch |e| {
            std.debug.print("open: {s}\n", .{problem});
            return e;
        };
        s.today = today;
        s.side_max = r.side_max;
        r.srv.notes = s;
    }

    /// The server stopped and started again: the store dropped and opened from the same files.
    pub fn restart(r: *Rig) !void {
        r.srv.notes.?.deinit();
        r.srv.notes = null;
        try r.open();
    }

    pub fn store(r: *Rig) *notes_route.Store {
        return r.srv.notes.?;
    }

    pub fn deinit(r: *Rig) void {
        if (r.srv.notes) |s| s.deinit();
        r.srv.notes = null;
        r.srv.deinit();
        testing.allocator.free(r.snap);
        r.tmp.cleanup();
        testing.allocator.destroy(r);
    }

    /// The journal file's bytes.
    pub fn journal(r: *Rig, a: std.mem.Allocator) ![]u8 {
        return r.tmp.dir.readFileAlloc(testing.io, "snap.json.journal", a, .unlimited);
    }
};

pub const Reply = struct { status: u16, body: []const u8, value: json.Value };

/// One request through the router (the key check is the listener's, tested apart); the reply parsed into `a`.
pub fn request(srv: *server.Server, a: std.mem.Allocator, method: []const u8, path: []const u8, data: ?[]const u8) !Reply {
    return requestSized(srv, a, method, path, data, null);
}

/// request() with Content-Length `declared` and no body sent (a body the server must refuse unread).
pub fn requestSized(srv: *server.Server, a: std.mem.Allocator, method: []const u8, path: []const u8, data: ?[]const u8, declared: ?usize) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const len = declared orelse if (data) |b| b.len else 0;
    const head = if (data != null or declared != null) try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ method, path, len }) else try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: t\r\n\r\n", .{ method, path });
    const sender = try std.Thread.spawn(.{}, sendAll, .{ pair[0], head, if (declared == null) data orelse "" else "" });
    var conn = try Conn.init(testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 5000, .read_ms = 5000, .write_ms = 5000 };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const ready = conn.readRequest(arena.allocator(), false) catch .closed;
    if (ready != .ready) {
        sender.join();
        return error.BadRequest;
    }
    var raw: []u8 = &.{};
    const reader = try std.Thread.spawn(.{}, readAll, .{ pair[0], a, &raw });
    routes.dispatch(srv, &conn, arena.allocator());
    _ = std.c.shutdown(pair[1], std.posix.SHUT.WR);
    sender.join();
    reader.join();
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    const code = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    const text = raw[split + 4 ..];
    const value = switch (try json.parse(a, text)) {
        .ok => |v| v,
        .err => json.Value.null,
    };
    return .{ .status = code, .body = text, .value = value };
}

fn sendAll(fd: std.c.fd_t, head: []const u8, data: []const u8) void {
    for ([_][]const u8{ head, data }) |part| {
        var sent: usize = 0;
        while (sent < part.len) {
            const n = std.c.write(fd, part[sent..].ptr, part.len - sent);
            if (n <= 0) return;
            sent += @intCast(n);
        }
    }
}

/// Reads one whole response (its head, then Content-Length bytes), giving up after 10 s without data.
fn readAll(fd: std.c.fd_t, a: std.mem.Allocator, out: *[]u8) void {
    var got: std.ArrayList(u8) = .empty;
    var tmp: [8192]u8 = undefined;
    while (true) {
        if (std.mem.indexOf(u8, got.items, "\r\n\r\n")) |end| {
            const head = got.items[0..end];
            const at = std.mem.indexOf(u8, head, "Content-Length:") orelse break;
            const line = head[at + "Content-Length:".len ..];
            const stop = std.mem.indexOfScalar(u8, line, '\r') orelse line.len;
            const len = std.fmt.parseInt(usize, std.mem.trim(u8, line[0..stop], " "), 10) catch break;
            if (got.items.len >= end + 4 + len) break;
        }
        var pfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        if ((std.posix.poll(&pfd, 10_000) catch 0) == 0) break;
        const n = std.c.read(fd, &tmp, tmp.len);
        if (n <= 0) break;
        got.appendSlice(a, tmp[0..@intCast(n)]) catch break;
    }
    out.* = got.items;
}

/// The error message of a refused reply.
pub fn message(r: Reply) []const u8 {
    const e = r.value.get("error") orelse return "";
    return e.strField("message") orelse "";
}
