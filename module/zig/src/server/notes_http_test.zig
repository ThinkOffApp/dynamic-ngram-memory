//! /v1/notes over HTTP: each endpoint's happy path and refusals, writes off without a journal, and the key gate.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const rig = @import("notes_rig_test.zig");
const notes_route = @import("notes_route.zig");
const listener_mod = @import("listener.zig");
const testing = std.testing;

test "each endpoint answers, and refuses what it must with the status and message it says" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const srv = r.srv;
    // happy paths: add, list, one note, change, take back (default text), erase
    const added = try rig.request(srv, a, "POST", "/v1/notes", "{\"text\": \"My favourite colour is green.\"}");
    try testing.expectEqual(@as(u16, 201), added.status);
    try testing.expectEqual(@as(?i64, 30), added.value.get("id").?.int64());
    try testing.expectEqualStrings("current", added.value.strField("state").?);
    try testing.expectEqual(@as(?i64, 1), added.value.get("version").?.int64());
    const listed = try rig.request(srv, a, "GET", "/v1/notes", null);
    try testing.expectEqual(@as(u16, 200), listed.status);
    try testing.expectEqual(@as(usize, 25), listed.value.get("data").?.array.len); // 24 current in the snapshot, and it
    const all = try rig.request(srv, a, "GET", "/v1/notes?history=1", null);
    try testing.expectEqual(@as(usize, 30), all.value.get("data").?.array.len);
    const one = try rig.request(srv, a, "GET", "/v1/notes/30", null);
    try testing.expectEqualStrings("[30] \"My favourite colour is green.\"", one.value.strField("line").?);
    const changed = try rig.request(srv, a, "POST", "/v1/notes/30", "{\"text\": \"My favourite colour is blue now.\"}");
    try testing.expectEqual(@as(u16, 201), changed.status);
    try testing.expectEqual(@as(?i64, 2), changed.value.get("version").?.int64());
    const old = try rig.request(srv, a, "GET", "/v1/notes/30", null);
    try testing.expectEqualStrings("superseded", old.value.strField("state").?);
    try testing.expectEqualStrings("[30] \"My favourite colour is green.\" (no longer current: replaced on 16 October 2026 by [31])", old.value.strField("line").?);
    const took = try rig.request(srv, a, "DELETE", "/v1/notes/31", null);
    try testing.expectEqual(@as(u16, 201), took.status);
    try testing.expectEqualStrings("I take that back.", took.value.get("note").?.strField("text").?);
    const gone = try rig.request(srv, a, "GET", "/v1/notes/31", null);
    try testing.expectEqualStrings("tombstoned", gone.value.strField("state").?);
    const same = try rig.request(srv, a, "POST", "/v1/notes", "{\"text\": \"My dog has grey eyes\", \"subject\": \"my dog\"}");
    try testing.expectEqual(@as(u16, 200), same.status); // a snapshot note in other words: already known, nothing written
    try testing.expectEqualStrings("ALREADY_KNOWN", same.value.strField("verdict").?);
    try testing.expectEqual(@as(?i64, 6), same.value.get("id").?.int64());

    const Case = struct { method: []const u8, path: []const u8, body: ?[]const u8 = null, status: u16, has: []const u8 };
    const cases = [_]Case{
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": ", .status = 400, .has = "not JSON" },
        .{ .method = "POST", .path = "/v1/notes", .body = "[1]", .status = 400, .has = "JSON object" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{}", .status = 400, .has = "text is required" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": 3}", .status = 400, .has = "text must be a string" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": \"  \"}", .status = 400, .has = "must not be empty" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": \"x\", \"colour\": 1}", .status = 400, .has = "unknown field colour" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": \"x\", \"subject\": \" me\"}", .status = 400, .has = "subject must be" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": \"x\", \"replaces\": \"3\"}", .status = 400, .has = "replaces must be a note id" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": \"\\ud800\"}", .status = 400, .has = "valid Unicode" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": \"x\", \"replaces\": 999}", .status = 404, .has = "no note has this id" },
        .{ .method = "POST", .path = "/v1/notes", .body = "{\"text\": \"x\", \"replaces\": 2, \"subject\": \"my dog\"}", .status = 400, .has = "keeps the subject" },
        .{ .method = "POST", .path = "/v1/notes/999", .body = "{\"text\": \"x\"}", .status = 404, .has = "no note has this id" },
        .{ .method = "POST", .path = "/v1/notes/2", .body = "{\"text\": \"x\", \"subject\": \"me\"}", .status = 400, .has = "unknown field subject" },
        .{ .method = "POST", .path = "/v1/notes/30", .body = "{\"text\": \"x\"}", .status = 409, .has = "no longer current" },
        .{ .method = "DELETE", .path = "/v1/notes/31", .status = 409, .has = "taken back already" },
        .{ .method = "DELETE", .path = "/v1/notes/32", .status = 409, .has = "cannot be taken back" },
        .{ .method = "DELETE", .path = "/v1/notes/999", .status = 404, .has = "no note has this id" },
        .{ .method = "DELETE", .path = "/v1/notes/999?erase=true", .status = 404, .has = "no note has this id" },
        .{ .method = "DELETE", .path = "/v1/notes/5?erase=true", .status = 409, .has = "comes from the snapshot" },
        .{ .method = "DELETE", .path = "/v1/notes/30?erase=true", .body = "{\"text\": \"x\"}", .status = 400, .has = "takes no body" },
        .{ .method = "DELETE", .path = "/v1/notes", .status = 405, .has = "DELETE takes a note" },
        .{ .method = "PUT", .path = "/v1/notes", .body = "{}", .status = 405, .has = "use GET or POST" },
        .{ .method = "PATCH", .path = "/v1/notes/2", .body = "{}", .status = 405, .has = "use GET or POST" },
        .{ .method = "GET", .path = "/v1/notes/999", .status = 404, .has = "no note has this id" },
        .{ .method = "GET", .path = "/v1/notes/abc", .status = 404, .has = "whole number" },
    };
    for (cases) |c| {
        const got = try rig.request(srv, a, c.method, c.path, c.body);
        errdefer std.debug.print("{s} {s} -> {d} {s}\n", .{ c.method, c.path, got.status, got.body });
        try testing.expectEqual(c.status, got.status);
        try testing.expect(std.mem.indexOf(u8, rig.message(got), c.has) != null);
    }
    // a note's text over 16 KiB, and a body over 1 MiB (refused from its Content-Length, unread)
    const long = try a.alloc(u8, (16 << 10) + 1);
    @memset(long, 'a');
    const big = try rig.request(srv, a, "POST", "/v1/notes", try std.fmt.allocPrint(a, "{{\"text\": \"{s}\"}}", .{long}));
    try testing.expectEqual(@as(u16, 413), big.status);
    const huge = try rig.requestSized(srv, a, "POST", "/v1/notes", null, (1 << 20) + 1);
    try testing.expectEqual(@as(u16, 413), huge.status);
    // none of the refusals wrote anything: the notes are as the happy paths left them
    const after = try rig.request(srv, a, "GET", "/v1/notes?history=1", null);
    try testing.expectEqual(@as(usize, 32), after.value.get("data").?.array.len);
    // and an erase answers what it took out
    const erased = try rig.request(srv, a, "DELETE", "/v1/notes/30?erase=true", null);
    try testing.expectEqual(@as(u16, 200), erased.status);
    try testing.expectEqual(@as(usize, 1), erased.value.get("removed").?.array.len);
    try testing.expect(erased.value.get("complete").?.bool); // this engine keeps no prompt cache
}

test "without --notes the routes say so, and a store without a journal refuses writes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    const s = r.srv.notes.?;
    r.srv.notes = null;
    defer r.srv.notes = s;
    const off = try rig.request(r.srv, a, "GET", "/v1/notes", null);
    try testing.expectEqual(@as(u16, 404), off.status);
    try testing.expect(std.mem.indexOf(u8, rig.message(off), "without --notes") != null);
    var problem: []const u8 = "";
    const ro = try notes_route.Store.load(testing.allocator, testing.io, rig.snapshot, r.text.text(), &problem);
    r.srv.notes = ro;
    defer ro.deinit();
    const refused = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"x\"}");
    try testing.expectEqual(@as(u16, 503), refused.status);
    try testing.expect(std.mem.indexOf(u8, rig.message(refused), "no journal") != null);
    const read = try rig.request(r.srv, a, "GET", "/v1/notes", null);
    try testing.expectEqual(@as(u16, 200), read.status);
}

/// One raw request to 127.0.0.1:`port`, with `key` as a Bearer token when given; the status.
fn over(port: u16, method: []const u8, path: []const u8, body: []const u8, key: ?[]const u8) !u16 {
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (fd < 0) return error.Socket;
    defer _ = std.c.close(fd);
    var addr: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.posix.sockaddr.in)) != 0) return error.Connect;
    var buf: [512]u8 = undefined;
    const auth = if (key) |k| try std.fmt.bufPrint(&buf, "Authorization: Bearer {s}\r\n", .{k}) else "";
    var head: [1024]u8 = undefined;
    const req = try std.fmt.bufPrint(&head, "{s} {s} HTTP/1.1\r\nHost: t\r\n{s}Content-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, path, auth, body.len, body });
    if (std.c.write(fd, req.ptr, req.len) != req.len) return error.Write;
    var out: [16]u8 = undefined;
    var got: usize = 0;
    while (got < 12) {
        const n = std.c.read(fd, out[got..].ptr, out.len - got);
        if (n <= 0) break;
        got += @intCast(n);
    }
    if (got < 12) return error.Short;
    return std.fmt.parseInt(u16, out[9..12], 10);
}

test "writes need the key when the server has one, as every /v1 route does" {
    var problem: []const u8 = "";
    var keys = try server.auth.Store.init(testing.allocator, &.{"sekrit-key"}, "", null, false, &problem);
    defer keys.deinit();
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    r.srv.keys = &keys;
    defer r.srv.keys = null;
    const lis = try listener_mod.Listener.open(.{ .ip4 = .loopback(0) });
    var stop: std.atomic.Value(bool) = .init(false);
    const t = try std.Thread.spawn(.{}, server.Server.serve, .{ r.srv, lis, &stop });
    defer {
        stop.store(true, .release);
        t.join();
        lis.close();
        while (r.srv.open_connections.load(.acquire) > 0) std.Io.sleep(testing.io, .fromMilliseconds(5), .awake) catch {};
    }
    const port = lis.port();
    const body = "{\"text\": \"My favourite colour is green.\"}";
    try testing.expectEqual(@as(u16, 401), try over(port, "POST", "/v1/notes", body, null));
    try testing.expectEqual(@as(u16, 401), try over(port, "POST", "/v1/notes", body, "wrong-key"));
    try testing.expectEqual(@as(u16, 401), try over(port, "DELETE", "/v1/notes/5", "", null));
    try testing.expectEqual(@as(u16, 401), try over(port, "GET", "/v1/notes", "", null));
    try testing.expectEqual(@as(u16, 201), try over(port, "POST", "/v1/notes", body, "sekrit-key"));
    try testing.expectEqual(@as(u16, 200), try over(port, "GET", "/v1/notes", "", "sekrit-key"));
    try testing.expectEqual(@as(usize, 30), r.store().model.notes.items.len); // only the keyed write was made
}
