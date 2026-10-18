//! --notes at startup: a snapshot naming a tokenizer.json SHA-256 starts only on the model's, read and equal.
const std = @import("std");
const notes_route = @import("notes_route.zig");
const Bytes = @import("notes_test.zig").Bytes;
const testing = std.testing;
const Sha256 = std.crypto.hash.sha2.Sha256;

const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");
const tokenizer = "{\"model\": {\"type\": \"bytes\"}}";

/// A test's folder: the snapshot (naming `sha`, or no hash) at snap.json, and a model folder at model/.
const Setup = struct {
    tmp: std.testing.TmpDir,
    snap: []u8,
    model: []u8,

    fn init(sha: ?[]const u8) !Setup {
        const a = testing.allocator;
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const text = if (sha) |h| blk: {
            const named = try std.fmt.allocPrint(a, "\"tokenizer\":{{\"name\":\"bytes\",\"sha256\":\"{s}\"}}", .{h});
            defer a.free(named);
            break :blk try std.mem.replaceOwned(u8, a, snapshot, "\"tokenizer\":{\"name\":\"bytes\"}", named);
        } else try a.dupe(u8, snapshot);
        defer a.free(text);
        if (sha != null) try testing.expect(!std.mem.eql(u8, text, snapshot)); // the hash is really named
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json", .data = text });
        try tmp.dir.createDir(testing.io, "model", .default_dir);
        const snap = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/snap.json", .{tmp.sub_path});
        errdefer a.free(snap);
        return .{ .tmp = tmp, .snap = snap, .model = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/model", .{tmp.sub_path}) };
    }

    fn deinit(s: *Setup) void {
        testing.allocator.free(s.snap);
        testing.allocator.free(s.model);
        s.tmp.cleanup();
    }

    fn tokenizerFile(s: *Setup, data: []const u8) !void {
        try s.tmp.dir.writeFile(testing.io, .{ .sub_path = "model/tokenizer.json", .data = data });
    }

    /// Store.open as serve does (limit: the tokenizer.json read limit; null, the server's).
    fn open(s: *Setup, model_dir: []const u8, limit: ?usize, problem: *[]const u8) !*notes_route.Store {
        var text: Bytes = .{};
        return notes_route.Store.openLimited(testing.allocator, testing.io, s.snap, text.text(), model_dir, limit orelse notes_route.tokenizer_limit, problem);
    }
};

fn hexOf(data: []const u8) [64]u8 {
    var out: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(data, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

test "a snapshot naming the model's tokenizer.json SHA-256 starts, checked by its hash" {
    const sha = hexOf(tokenizer);
    var s = try Setup.init(&sha);
    defer s.deinit();
    try s.tokenizerFile(tokenizer);
    var problem: []const u8 = "";
    const store = try s.open(s.model, null, &problem);
    defer store.deinit();
    try testing.expectEqual(notes_route.Check.sha256, store.check);
    var buf: [1024]u8 = undefined;
    const line = store.startupLine(&buf, "snap.json");
    try testing.expect(std.mem.indexOf(u8, line, "SHA-256 equals the snapshot's") != null);
    try testing.expect(std.mem.indexOf(u8, line, "vocabulary-only") == null);
}

test "a snapshot naming no tokenizer hash starts on its vocabulary size alone, and says so" {
    var s = try Setup.init(null);
    defer s.deinit();
    var problem: []const u8 = "";
    const store = try s.open(s.model, null, &problem); // no tokenizer.json at all: nothing to check it against
    defer store.deinit();
    try testing.expectEqual(notes_route.Check.vocabulary_only, store.check);
    var buf: [1024]u8 = undefined;
    const line = store.startupLine(&buf, "snap.json");
    try testing.expect(std.mem.indexOf(u8, line, "vocabulary-only compatibility") != null);
    try testing.expect(std.mem.indexOf(u8, line, "names no tokenizer.json SHA-256") != null);
}

test "a named hash that is not the model's tokenizer.json refuses startup" {
    const sha = hexOf("another tokenizer");
    var s = try Setup.init(&sha);
    defer s.deinit();
    try s.tokenizerFile(tokenizer);
    var problem: []const u8 = "";
    try testing.expectError(error.TokenizerMismatch, s.open(s.model, null, &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "not this model's") != null);
}

test "a named hash with no tokenizer.json in the model folder refuses startup" {
    const sha = hexOf(tokenizer);
    var s = try Setup.init(&sha);
    defer s.deinit();
    var problem: []const u8 = "";
    try testing.expectError(error.FileNotFound, s.open(s.model, null, &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "has no tokenizer.json") != null);
    // and with no model folder at all
    problem = "";
    try testing.expectError(error.NoModelDir, s.open("", null, &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "no model folder") != null);
}

test "a named hash with an unreadable tokenizer.json refuses startup" {
    const sha = hexOf(tokenizer);
    var s = try Setup.init(&sha);
    defer s.deinit();
    // a directory where the file should be: it opens and cannot be read, for any user
    try s.tmp.dir.createDir(testing.io, "model/tokenizer.json", .default_dir);
    var problem: []const u8 = "";
    try testing.expectError(error.IsDir, s.open(s.model, null, &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "cannot be read") != null);
}

test "a named hash with a tokenizer.json its user may not read refuses startup" {
    const sha = hexOf(tokenizer);
    var s = try Setup.init(&sha);
    defer s.deinit();
    try s.tokenizerFile(tokenizer);
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/tokenizer.json", .{s.model}, 0);
    defer testing.allocator.free(path);
    try testing.expectEqual(@as(c_int, 0), std.c.fchmodat(std.posix.AT.FDCWD, path, 0o000, 0));
    defer _ = std.c.fchmodat(std.posix.AT.FDCWD, path, 0o600, 0);
    if (std.c.access(path, std.c.R_OK) == 0) return error.SkipZigTest; // a user who reads every file (root)
    var problem: []const u8 = "";
    try testing.expectError(error.AccessDenied, s.open(s.model, null, &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "cannot be read") != null);
}

test "a named hash with a tokenizer.json over the read limit refuses startup" {
    const sha = hexOf(tokenizer);
    var s = try Setup.init(&sha);
    defer s.deinit();
    try s.tokenizerFile(tokenizer);
    var problem: []const u8 = "";
    try testing.expectError(error.StreamTooLong, s.open(s.model, tokenizer.len, &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "over the read limit") != null);
    // the same file under the server's limit passes: the limit, not the file, refused it
    const store = try s.open(s.model, null, &problem);
    store.deinit();
}
