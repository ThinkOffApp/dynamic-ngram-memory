//! Keep the pristine engine export separate from the standalone adapter/build glue.
const std = @import("std");
test "standalone core declarations" {
    std.testing.refAllDecls(@import("server/notes/model.zig"));
    std.testing.refAllDecls(@import("server/notes/index.zig"));
    std.testing.refAllDecls(@import("server/notes/journal.zig"));
    std.testing.refAllDecls(@import("server/notes/storage.zig").Storage);
    std.testing.refAllDecls(@import("server/notes/operations.zig"));
}

// This byte codec is test-only and matches the explicitly byte-tokenized fixture.
fn fixtureEncode(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, _: bool) @import("server/model_text.zig").Error![]u32 {
    const ids = try a.alloc(u32, bytes.len);
    for (bytes, ids) |byte, *id| id.* = byte;
    return ids;
}

test "standalone storage loads and releases byte fixture without a host engine" {
    const Storage = @import("server/notes/storage.zig").Storage;
    var context: u8 = 0;
    const text: @import("server/model_text.zig").Text = .{
        .ctx = &context,
        .encode_fn = fixtureEncode,
        .vocabulary_size = 256,
    };
    var problem: []const u8 = "";
    const store = try Storage.load(std.testing.allocator, @embedFile("server/notes/testdata/tb_empty_bytes_snapshot.json"), text, null, &problem);
    defer store.deinit();
    try std.testing.expectEqual(@as(usize, 0), store.model.notes.items.len);
    try std.testing.expect(store.journal == null); // fixture-only mode is not a writable store
}

test "standalone durable write survives close and reopen with stable id" {
    const Storage = @import("server/notes/storage.zig").Storage;
    const ops = @import("server/notes/operations.zig");
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const temp = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", temp);
    const path = try std.fs.path.join(temp, &.{ dir, "notes.journal" });
    var context: u8 = 0;
    const text: @import("server/model_text.zig").Text = .{
        .ctx = &context,
        .encode_fn = fixtureEncode,
        .vocabulary_size = 256,
    };
    const snapshot = @embedFile("server/notes/testdata/tb_empty_bytes_snapshot.json");
    var problem: []const u8 = "";
    var id: u32 = 0;
    var other_id: u32 = 0;
    {
        const store = try Storage.load(a, snapshot, text, path, &problem);
        defer store.deinit();
        try store.beginRun(temp, .{});
        var refusal: ops.Refusal = .{};
        const written = try ops.perform(store, temp, .{
            .op = .add,
            .text = "My aunt Brigitta grows tomatoes.",
        }, 739900000000, &refusal);
        id = written.note.?.id;
        try std.testing.expectEqualStrings("w1", written.source);
        const other = try ops.perform(store, temp, .{
            .op = .add,
            .text = "My cousin Aurelio repairs clocks.",
        }, 739900000001, &refusal);
        other_id = other.note.?.id;
        try std.testing.expectEqual(@as(usize, 2), store.model.notes.items.len);
    }
    {
        const store = try Storage.load(a, snapshot, text, path, &problem);
        defer store.deinit();
        try std.testing.expectEqual(@as(usize, 2), store.model.notes.items.len);
        const note = store.model.notes.items[store.model.placeOf(id).?];
        try std.testing.expectEqualStrings("My aunt Brigitta grows tomatoes.", note.text());
        try std.testing.expectEqualStrings("w1", note.source().?);
        try std.testing.expectEqual(@as(u64, 3), store.log.next_source);
        try store.beginRun(temp, .{});
        var erased = try @import("standalone_erase.zig").prepare(store, temp, id);
        defer erased.deinit();
        try std.testing.expect(erased.intent.short); // no assertion about engine caches
        try erased.commit(temp);
        try erased.finish(temp);
        try std.testing.expectEqual(@as(usize, 1), store.model.notes.items.len);
        try std.testing.expectEqualStrings("My cousin Aurelio repairs clocks.", store.model.notes.items[store.model.placeOf(other_id).?].text());
        const disk = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, temp, .limited(1 << 20));
        try std.testing.expect(std.mem.indexOf(u8, disk, "My aunt Brigitta grows tomatoes.") == null);
        try std.testing.expect(std.mem.indexOf(u8, disk, "My cousin Aurelio repairs clocks.") != null);
    }
    {
        const store = try Storage.load(a, snapshot, text, path, &problem);
        defer store.deinit();
        try std.testing.expectEqual(@as(usize, 1), store.model.notes.items.len);
        try std.testing.expectEqualStrings("My cousin Aurelio repairs clocks.", store.model.notes.items[store.model.placeOf(other_id).?].text());
        try std.testing.expectEqual(@as(u64, 3), store.log.next_source); // erase cannot reuse w1
    }
}
