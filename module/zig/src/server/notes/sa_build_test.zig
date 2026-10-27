//! Suffix-array packing and occurrence boundaries.
const std = @import("std");
const suffix = @import("sa.zig");
const Packed = suffix.Packed;
const Rank = suffix.Rank;
const Source = suffix.Source;
const build = suffix.build;

test "packed ints of each width read back" {
    const a = std.testing.allocator;
    for ([_]u6{ 1, 3, 7, 8, 11, 21, 31, 32 }) |w| {
        const p = try Packed.alloc(a, 1000, w);
        defer a.free(p.bytes);
        const mask: u64 = (@as(u64, 1) << w) - 1;
        for (0..1000) |i| p.put(i, @intCast((i *% 2654435761) & mask));
        for (0..1000) |i| try std.testing.expectEqual(@as(u32, @intCast((i *% 2654435761) & mask)), p.get(i));
    }
}

test "rank counts the set bits before a place" {
    const a = std.testing.allocator;
    const r = try Rank.init(a, 3000);
    defer a.free(r.words);
    defer a.free(r.blocks);
    var rng = std.Random.DefaultPrng.init(3);
    var want: [3001]u32 = undefined;
    var n: u32 = 0;
    for (0..3000) |i| {
        want[i] = n;
        if (rng.random().boolean()) {
            r.set(i);
            n += 1;
        }
    }
    want[3000] = n;
    r.seal();
    for (0..3001) |i| try std.testing.expectEqual(want[i], r.rank(i));
}

test "a range holds every occurrence of a pattern and nothing else, never across a note" {
    const a = std.testing.allocator;
    const N = struct { keys: []const u32, cur: bool };
    const notes = [_]N{
        .{ .keys = &.{ 5, 7, 5, 7, 9 }, .cur = true }, // 5 7 twice in one note
        .{ .keys = &.{ 7, 9 }, .cur = false },
        .{ .keys = &.{ 9, 5 }, .cur = true },
        .{ .keys = &.{7}, .cur = true }, // ends in 7; the next note starts with 5: "7 5" must not match across them
        .{ .keys = &.{ 5, 7 }, .cur = true },
    };
    var ids: [8]u32 = @splat(0);
    var with: [notes.len]Source = undefined;
    for (notes, &with) |n, *w| w.* = .{ .keys = n.keys, .ids = ids[0..n.keys.len], .cur = n.cur };
    const b = try build(a, &with, &.{});
    defer b.deinit();
    var buf: [4]u32 = undefined;
    const Want = struct { pat: []const u32, cur: []const u32, hist: []const u32 };
    const cases = [_]Want{
        .{ .pat = &.{ 5, 7 }, .cur = &.{ 0, 0, 4 }, .hist = &.{} },
        .{ .pat = &.{ 7, 9 }, .cur = &.{0}, .hist = &.{1} },
        .{ .pat = &.{ 7, 5 }, .cur = &.{0}, .hist = &.{} },
        .{ .pat = &.{9}, .cur = &.{ 0, 2 }, .hist = &.{1} },
        .{ .pat = &.{ 5, 7, 9 }, .cur = &.{0}, .hist = &.{} },
        .{ .pat = &.{ 9, 7 }, .cur = &.{}, .hist = &.{} },
    };
    for (cases) |c| {
        const pat = b.codes(c.pat, &buf).?;
        for ([_]Packed{ b.sa_cur, b.sa_hist }, [_][]const u32{ c.cur, c.hist }) |sa, want| {
            const r = b.range(sa, pat);
            var got: std.ArrayList(u32) = .empty;
            defer got.deinit(a);
            for (r[0]..r[1]) |i| try got.append(a, b.placeOf(sa.get(i)));
            std.sort.pdq(u32, got.items, {}, std.sort.asc(u32));
            try std.testing.expectEqualSlices(u32, want, got.items);
        }
    }
    try std.testing.expect(b.codes(&.{ 5, 6 }, &buf) == null); // 6 is in no note
}
