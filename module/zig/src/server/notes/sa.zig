//! The notes' n-grams as a suffix array (Infini-gram, Liu et al. 2024): an n-gram is one binary-searched range.
const std = @import("std");
const exact = @import("exact_alloc.zig");
const Allocator = std.mem.Allocator;

/// The separator's code: after every note, below every key, so no n-gram crosses a note and no pattern matches it.
pub const sep: u32 = 0;

/// Unsigned ints of `width` bits (1 to 32), packed little-endian, with 8 bytes of tail so any read is one u64 load.
pub const Packed = struct {
    bytes: []u8 = &.{},
    width: u6 = 1,
    len: usize = 0,

    pub fn alloc(a: Allocator, len: usize, width: u6) Allocator.Error!Packed {
        const bytes = try a.alloc(u8, (len * width + 7) / 8 + 8);
        @memset(bytes, 0);
        return .{ .bytes = bytes, .width = width, .len = len };
    }

    pub inline fn get(p: Packed, i: usize) u32 {
        const bit = i * p.width;
        const word = std.mem.readInt(u64, p.bytes[bit >> 3 ..][0..8], .little);
        const mask = (@as(u64, 1) << p.width) - 1;
        return @intCast((word >> @intCast(bit & 7)) & mask);
    }

    /// Only into zeroed bits (a build writes each place once).
    pub fn put(p: Packed, i: usize, v: u32) void {
        const bit = i * p.width;
        const at = p.bytes[bit >> 3 ..][0..8];
        const word = std.mem.readInt(u64, at, .little);
        std.mem.writeInt(u64, at, word | (@as(u64, v) << @intCast(bit & 7)), .little);
    }

    pub fn byteLen(p: Packed) usize {
        return p.bytes.len;
    }
};

/// The bits an unsigned value up to `max` needs (at least 1).
pub fn widthFor(max: u64) u6 {
    return @intCast(@max(1, 64 - @clz(max)));
}

/// rank over a bit vector: the set bits before a place, from a count per 64 bits and one popcount.
pub const Rank = struct {
    words: []u64 = &.{},
    blocks: []u32 = &.{}, // set bits before each word

    pub fn init(a: Allocator, bits: usize) Allocator.Error!Rank {
        const nw = (bits + 63) / 64 + 1;
        const words = try a.alloc(u64, nw);
        @memset(words, 0);
        const blocks = a.alloc(u32, nw) catch |e| {
            a.free(words);
            return e;
        };
        return .{ .words = words, .blocks = blocks };
    }

    pub fn set(r: Rank, i: usize) void {
        r.words[i >> 6] |= @as(u64, 1) << @intCast(i & 63);
    }

    /// The counts, once every bit is set.
    pub fn seal(r: Rank) void {
        var total: u32 = 0;
        for (r.blocks, r.words) |*b, w| {
            b.* = total;
            total += @popCount(w);
        }
    }

    /// The set bits strictly before `i`.
    pub inline fn rank(r: Rank, i: usize) u32 {
        const w = i >> 6;
        const below = (@as(u64, 1) << @intCast(i & 63)) - 1;
        return r.blocks[w] + @popCount(r.words[w] & below);
    }

    pub fn byteLen(r: Rank) usize {
        return r.words.len * 8 + r.blocks.len * 4;
    }
};

/// One note's folded keys and token ids for the builder; `cur`: its suffixes go to the current view's array.
pub const Source = struct { keys: []const u32, ids: []const u32, cur: bool };

pub const BuildError = error{ OutOfMemory, Unsupported };

/// The suffix arrays of some notes (places 0..notes-1), immutable once built and shared by the versions that read it.
pub const Base = struct {
    mem: exact.Exact,
    refs: std.atomic.Value(u32) = .init(1),
    notes: u32 = 0, // the notes it covers: places 0..notes-1
    dict: []const u32 = &.{}, // the keys it holds, ascending; a key's code is its place here + 1
    text: Packed = .{}, // each note's codes, then sep
    seps: Rank = .{}, // a bit at each separator: rank(pos) is the note place of a position
    sa_cur: Packed = .{}, // positions of the suffixes of notes current when it was built, in suffix order
    sa_hist: Packed = .{}, // and of the others
    wordy: []u64 = &.{}, // by code: its tokens are words (a 1-gram of it is a key unless punct is set)
    tokens: usize = 0,

    pub fn retain(b: *Base) *Base {
        _ = b.refs.fetchAdd(1, .monotonic);
        return b;
    }

    pub fn release(b: *Base) void {
        if (b.refs.fetchSub(1, .acq_rel) != 1) return;
        const gpa = b.mem.child;
        b.mem.deinit();
        gpa.destroy(b);
    }

    /// The bytes it holds (its record list included), for the size bench.
    pub fn bytes(b: *const Base) usize {
        return b.mem.bytes() + @sizeOf(Base);
    }

    /// A key's code, or null when no note of the base holds it.
    pub fn codeOf(b: *const Base, key: u32) ?u32 {
        var lo: usize = 0;
        var hi: usize = b.dict.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (b.dict[mid] == key) return @intCast(mid + 1);
            if (b.dict[mid] < key) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    pub inline fn placeOf(b: *const Base, pos: usize) u32 {
        return b.seps.rank(pos);
    }

    pub inline fn wordyCode(b: *const Base, code: u32) bool {
        return (b.wordy[code >> 6] >> @intCast(code & 63)) & 1 != 0;
    }

    /// The suffix at `pos` against `pat`: -1 below, 0 when it starts with it, 1 above.
    inline fn order(b: *const Base, pos: usize, pat: []const u32) i2 {
        for (pat, 0..) |c, k| {
            const t = b.text.get(pos + k);
            if (t != c) return if (t < c) -1 else 1; // sep (0) is below every code, and ends the suffix
        }
        return 0;
    }

    /// The range of `sa` whose suffixes start with the codes `pat`.
    pub fn range(b: *const Base, sa: Packed, pat: []const u32) [2]usize {
        var lo: usize = 0;
        var hi: usize = sa.len;
        while (lo < hi) { // the first suffix not below
            const mid = lo + (hi - lo) / 2;
            if (b.order(sa.get(mid), pat) < 0) lo = mid + 1 else hi = mid;
        }
        const first = lo;
        hi = sa.len;
        while (lo < hi) { // the first suffix above
            const mid = lo + (hi - lo) / 2;
            if (b.order(sa.get(mid), pat) <= 0) lo = mid + 1 else hi = mid;
        }
        return .{ first, lo };
    }

    /// Codes of `keys`, or null when the base lacks one of them.
    pub fn codes(b: *const Base, keys: []const u32, out: []u32) ?[]u32 {
        for (keys, out[0..keys.len]) |k, *c| c.* = b.codeOf(k) orelse return null;
        return out[0..keys.len];
    }
};

const Entry = struct { prefix: u64, pos: u32 };

const SortCtx = struct {
    text: []const u32,
    /// Suffix order: the codes up to and including the first separator, then the position (equal suffixes).
    fn less(c: SortCtx, x: Entry, y: Entry) bool {
        if (x.prefix != y.prefix) return x.prefix < y.prefix;
        // the same first three codes; a separator among them ends both suffixes equal
        var k: usize = 0;
        while (k < 3) : (k += 1) if (c.text[x.pos + k] == sep) return x.pos < y.pos;
        while (true) : (k += 1) {
            const a = c.text[x.pos + k];
            const b = c.text[y.pos + k];
            if (a != b) return a < b;
            if (a == sep) return x.pos < y.pos;
        }
    }
};

/// A stable LSD radix sort of `xs` by the low `bits` bits of their prefixes, 11 bits a pass (`tmp` as long as `xs`).
fn radix(xs: []Entry, tmp: []Entry, bits: usize) void {
    var src = xs;
    var dst = tmp;
    var shift: usize = 0;
    while (shift < bits) : (shift += 11) {
        var count: [2049]usize = @splat(0);
        for (src) |e| count[((e.prefix >> @intCast(shift)) & 2047) + 1] += 1;
        for (1..2049) |d| count[d] += count[d - 1];
        for (src) |e| {
            const d = (e.prefix >> @intCast(shift)) & 2047;
            dst[count[d]] = e;
            count[d] += 1;
        }
        std.mem.swap([]Entry, &src, &dst);
    }
    if (src.ptr != xs.ptr) @memcpy(xs, src);
}

/// The base over `notes` (places 0..len-1); the ids of one key must agree on `vwordy` (Unsupported otherwise).
pub fn build(gpa: Allocator, notes: []const Source, vwordy: []const bool) BuildError!*Base {
    const b = try gpa.create(Base);
    b.* = .{ .mem = exact.Exact.init(gpa) };
    errdefer {
        b.mem.deinit();
        gpa.destroy(b);
    }
    const a = b.mem.allocator();
    var tokens: usize = 0;
    var n_cur: usize = 0;
    for (notes) |n| {
        tokens += n.keys.len;
        if (n.cur) n_cur += n.keys.len;
    }
    // the dictionary: every key held, ascending (a hash set first: the store's distinct keys are few)
    var codes: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer codes.deinit(gpa);
    for (notes) |n| for (n.keys) |k| {
        try codes.put(gpa, k, 0);
    };
    const distinct = codes.count();
    const dict = try a.alloc(u32, distinct);
    var kit = codes.keyIterator();
    var at: usize = 0;
    while (kit.next()) |k| : (at += 1) dict[at] = k.*;
    std.sort.pdq(u32, dict, {}, std.sort.asc(u32));
    for (dict, 1..) |k, c| codes.getPtr(k).?.* = @intCast(c);
    b.dict = dict;
    // the text as u32 codes for the sort (packed after), and each code's wordiness (one key's ids must agree)
    const positions = tokens + notes.len;
    const text = try gpa.alloc(u32, positions + 3);
    defer gpa.free(text);
    @memset(text[positions..], sep);
    b.seps = try Rank.init(a, positions);
    b.wordy = try a.alloc(u64, (distinct + 1 + 63) / 64);
    @memset(b.wordy, 0);
    const known = try gpa.alloc(u64, b.wordy.len);
    defer gpa.free(known);
    @memset(known, 0);
    at = 0;
    for (notes) |n| {
        for (n.keys, n.ids) |k, t| {
            const c = codes.get(k).?;
            text[at] = c;
            at += 1;
            const w = t < vwordy.len and vwordy[t];
            const bit = @as(u64, 1) << @intCast(c & 63);
            if (known[c >> 6] & bit != 0) {
                if ((b.wordy[c >> 6] & bit != 0) != w) return error.Unsupported;
            } else {
                known[c >> 6] |= bit;
                if (w) b.wordy[c >> 6] |= bit;
            }
        }
        text[at] = sep;
        b.seps.set(at);
        at += 1;
    }
    b.seps.seal();
    const cw = widthFor(distinct);
    b.text = try Packed.alloc(a, positions, cw);
    for (text[0..positions], 0..) |c, i| b.text.put(i, c);
    // each view's suffixes radix-sorted by their first three codes, then runs of an equal sep-free prefix sorted whole
    const fits = 3 * @as(usize, cw) <= 63;
    const ents = try gpa.alloc(Entry, tokens);
    defer gpa.free(ents);
    var ci: usize = 0;
    var hi_at: usize = n_cur;
    at = 0;
    for (notes) |n| {
        for (0..n.keys.len) |j| {
            const p = at + j;
            const pre: u64 = if (fits) (@as(u64, text[p]) << @intCast(2 * cw)) | (@as(u64, text[p + 1]) << cw) | text[p + 2] else 0;
            const e: Entry = .{ .prefix = pre, .pos = @intCast(p) };
            if (n.cur) {
                ents[ci] = e;
                ci += 1;
            } else {
                ents[hi_at] = e;
                hi_at += 1;
            }
        }
        at += n.keys.len + 1;
    }
    const tmp = try gpa.alloc(Entry, tokens);
    defer gpa.free(tmp);
    const ctx: SortCtx = .{ .text = text };
    for ([2][2]usize{ .{ 0, n_cur }, .{ n_cur, tokens } }) |r| {
        const view = ents[r[0]..r[1]];
        if (!fits) {
            std.sort.pdq(Entry, view, ctx, SortCtx.less);
            continue;
        }
        radix(view, tmp[0..view.len], 3 * @as(usize, cw));
        var lo: usize = 0;
        while (lo < view.len) {
            var hi = lo + 1;
            while (hi < view.len and view[hi].prefix == view[lo].prefix) hi += 1;
            const p = view[lo].pos;
            if (hi - lo > 1 and text[p] != sep and text[p + 1] != sep and text[p + 2] != sep)
                std.sort.pdq(Entry, view[lo..hi], ctx, SortCtx.less);
            lo = hi;
        }
    }
    const pw = widthFor(@max(positions, 1) - 1);
    b.sa_cur = try Packed.alloc(a, n_cur, pw);
    for (ents[0..n_cur], 0..) |e, i| b.sa_cur.put(i, e.pos);
    b.sa_hist = try Packed.alloc(a, tokens - n_cur, pw);
    for (ents[n_cur..], 0..) |e, i| b.sa_hist.put(i, e.pos);
    b.notes = @intCast(notes.len);
    b.tokens = tokens;
    return b;
}

/// The distinct n-grams of one view (n_min to n_max; a 1-gram only of a key `keep1` passes), from its sorted suffixes.
pub fn distinctGrams(b: *const Base, sa: Packed, n_min: usize, n_max: usize, ctx: anytype, keep1: fn (@TypeOf(ctx), u32) bool) usize {
    var count: usize = 0;
    var n = n_min;
    while (n <= n_max) : (n += 1) {
        var i: usize = 0;
        while (i < sa.len) : (i += 1) {
            const p = sa.get(i);
            var whole = true;
            for (0..n) |k| whole = whole and b.text.get(p + k) != sep;
            if (!whole) continue;
            if (n == 1 and !keep1(ctx, b.dict[b.text.get(p) - 1])) continue;
            var same = i > 0;
            if (same) {
                const q = sa.get(i - 1);
                for (0..n) |k| same = same and b.text.get(q + k) == b.text.get(p + k);
            }
            if (!same) count += 1;
        }
    }
    return count;
}

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
    defer b.release();
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
