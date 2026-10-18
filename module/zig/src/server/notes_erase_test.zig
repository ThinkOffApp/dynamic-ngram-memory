//! Erase (LLWZ 4.5): after it, the note's text is in no live allocation of the store and on no file; its ids nowhere.
const std = @import("std");
const server = @import("root.zig");
const rig = @import("notes_rig_test.zig");
const index = @import("notes/index.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

/// Keeps every live allocation, so a test can search all the memory the store holds (as LLWZ's walk_for walks it).
pub const Tracking = struct {
    child: Allocator,
    mutex: std.Io.Mutex = .init,
    live: std.AutoHashMapUnmanaged(usize, usize) = .empty,

    pub fn allocator(t: *Tracking) Allocator {
        return .{ .ptr = t, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = Allocator.noRemap, .free = free } };
    }

    pub fn deinit(t: *Tracking) void {
        t.live.deinit(t.child);
    }

    fn self(ctx: *anyopaque) *Tracking {
        return @ptrCast(@alignCast(ctx));
    }

    fn alloc(ctx: *anyopaque, len: usize, al: Alignment, ret: usize) ?[*]u8 {
        const t = self(ctx);
        const p = t.child.rawAlloc(len, al, ret) orelse return null;
        t.mutex.lockUncancelable(testing.io);
        defer t.mutex.unlock(testing.io);
        t.live.put(t.child, @intFromPtr(p), len) catch {
            t.child.rawFree(p[0..len], al, ret);
            return null;
        };
        return p;
    }

    fn resize(ctx: *anyopaque, memory: []u8, al: Alignment, new_len: usize, ret: usize) bool {
        const t = self(ctx);
        if (!t.child.rawResize(memory, al, new_len, ret)) return false;
        t.mutex.lockUncancelable(testing.io);
        defer t.mutex.unlock(testing.io);
        t.live.putAssumeCapacity(@intFromPtr(memory.ptr), new_len);
        return true;
    }

    fn free(ctx: *anyopaque, memory: []u8, al: Alignment, ret: usize) void {
        const t = self(ctx);
        t.mutex.lockUncancelable(testing.io);
        _ = t.live.remove(@intFromPtr(memory.ptr));
        t.mutex.unlock(testing.io);
        t.child.rawFree(memory, al, ret);
    }

    /// How many live allocations hold `needle`.
    pub fn holding(t: *Tracking, needle: []const u8) usize {
        var n: usize = 0;
        var it = t.live.iterator();
        while (it.next()) |kv| {
            const bytes = @as([*]const u8, @ptrFromInt(kv.key_ptr.*))[0..kv.value_ptr.*];
            if (std.mem.indexOf(u8, bytes, needle) != null) n += 1;
        }
        return n;
    }
};

/// The secret words, and where the store, the index or the files may still hold them.
const Where = struct { memory: usize, journal: bool, files: usize, ids: usize };

fn look(t: *Tracking, r: *rig.Rig, a: Allocator, own: []const u32, own_keys: []const u32) !Where {
    var w: Where = .{ .memory = t.holding("Quixotic Zanzibar") + t.holding("quixotic zanzibar"), .journal = false, .files = 0, .ids = 0 };
    const j = try r.journal(a);
    w.journal = std.mem.indexOf(u8, j, "Zanzibar") != null or std.mem.indexOf(u8, j, "zanzibar") != null;
    var it = r.tmp.dir.iterate();
    while (try it.next(testing.io)) |e| {
        const bytes = try r.tmp.dir.readFileAlloc(testing.io, e.name, a, .unlimited);
        if (std.mem.indexOf(u8, bytes, "anzibar") != null) w.files += 1;
    }
    // the ids: in no note's tokens, and their keys in no n-gram and no phrase of the index lookups read now
    const s = r.store();
    const m = s.model;
    for (m.notes.items) |*n| for (n.tokens()) |id| if (std.mem.indexOfScalar(u32, own, id) != null) {
        w.ids += 1;
    };
    const v = s.acquire();
    defer s.release(v);
    const keys = own_keys;
    var gi = v.idx.grams.keyIterator();
    while (gi.next()) |g| for (0..4) |k| {
        const key: u32 = @truncate(g.* >> @intCast(32 * k));
        if (std.mem.indexOfScalar(u32, keys, key) != null) w.ids += 1;
    };
    var pi = v.idx.phrases.keyIterator();
    while (pi.next()) |p| for (p.*) |key| if (std.mem.indexOfScalar(u32, keys, key) != null) {
        w.ids += 1;
    };
    // sa: the keys every base still alive holds (the current one, retired versions', none other)
    var bases: std.ArrayList(*const @import("notes/sa.zig").Base) = .empty;
    if (v.idx.base) |b| try bases.append(a, b);
    for (s.retired.items) |old| if (old.idx.base) |b| try bases.append(a, b);
    for (bases.items) |b| for (b.dict) |key| if (std.mem.indexOfScalar(u32, keys, key) != null) {
        w.ids += 1;
    };
    // and the token ids of models an erase left for later (the negative control's)
    for (s.parked.items) |pm| for (pm.notes.items) |*n| for (n.tokens()) |id| if (std.mem.indexOfScalar(u32, own, id) != null) {
        w.ids += 1;
    };
    return w;
}

/// What the search finds before the erase, right after it, and after a restart (or, deferred, after a merge).
const Found = struct { before: Where, after: Where, again: Where };

/// The erase script on a rig with the index of `kind`; `deferred`: the erase leaves its merge for later.
fn eraseRun(kind: @import("notes_store.zig").IndexKind, deferred: bool) !Found {
    var tracking: Tracking = .{ .child = testing.allocator };
    defer tracking.deinit();
    const r = try rig.Rig.initKind(tracking.allocator(), kind, 1024);
    defer r.deinit();
    r.store().defer_erase_merge = deferred;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = r.text.text();
    const secret = "My zebra is called Quixotic Zanzibar.";
    const made = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"My zebra is called Quixotic Zanzibar.\", \"subject\": \"my zebra\"}");
    const id = made.value.get("id").?.int64().?;
    _ = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"my zebra is called quixotic zanzibar\", \"subject\": \"my zebra\"}");
    _ = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\": \"My zebra likes apples.\", \"subject\": \"my zebra\"}");
    const qs = [_][]const u8{ "What is my zebra called?", "What was my zebra called before?", "What does my zebra like?" };
    // the scratches fill with blocks that quote it, and the take-back's own words say it again
    for (qs) |q| _ = try r.store().retrieve(a, text, try text.encode(a, q, false), false, 512, &.{});
    _ = try rig.request(r.srv, a, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}", .{id}), "{\"text\": \"My zebra is not called Quixotic Zanzibar any more.\"}");
    for (qs) |q| _ = try r.store().retrieve(a, text, try text.encode(a, q, false), true, 512, &.{});
    // its own ids: in the secret texts and in nothing that stays (the other notes, the framing, the questions)
    var keep: [256]bool = @splat(false);
    for (r.store().model.notes.items) |*n| if (std.mem.indexOf(u8, n.text(), "anzibar") == null) for (n.text()) |b| {
        keep[b] = true;
    };
    for (r.store().model.notes.items) |*n| for (n.line()) |b| if (std.mem.indexOf(u8, n.text(), "anzibar") == null) {
        keep[b] = true;
    };
    for ([_][]const u8{ r.store().model.head, r.store().model.cur_title, r.store().model.hist_title, " my zebra", "[0123456789] ()\"" } ++ qs) |s| for (s) |b| {
        keep[b] = true;
    };
    var own: std.ArrayList(u32) = .empty;
    for (secret ++ "quixotic zanzibar") |b| if (!keep[b] and std.mem.indexOfScalar(u32, own.items, b) == null) try own.append(a, b);
    // and its own index keys: a key folds case, so "Q" and a kept note's "q" share one
    var own_keys: std.ArrayList(u32) = .empty;
    for (own.items) |own_id| {
        const key = r.store().model.keyOf(own_id);
        var shared = false;
        for (0..256) |b| shared = shared or (keep[b] and r.store().model.keyOf(@intCast(b)) == key);
        if (!shared and std.mem.indexOfScalar(u32, own_keys.items, key) == null) try own_keys.append(a, key);
    }
    try testing.expect(own.items.len > 0 and own_keys.items.len > 0); // so the id checks can fail
    if (kind == .sa) { // the notes it wrote are in the side buffer, not yet in a base
        const ix = r.store().current.idx;
        try testing.expect(ix.notes() > ix.base.?.notes);
    }
    const before = try look(&tracking, r, a, own.items, own_keys.items);
    try testing.expect(before.memory > 0 and before.journal and before.files > 0 and before.ids > 0); // the search finds it
    const erased = try rig.request(r.srv, a, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id}), null);
    try testing.expectEqual(@as(u16, 200), erased.status);
    try testing.expectEqual(@as(usize, 3), erased.value.get("removed").?.array.len); // its write, the same-as, the take-back
    // right after the call returns: nothing is searched for until this look
    const after = try look(&tracking, r, a, own.items, own_keys.items);
    if (deferred) {
        try r.store().merge(); // the later merge the deferred erase waited for
        return .{ .before = before, .after = after, .again = try look(&tracking, r, a, own.items, own_keys.items) };
    }
    try testing.expectEqual(@as(u32, 1), r.text.forgot); // the tokenizer dropped its kept encodes (a block quoted it)
    // what stays is what never teaching it gives: the other zebra note, current, and the questions still answer
    const left = try r.store().retrieve(a, text, try text.encode(a, qs[2], false), false, 512, &.{});
    try testing.expect(std.mem.indexOf(u8, left.block, "My zebra likes apples.") != null);
    try testing.expect(r.store().free.items.len <= 1); // the pool was swept; this lookup made the one it holds
    try r.restart();
    return .{ .before = before, .after = after, .again = try look(&tracking, r, a, own.items, own_keys.items) };
}

const none: Where = .{ .memory = 0, .journal = false, .files = 0, .ids = 0 };

test "an erase leaves the note's text in no live allocation, file or journal, and its ids in no structure" {
    const got = try eraseRun(.postings, false);
    try testing.expectEqual(none, got.after);
    try testing.expectEqual(none, got.again);
}

test "the suffix-array index: right after an erase returns, its text is in no allocation and its ids in no base" {
    const got = try eraseRun(.sa, false);
    try testing.expectEqual(none, got.after);
    try testing.expectEqual(none, got.again);
}

test "negative control: an erase that leaves the merge for later is found right after it, and gone after the merge" {
    const got = try eraseRun(.sa, true);
    try testing.expect(got.after.memory > 0 and got.after.ids > 0);
    try testing.expect(!got.after.journal and got.after.files == 0); // the files are rewritten either way
    try testing.expectEqual(none, got.again);
}
