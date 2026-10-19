//! notes-zig: the Zig lookup of the test suite's note store, from the command line. docs/NOTES-ZIG.md.
//!
//!   notes-zig lookup SNAP.json QUERIES.txt [--break subject|hop|exclude-hop|score|tokens|candidates|score-length]   one JSON line per query
//!   notes-zig bench SNAP.json QUERIES.txt [--threads 1,8] [--reps 3]   load, memory and lookup times, as JSON
//!       debug: [--fail-alloc K] [--fail-spawn N] [--legacy-barrier] [--corrupt-one]   a worker's allocations fail,
//!       starting worker N fails, (with a fault) the old start that hung, or one timed result taken as wrong: the
//!       bench must stop with an error (exit 2), not hang. Any timed lookup that differs from the first pass fails it.
//!   notes-zig info SNAP.json                                       what the snapshot holds
//!
//! A query line is "past cap id id ... [| note note ...]": past 0 or 1 (as Python's retrieve() decides it, or
//! forced), the cap in tokens, the question's token ids as the snapshot numbers them, and after a bar the note ids
//! to exclude (retrieve()'s exclude). zengine/notes_export.py writes such lines.
//! `--break` is the golden test's negative control: a deliberately broken lookup (subject, hop, exclude-hop), or a
//! right lookup that reports a wrong score, token sum or candidate count, or a score too few (score, tokens,
//! candidates, score-length).
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const notes = @import("notes");

const Query = struct { past: bool, cap: u32, ids: []const u32, exclude: []const u32 };

fn usage() noreturn {
    std.debug.print(
        \\usage: notes-zig lookup SNAP.json QUERIES.txt [--break subject|hop|exclude-hop|score|tokens|candidates|score-length]
        \\       notes-zig bench SNAP.json QUERIES.txt [--threads 1,8] [--reps 3] [--fail-alloc K] [--fail-spawn N] [--legacy-barrier] [--corrupt-one]
        \\       notes-zig info SNAP.json
        \\
    , .{});
    std.process.exit(2);
}

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("notes-zig: " ++ fmt ++ "\n", args);
    std.process.exit(2);
}

fn readQueries(gpa: std.mem.Allocator, io: Io, path: []const u8) ![]Query {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    var out: std.ArrayList(Query) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |ln| {
        if (std.mem.trim(u8, ln, " \r\t").len == 0) continue;
        const bar = std.mem.indexOfScalar(u8, ln, '|') orelse ln.len;
        var f = std.mem.tokenizeAny(u8, ln[0..bar], " \t\r");
        const past = std.fmt.parseInt(u8, f.next() orelse fail("bad query line: {s}", .{ln}), 10) catch
            fail("bad past flag in: {s}", .{ln});
        const cap = std.fmt.parseInt(u32, f.next() orelse fail("bad query line: {s}", .{ln}), 10) catch
            fail("bad cap in: {s}", .{ln});
        var ids: std.ArrayList(u32) = .empty;
        while (f.next()) |t| try ids.append(gpa, std.fmt.parseInt(u32, t, 10) catch fail("bad token id {s}", .{t}));
        var ex: std.ArrayList(u32) = .empty;
        if (bar < ln.len) {
            var x = std.mem.tokenizeAny(u8, ln[bar + 1 ..], " \t\r");
            while (x.next()) |t| try ex.append(gpa, std.fmt.parseInt(u32, t, 10) catch fail("bad note id {s}", .{t}));
        }
        try out.append(gpa, .{ .past = past != 0, .cap = cap, .ids = ids.items, .exclude = ex.items });
    }
    return out.items;
}

fn jsonString(w: *Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    var start: usize = 0;
    for (s, 0..) |c, i| {
        const esc: ?[]const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => null,
        };
        if (esc == null and c >= 0x20) continue;
        try w.writeAll(s[start..i]);
        if (esc) |e| try w.writeAll(e) else try w.print("\\u{x:0>4}", .{c});
        start = i + 1;
    }
    try w.writeAll(s[start..]);
    try w.writeByte('"');
}

fn writeResult(w: *Io.Writer, r: notes.Result) !void {
    try w.writeAll("{\"ids\":[");
    for (r.ids, 0..) |id, k| {
        if (k > 0) try w.writeByte(',');
        try w.print("{d}", .{id});
    }
    try w.writeAll("],\"scores\":[");
    for (r.scores, 0..) |s, k| {
        if (k > 0) try w.writeByte(',');
        try w.print("\"{x:0>16}\"", .{@as(u64, @bitCast(s))}); // the exact bits: no decimal round trip
    }
    try w.print("],\"tokens\":{d},\"candidates\":{d},\"block\":", .{ r.tokens, r.candidates });
    try jsonString(w, r.block);
    try w.writeAll("}\n");
}

fn digest(r: notes.Result) u64 {
    var h = std.hash.Wyhash.init(0);
    h.update(std.mem.sliceAsBytes(r.ids));
    h.update(r.block);
    return h.final();
}

fn nowNs(io: Io) i128 {
    return @intCast(Io.Clock.awake.now(io).nanoseconds);
}

fn maxRssBytes() u64 {
    const ru = std.posix.getrusage(std.posix.rusage.SELF);
    return @intCast(ru.maxrss); // bytes on macOS
}

// ------------------------------------------------------------------------------------------- bench
/// Faults injected into the bench's thread start (debug flags), to show it stops with an error instead of hanging.
const Inject = struct {
    fail_alloc: ?usize = null, // this worker's allocations fail (a FailingAllocator)
    fail_spawn: ?usize = null, // starting this worker fails, as if the system refused the thread
    legacy_barrier: bool = false, // a worker whose setup failed does not count itself, as before: the bench hangs
    corrupt_one: bool = false, // worker 0's first timed result is taken as wrong: the bench must fail
};

const Worker = struct {
    idx: *const notes.Index,
    qs: []const Query,
    ref: []const u64,
    reps: usize,
    seed: u64,
    io: Io,
    started: *std.atomic.Value(u32), // every started worker counts itself once, set up or not
    go: *std.atomic.Value(u32), // 1: run the timed lookups; 2: stop (a worker or a thread start failed)
    legacy: bool = false,
    corrupt: bool = false,
    failing: ?std.testing.FailingAllocator = null,
    sc: ?notes.Scratch = null,
    order: ?[]u32 = null,
    lat: ?[]u64 = null,
    bad: usize = 0,
    err: ?anyerror = null,

    fn allocator(self: *Worker) std.mem.Allocator {
        return if (self.failing) |*f| f.allocator() else std.heap.c_allocator;
    }

    fn run(self: *Worker) void {
        self.setup() catch |e| {
            self.err = e;
            if (!self.legacy) _ = self.started.fetchAdd(1, .acq_rel);
            return;
        };
        _ = self.started.fetchAdd(1, .acq_rel);
        while (self.go.load(.acquire) == 0) std.atomic.spinLoopHint();
        if (self.go.load(.acquire) != 1) return;
        self.timed() catch |e| {
            self.err = e;
        };
    }

    fn setup(self: *Worker) !void {
        const a = self.allocator();
        self.sc = try notes.Scratch.init(a, self.idx);
        const n = self.qs.len * self.reps;
        const order = try a.alloc(u32, n);
        self.order = order;
        for (order, 0..) |*x, k| x.* = @intCast(k % self.qs.len);
        var prng = std.Random.DefaultPrng.init(self.seed);
        prng.random().shuffle(u32, order);
        self.lat = try a.alloc(u64, n);
    }

    fn timed(self: *Worker) !void {
        const sc = &self.sc.?;
        const lat = self.lat.?;
        for (self.order.?, 0..) |q, k| {
            const t0 = nowNs(self.io);
            const r = try notes.lookup(self.idx, sc, self.qs[q].ids, self.qs[q].past, self.qs[q].cap, self.qs[q].exclude, .{});
            const t1 = nowNs(self.io);
            lat[k] = @intCast(t1 - t0);
            const d = digest(r) +% @intFromBool(self.corrupt and k == 0); // the forced-mismatch control
            if (d != self.ref[q]) self.bad += 1;
        }
    }

    fn deinit(self: *Worker) void {
        const a = self.allocator();
        if (self.sc) |*sc| sc.deinit();
        if (self.order) |x| a.free(x);
        if (self.lat) |x| a.free(x);
        self.sc = null;
        self.order = null;
        self.lat = null;
    }
};

/// Starts nt workers and runs them at once. Every started worker counts itself, whether its setup worked or not,
/// so a failed setup or a failed thread start never leaves the start waiting: on any error the started workers are
/// told to stop, all are joined and freed, and the error is returned.
fn runWorkers(gpa: std.mem.Allocator, io: Io, idx: *const notes.Index, qs: []const Query, ref: []const u64, reps: usize, nt: usize, inject: Inject, wall_ns: *i128) ![]Worker {
    var started = std.atomic.Value(u32).init(0);
    var go = std.atomic.Value(u32).init(0);
    const ws = try gpa.alloc(Worker, nt);
    errdefer gpa.free(ws);
    const hs = try gpa.alloc(std.Thread, nt);
    defer gpa.free(hs);
    for (ws, 0..) |*x, k| {
        x.* = .{ .idx = idx, .qs = qs, .ref = ref, .reps = reps, .seed = k, .io = io, .started = &started, .go = &go, .legacy = inject.legacy_barrier, .corrupt = inject.corrupt_one and k == 0 };
        if (inject.fail_alloc == k) x.failing = std.testing.FailingAllocator.init(std.heap.c_allocator, .{ .fail_index = 0 });
    }
    var spawned: usize = 0;
    var first: ?anyerror = null;
    for (hs, 0..) |*h, k| {
        if (inject.fail_spawn == k) {
            first = error.InjectedSpawnFailure;
            break;
        }
        h.* = std.Thread.spawn(.{}, Worker.run, .{&ws[k]}) catch |e| {
            first = e;
            break;
        };
        spawned += 1;
    }
    while (started.load(.acquire) < spawned) std.atomic.spinLoopHint();
    for (ws[0..spawned]) |x| {
        if (first == null) first = x.err;
    }
    if (first) |e| {
        go.store(2, .release);
        for (hs[0..spawned]) |h| h.join();
        for (ws[0..spawned]) |*x| x.deinit();
        return e;
    }
    const s0 = nowNs(io);
    go.store(1, .release);
    for (hs) |h| h.join();
    wall_ns.* = nowNs(io) - s0;
    for (ws) |x| {
        if (first == null) first = x.err;
    }
    if (first) |e| {
        for (ws) |*x| x.deinit();
        return e;
    }
    return ws;
}

fn pct(xs: []const u64, p: f64) u64 {
    // nearest rank, as notes.pct
    const n: f64 = @floatFromInt(xs.len);
    var k: i64 = @intFromFloat(@ceil(p / 100.0 * n));
    k -= 1;
    k = @max(0, @min(@as(i64, @intCast(xs.len)) - 1, k));
    return xs[@intCast(k)];
}

fn bench(gpa: std.mem.Allocator, io: Io, w: *Io.Writer, snap: []const u8, qpath: []const u8, threads: []const usize, reps: usize, inject: Inject) !void {
    const rss0 = maxRssBytes();
    const t0 = nowNs(io);
    const text = try Io.Dir.cwd().readFileAlloc(io, snap, gpa, .unlimited);
    const t1 = nowNs(io);
    const idx = try notes.Index.load(gpa, text);
    const t2 = nowNs(io);
    const file_bytes = text.len;
    gpa.free(text);
    defer idx.deinit();
    const qs = try readQueries(gpa, io, qpath);
    // one pass on one thread: the reference every timed lookup is checked against
    const ref = try gpa.alloc(u64, qs.len);
    {
        var sc = try notes.Scratch.init(gpa, idx);
        defer sc.deinit();
        for (qs, 0..) |q, k| ref[k] = digest(try notes.lookup(idx, &sc, q.ids, q.past, q.cap, q.exclude, .{}));
    }
    try w.print("{{\"zig\":\"{s}\",\"mode\":\"{s}\",\"file_bytes\":{d},\"read_ms\":{d:.3},\"load_ms\":{d:.3},\"index_used_bytes\":{d},\"index_held_bytes\":{d},\"rss_before_load\":{d},\"rss_peak\":{d},\"notes\":{d},\"current\":{d},\"history\":{d},\"grams\":{d},\"queries\":{d},\"reps\":{d},\"cpus\":{d},\"runs\":[", .{
        builtin.zig_version_string,             @tagName(builtin.mode),
        file_bytes,                             @as(f64, @floatFromInt(t1 - t0)) / 1e6,
        @as(f64, @floatFromInt(t2 - t1)) / 1e6, idx.usedBytes(),
        idx.heldBytes(),                        rss0,
        maxRssBytes(),                          idx.notes(),
        idx.n_cur,                              idx.n_hist,
        idx.grams.count(),                      qs.len,
        reps,                                   std.Thread.getCpuCount() catch 0,
    });
    var differ_total: usize = 0;
    for (threads, 0..) |nt, ti| {
        var wall_ns: i128 = 0;
        const ws = try runWorkers(gpa, io, idx, qs, ref, reps, nt, inject, &wall_ns);
        defer {
            for (ws) |*x| x.deinit();
            gpa.free(ws);
        }
        var all: std.ArrayList(u64) = .empty;
        defer all.deinit(gpa);
        var bad: usize = 0;
        for (ws) |x| {
            try all.appendSlice(gpa, x.lat.?);
            bad += x.bad;
        }
        differ_total += bad;
        std.mem.sort(u64, all.items, {}, std.sort.asc(u64));
        const wall = @as(f64, @floatFromInt(wall_ns)) / 1e9;
        var sum: u128 = 0;
        for (all.items) |x| sum += x;
        if (ti > 0) try w.writeByte(',');
        try w.print("{{\"threads\":{d},\"n\":{d},\"p50_us\":{d:.2},\"p95_us\":{d:.2},\"mean_us\":{d:.2},\"max_us\":{d:.2},\"wall_s\":{d:.4},\"per_s\":{d:.0},\"differ_from_reference\":{d}}}", .{
            nt,
            all.items.len,
            @as(f64, @floatFromInt(pct(all.items, 50))) / 1e3,
            @as(f64, @floatFromInt(pct(all.items, 95))) / 1e3,
            @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(all.items.len)) / 1e3,
            @as(f64, @floatFromInt(all.items[all.items.len - 1])) / 1e3,
            wall,
            @as(f64, @floatFromInt(all.items.len)) / wall,
            bad,
        });
    }
    try w.writeAll("]}\n");
    if (differ_total > 0) { // a timed lookup that does not match the first pass: the timings are not of right results
        try w.flush();
        std.debug.print("notes-zig: {d} timed lookups differ from the single-thread reference\n", .{differ_total});
        return error.ConcurrentResultsDiffer;
    }
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const gpa = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) usage();
    var out_buf: [1 << 16]u8 = undefined;
    var out_w: Io.File.Writer = .init(.stdout(), io, &out_buf);
    const w = &out_w.interface;
    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "info")) {
        const t0 = nowNs(io);
        const idx = notes.loadFile(gpa, io, args[2]) catch |e| fail("cannot load {s}: {t}", .{ args[2], e });
        const t1 = nowNs(io);
        defer idx.deinit();
        try w.print("{s}: {d} notes ({d} current, {d} earlier), {d} distinct n-grams, {d} vocab ids, {d} phrases; " ++
            "loaded in {d:.1} ms, index {d} bytes ({d} held)\n", .{ args[2], idx.notes(), idx.n_cur, idx.n_hist, idx.grams.count(), idx.vkey.len, idx.phrases.count(), @as(f64, @floatFromInt(t1 - t0)) / 1e6, idx.usedBytes(), idx.heldBytes() });
    } else if (std.mem.eql(u8, cmd, "lookup")) {
        if (args.len < 4) usage();
        var br: notes.Breaks = .{};
        var k: usize = 4;
        while (k < args.len) : (k += 1) {
            if (std.mem.eql(u8, args[k], "--break") and k + 1 < args.len) {
                k += 1;
                if (std.mem.eql(u8, args[k], "subject")) br.no_subject = true else if (std.mem.eql(u8, args[k], "hop"))
                    br.no_hop = true
                else if (std.mem.eql(u8, args[k], "exclude-hop"))
                    br.no_exclude_in_hop = true
                else if (std.mem.eql(u8, args[k], "score"))
                    br.nudge_score = true
                else if (std.mem.eql(u8, args[k], "tokens"))
                    br.tokens_plus_one = true
                else if (std.mem.eql(u8, args[k], "candidates"))
                    br.candidates_plus_one = true
                else if (std.mem.eql(u8, args[k], "score-length"))
                    br.drop_last_score = true
                else
                    fail("--break takes subject, hop, exclude-hop, score, tokens, candidates or score-length, not {s}", .{args[k]});
            } else usage();
        }
        const idx = notes.loadFile(gpa, io, args[2]) catch |e| fail("cannot load {s}: {t}", .{ args[2], e });
        defer idx.deinit();
        const qs = try readQueries(arena, io, args[3]);
        var sc = try notes.Scratch.init(gpa, idx);
        defer sc.deinit();
        for (qs) |q| try writeResult(w, try notes.lookup(idx, &sc, q.ids, q.past, q.cap, q.exclude, br));
    } else if (std.mem.eql(u8, cmd, "bench")) {
        if (args.len < 4) usage();
        var threads: std.ArrayList(usize) = .empty;
        var reps: usize = 3;
        var inject: Inject = .{};
        var k: usize = 4;
        while (k < args.len) : (k += 1) {
            if (std.mem.eql(u8, args[k], "--threads") and k + 1 < args.len) {
                k += 1;
                var it = std.mem.tokenizeScalar(u8, args[k], ',');
                while (it.next()) |t| try threads.append(arena, std.fmt.parseInt(usize, t, 10) catch fail("bad --threads", .{}));
            } else if (std.mem.eql(u8, args[k], "--reps") and k + 1 < args.len) {
                k += 1;
                reps = std.fmt.parseInt(usize, args[k], 10) catch fail("bad --reps", .{});
            } else if (std.mem.eql(u8, args[k], "--fail-alloc") and k + 1 < args.len) {
                k += 1;
                inject.fail_alloc = std.fmt.parseInt(usize, args[k], 10) catch fail("bad --fail-alloc", .{});
            } else if (std.mem.eql(u8, args[k], "--fail-spawn") and k + 1 < args.len) {
                k += 1;
                inject.fail_spawn = std.fmt.parseInt(usize, args[k], 10) catch fail("bad --fail-spawn", .{});
            } else if (std.mem.eql(u8, args[k], "--legacy-barrier")) {
                inject.legacy_barrier = true;
            } else if (std.mem.eql(u8, args[k], "--corrupt-one")) {
                inject.corrupt_one = true;
            } else usage();
        }
        if (threads.items.len == 0) try threads.appendSlice(arena, &.{ 1, 8 });
        bench(gpa, io, w, args[2], args[3], threads.items, reps, inject) catch |e| fail("bench failed: {t}", .{e});
    } else usage();
    try w.flush();
}
