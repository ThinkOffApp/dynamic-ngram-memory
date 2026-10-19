//! The read path of the test suite's note store (zengine/notes.py), ported to Zig. docs/NOTES-ZIG.md.
//!
//! A snapshot (zengine/notes_export.py) holds one owner's notes as Python's NoteStore keeps them. `Index.load`
//! reads it and builds the two views' n-gram postings from the notes' token ids. `lookup` takes a question's token
//! ids and returns the notes and the block that NoteStore.retrieve() returns, by the same steps in the same order:
//! the question's n-grams, IDF weights over the views searched, the subject boost, the candidate cut, the floor,
//! one hop, the ranking, pack() and render(). Python's fit() re-counts the whole block with the tokenizer and drops
//! the lowest note while it is over the cap; there is no tokenizer here, so the tokens are pack()'s own sum.
//! Writes stay in Python. An Index is read-only once loaded: any number of threads may look up at once, each with
//! a Scratch of its own.
const std = @import("std");
const Allocator = std.mem.Allocator;

const builtin = @import("builtin");

/// A log function, called through a pointer.
pub const LogFn = *const fn (f64) callconv(.c) f64;

fn zigLog(x: f64) callconv(.c) f64 {
    return @log(x);
}

/// The C library's log, the one CPython's math.log calls, looked up by name when the index loads, so every weight
/// is the same double as Python's. A plain `extern fn log` is not enough: the linker can bind it to Zig's own
/// compiler_rt copy, which differs from the C library's in the last bit for some inputs (measured: log(12/11) on
/// this Mac). Falls back to Zig's own where the C library has none.
pub fn libmLog() LogFn {
    // RTLD_DEFAULT: the special handle that searches every loaded image (-2 on Apple systems, 0 elsewhere)
    const default: ?*anyopaque = if (builtin.os.tag.isDarwin()) @ptrFromInt(@as(usize, @bitCast(@as(isize, -2)))) else null;
    if (std.c.dlsym(default, "log")) |p| return @ptrCast(@alignCast(p));
    return &zigLog;
}

/// A key that is the token id itself has this bit; a fold string's key is its place in the interned list.
pub const key_id: u32 = 0x8000_0000;
/// No key: an unused place in a packed n-gram, and the key of a token id the vocabulary lacks (it matches nothing).
const none_key: u32 = 0xFFFF_FFFF;
/// N-grams are packed into one u128, four keys of 32 bits, so n_max is at most 4.
pub const max_n = 4;
/// subjects() looks for entity phrases of 2 to 8 keys.
const max_phrase = 8;
const no_note: u32 = 0xFFFF_FFFF;

pub const State = enum(u8) { current = 0, superseded = 1, tombstoned = 2 };
pub const Hop = enum { subject, subject_value, none };

pub const Settings = struct {
    cap: u32,
    n_min: u32,
    n_max: u32,
    fold: bool,
    hop: Hop,
    hop_weight: f64,
    hop_from: u32,
    hop_history: bool,
    min_rel: f64,
    subject: f64,
    punct: bool,
};

/// Deliberate faults for the golden test's negative control: a broken port must be caught.
pub const Breaks = struct {
    no_subject: bool = false, // skip the subject boost
    no_hop: bool = false, // skip the hop
    no_exclude_in_hop: bool = false, // let the hop bring in an excluded note
    // faults in what is reported only (the notes and the block stay right), for the controls of the golden test's
    // other checks: the scores, the tokens and the candidate count
    nudge_score: bool = false, // the first note's score one ulp up
    tokens_plus_one: bool = false, // pack()'s sum one too many
    candidates_plus_one: bool = false, // one candidate too many
    drop_last_score: bool = false, // one score fewer than notes
};

pub const LoadError = error{ BadSnapshot, Unsupported } || Allocator.Error || std.json.ParseError(std.json.Scanner);

// ------------------------------------------------------------------------------------------- the snapshot, as JSON
const JNote = struct {
    id: u32,
    state: u8,
    kind: u8,
    entity: u32,
    attribute: u32,
    pair: u32,
    value_ent: i32,
    told: i64,
    line: []const u8,
    cost: u32,
    tokens: []const u32,
};
const JEnt = struct { name: []const u8, cur: []const u32, keys: []const u32 };
const JPair = struct { entity: u32, attribute: u32, ids: []const u32 };
const JPhrase = struct { keys: []const u32, entity: u32 };
const JSettings = struct {
    cap: u32,
    n_min: u32,
    n_max: u32,
    fold: bool,
    hop: []const u8,
    hop_weight: f64,
    hop_from: u32,
    hop_history: bool,
    min_rel: f64,
    subject: f64,
    punct: bool,
    takeback: []const u8,
    unknown_rule: bool,
};
const JFixed = struct { head: u32, current: u32, history: u32, nl: u32 };
const JVocab = struct { size: u32, key: []const u32, wordy: []const u8 };
const JCounts = struct { current: u32, history: u32 };
const JSnap = struct {
    format: []const u8,
    settings: JSettings,
    head: []const u8,
    cur_title: []const u8,
    hist_title: []const u8,
    fixed: JFixed,
    vocab: JVocab,
    entities: []const JEnt,
    pairs: []const JPair,
    phrases: []const JPhrase,
    notes: []const JNote,
    n: JCounts,
    grams: JCounts,
};

// ------------------------------------------------------------------------------------------- n-grams and phrases
pub const Gram = u128;

/// Keys packed into one u128; the unused places hold none_key, so (a, b) and (a, b, c) never meet.
pub fn packGram(keys: []const u32) Gram {
    var g: Gram = 0;
    var j: usize = 0;
    while (j < max_n) : (j += 1) {
        const k: u32 = if (j < keys.len) keys[j] else none_key;
        g |= @as(Gram, k) << @intCast(32 * j);
    }
    return g;
}

const GramCtx = struct {
    pub fn hash(_: GramCtx, g: Gram) u64 {
        const lo: u64 = @truncate(g);
        const hi: u64 = @truncate(g >> 64);
        const m = @as(u128, lo ^ 0xa076_1d64_78bd_642f) *% @as(u128, hi ^ 0xe703_7ed1_a0b4_28db);
        return @as(u64, @truncate(m)) ^ @as(u64, @truncate(m >> 64));
    }
    pub fn eql(_: GramCtx, a: Gram, b: Gram) bool {
        return a == b;
    }
};

const Phrase = [max_phrase]u32;
const PhraseCtx = struct {
    pub fn hash(_: PhraseCtx, p: Phrase) u64 {
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&p));
    }
    pub fn eql(_: PhraseCtx, a: Phrase, b: Phrase) bool {
        return std.mem.eql(u32, &a, &b);
    }
};

fn packPhrase(keys: []const u32) Phrase {
    var p: Phrase = @splat(none_key);
    @memcpy(p[0..keys.len], keys);
    return p;
}

/// Postings of one n-gram: note places in the current view, then in the history view.
const Post = struct { cur_off: u32, cur_len: u32, hist_off: u32, hist_len: u32 };

// ------------------------------------------------------------------------------------------- the index
pub const Index = struct {
    arena: std.heap.ArenaAllocator,
    log: LogFn,
    settings: Settings,
    head: []const u8,
    cur_title: []const u8,
    hist_title: []const u8,
    fx_head: u32,
    fx_cur: u32,
    fx_hist: u32,
    fx_nl: u32,
    vkey: []const u32,
    vwordy: []const bool,
    // the notes, by place (0..notes-1); ids are Python's note ids
    note_id: []const u32,
    state: []const State,
    told: []const i64,
    trank: []const u32, // place in the order (-told, id): Python's tie-break after the score
    entity: []const u32,
    pair: []const u32,
    value_ent: []const i32,
    line: []const []const u8,
    cost: []const u32,
    min_line: i64, // the least any line can add in pack(): its cost and a line break
    id_place: []const u32, // a note id -> its place, or no_note
    ent_cur: []const []const u32, // entity -> places of its current notes (O.ent_cur)
    ent_keys: []const []const u32, // entity -> its pairs (O.ent_keys)
    pair_notes: []const []const u32, // pair -> places of its notes, every state, oldest first (O.by_key)
    grams: std.HashMapUnmanaged(Gram, Post, GramCtx, 80),
    post: []const u32,
    phrases: std.HashMapUnmanaged(Phrase, u32, PhraseCtx, 80),
    phrase_max: usize,
    n_cur: u32,
    n_hist: u32,

    pub fn notes(self: *const Index) usize {
        return self.note_id.len;
    }

    /// A note id's place in the index, if the snapshot has that note.
    pub fn placeOfId(self: *const Index, id: u32) ?u32 {
        if (id >= self.id_place.len or self.id_place[id] == no_note) return null;
        return self.id_place[id];
    }

    /// Bytes the index's arena has reserved once loaded (the parsed JSON is freed by then): what it holds, with the
    /// arena's room to grow (pages not yet written are not resident).
    pub fn heldBytes(self: *const Index) usize {
        return self.arena.queryCapacity();
    }

    /// Bytes of the index's own structures, counted array by array (the hash maps at their capacity).
    pub fn usedBytes(self: *const Index) usize {
        var n: usize = self.head.len + self.cur_title.len + self.hist_title.len;
        n += self.vkey.len * @sizeOf(u32) + self.vwordy.len * @sizeOf(bool);
        n += self.note_id.len * (@sizeOf(u32) + @sizeOf(State) + @sizeOf(i64) + @sizeOf(u32) * 5 + @sizeOf(i32) + @sizeOf([]const u8));
        for (self.line) |l| n += l.len;
        inline for (.{ "ent_cur", "ent_keys", "pair_notes" }) |f| {
            for (@field(self, f)) |xs| n += @sizeOf([]const u32) + xs.len * @sizeOf(u32);
        }
        n += self.post.len * @sizeOf(u32) + self.id_place.len * @sizeOf(u32);
        n += self.grams.capacity() * (@sizeOf(Gram) + @sizeOf(Post) + 1);
        n += self.phrases.capacity() * (@sizeOf(Phrase) + @sizeOf(u32) + 1);
        return n;
    }

    pub fn deinit(self: *Index) void {
        const gpa = self.arena.child_allocator;
        self.arena.deinit();
        gpa.destroy(self);
    }

    /// Reads a snapshot (the JSON text of notes_export.py) and builds the index. The snapshot's own counts are
    /// checked: notes per view, and distinct n-grams per view against the store's own index.
    pub fn load(gpa: Allocator, text: []const u8) LoadError!*Index {
        var parse_arena = std.heap.ArenaAllocator.init(gpa);
        defer parse_arena.deinit();
        const s = try std.json.parseFromSliceLeaky(JSnap, parse_arena.allocator(), text, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_if_needed,
        });
        if (!std.mem.eql(u8, s.format, "lw-notes-snapshot/1")) return error.BadSnapshot;

        const self = try gpa.create(Index);
        errdefer gpa.destroy(self);
        self.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.log = libmLog();

        const js = s.settings;
        if (js.n_min < 1 or js.n_min > js.n_max) return error.BadSnapshot;
        if (js.n_max > max_n) return error.Unsupported;
        self.settings = .{
            .cap = js.cap,
            .n_min = js.n_min,
            .n_max = js.n_max,
            .fold = js.fold,
            .hop = if (std.mem.eql(u8, js.hop, "subject")) .subject else if (std.mem.eql(u8, js.hop, "subject+value"))
                .subject_value
            else if (std.mem.eql(u8, js.hop, "none")) .none else return error.BadSnapshot,
            .hop_weight = js.hop_weight,
            .hop_from = js.hop_from,
            .hop_history = js.hop_history,
            .min_rel = js.min_rel,
            .subject = js.subject,
            .punct = js.punct,
        };
        self.head = try a.dupe(u8, s.head);
        self.cur_title = try a.dupe(u8, s.cur_title);
        self.hist_title = try a.dupe(u8, s.hist_title);
        self.fx_head = s.fixed.head;
        self.fx_cur = s.fixed.current;
        self.fx_hist = s.fixed.history;
        self.fx_nl = s.fixed.nl;

        // the vocabulary: every token id's key and wordy flag
        if (s.vocab.key.len != s.vocab.size or s.vocab.wordy.len != s.vocab.size) return error.BadSnapshot;
        self.vkey = try a.dupe(u32, s.vocab.key);
        const vw = try a.alloc(bool, s.vocab.size);
        for (s.vocab.wordy, 0..) |c, i| vw[i] = c == '1';
        self.vwordy = vw;

        // the notes
        const n = s.notes.len;
        const note_id = try a.alloc(u32, n);
        const state = try a.alloc(State, n);
        const told = try a.alloc(i64, n);
        const entity = try a.alloc(u32, n);
        const pair = try a.alloc(u32, n);
        const value_ent = try a.alloc(i32, n);
        const line = try a.alloc([]const u8, n);
        const cost = try a.alloc(u32, n);
        var max_id: u32 = 0;
        var line_bytes: usize = 0;
        for (s.notes) |x| {
            max_id = @max(max_id, x.id);
            line_bytes += x.line.len;
        }
        const id_place = try a.alloc(u32, @as(usize, max_id) + 1);
        @memset(id_place, no_note);
        self.id_place = id_place;
        const text_buf = try a.alloc(u8, line_bytes);
        var at: usize = 0;
        var n_cur: u32 = 0;
        var n_hist: u32 = 0;
        for (s.notes, 0..) |x, k| {
            if (id_place[x.id] != no_note) return error.BadSnapshot;
            id_place[x.id] = @intCast(k);
            note_id[k] = x.id;
            state[k] = std.enums.fromInt(State, x.state) orelse return error.BadSnapshot;
            if (state[k] == .current) n_cur += 1 else n_hist += 1;
            told[k] = x.told;
            if (x.entity >= s.entities.len or x.pair >= s.pairs.len) return error.BadSnapshot;
            entity[k] = x.entity;
            pair[k] = x.pair;
            if (x.value_ent >= @as(i32, @intCast(s.entities.len))) return error.BadSnapshot;
            value_ent[k] = x.value_ent;
            @memcpy(text_buf[at..][0..x.line.len], x.line);
            line[k] = text_buf[at..][0..x.line.len];
            at += x.line.len;
            cost[k] = x.cost;
        }
        if (n_cur != s.n.current or n_hist != s.n.history) return error.BadSnapshot;
        const by = try gpa.alloc(u32, n);
        defer gpa.free(by);
        for (by, 0..) |*x, k| x.* = @intCast(k);
        const TieOrder = struct {
            told: []const i64,
            id: []const u32,
            fn less(c: @This(), x: u32, y: u32) bool {
                if (c.told[x] != c.told[y]) return c.told[x] > c.told[y];
                return c.id[x] < c.id[y];
            }
        };
        std.sort.pdq(u32, by, TieOrder{ .told = told, .id = note_id }, TieOrder.less);
        const trank = try a.alloc(u32, n);
        for (by, 0..) |x, r| trank[x] = @intCast(r);
        self.trank = trank;
        self.note_id = note_id;
        self.state = state;
        self.told = told;
        self.entity = entity;
        self.pair = pair;
        self.value_ent = value_ent;
        self.line = line;
        self.cost = cost;
        var least: i64 = std.math.maxInt(i64);
        for (cost) |c| least = @min(least, @as(i64, c) + s.fixed.nl);
        self.min_line = least;
        self.n_cur = n_cur;
        self.n_hist = n_hist;

        const places = struct {
            fn of(al: Allocator, ids: []const u32, idp: []const u32) LoadError![]const u32 {
                const out = try al.alloc(u32, ids.len);
                for (ids, 0..) |id, j| {
                    if (id >= idp.len or idp[id] == no_note) return error.BadSnapshot;
                    out[j] = idp[id];
                }
                return out;
            }
        };
        const ent_cur = try a.alloc([]const u32, s.entities.len);
        const ent_keys = try a.alloc([]const u32, s.entities.len);
        for (s.entities, 0..) |e, k| {
            ent_cur[k] = try places.of(a, e.cur, id_place);
            for (e.keys) |p| if (p >= s.pairs.len) return error.BadSnapshot;
            ent_keys[k] = try a.dupe(u32, e.keys);
        }
        const pair_notes = try a.alloc([]const u32, s.pairs.len);
        for (s.pairs, 0..) |p, k| pair_notes[k] = try places.of(a, p.ids, id_place);
        self.ent_cur = ent_cur;
        self.ent_keys = ent_keys;
        self.pair_notes = pair_notes;

        // phrases: only 2 to 8 keys can ever match (subjects() looks at those lengths)
        self.phrases = .empty;
        self.phrase_max = 0;
        for (s.phrases) |p| {
            if (p.entity >= s.entities.len) return error.BadSnapshot;
            if (p.keys.len < 2 or p.keys.len > max_phrase) continue;
            try self.phrases.put(a, packPhrase(p.keys), p.entity);
            self.phrase_max = @max(self.phrase_max, p.keys.len);
        }

        try self.buildPostings(gpa, s.notes);
        var gc: u32 = 0;
        var gh: u32 = 0;
        var it = self.grams.valueIterator();
        while (it.next()) |p| {
            if (p.cur_len > 0) gc += 1;
            if (p.hist_len > 0) gh += 1;
        }
        if (gc != s.grams.current or gh != s.grams.history) return error.BadSnapshot;
        return self;
    }

    /// The postings of both views, from each note's token ids by the same grams() as a question's.
    fn buildPostings(self: *Index, gpa: Allocator, jn: []const JNote) LoadError!void {
        const a = self.arena.allocator();
        const Hit = struct { entry: u32, note: u32 };
        var hits: std.ArrayList(Hit) = .empty;
        defer hits.deinit(gpa);
        var keys: std.ArrayList(u32) = .empty;
        defer keys.deinit(gpa);
        var gl: std.ArrayList(Gram) = .empty;
        defer gl.deinit(gpa);
        var entries: std.ArrayList(Post) = .empty; // counts first, then offsets
        defer entries.deinit(gpa);
        var map: std.HashMapUnmanaged(Gram, u32, GramCtx, 80) = .empty;
        defer map.deinit(gpa);
        for (jn, 0..) |x, k| {
            keys.clearRetainingCapacity();
            for (x.tokens) |t| try keys.append(gpa, self.keyOf(t));
            gl.clearRetainingCapacity();
            try self.gramsOf(gpa, x.tokens, keys.items, &gl);
            for (gl.items) |g| {
                const r = try map.getOrPut(gpa, g);
                if (!r.found_existing) {
                    r.value_ptr.* = @intCast(entries.items.len);
                    try entries.append(gpa, .{ .cur_off = 0, .cur_len = 0, .hist_off = 0, .hist_len = 0 });
                }
                const e = &entries.items[r.value_ptr.*];
                if (self.state[k] == .current) e.cur_len += 1 else e.hist_len += 1;
                try hits.append(gpa, .{ .entry = r.value_ptr.*, .note = @intCast(k) });
            }
        }
        var off: u32 = 0;
        for (entries.items) |*e| {
            e.cur_off = off;
            off += e.cur_len;
            e.hist_off = off;
            off += e.hist_len;
        }
        const post = try a.alloc(u32, off);
        const fill = try gpa.alloc(u32, entries.items.len * 2);
        defer gpa.free(fill);
        @memset(fill, 0);
        for (hits.items) |h| {
            const e = entries.items[h.entry];
            if (self.state[h.note] == .current) {
                post[e.cur_off + fill[2 * h.entry]] = h.note;
                fill[2 * h.entry] += 1;
            } else {
                post[e.hist_off + fill[2 * h.entry + 1]] = h.note;
                fill[2 * h.entry + 1] += 1;
            }
        }
        self.post = post;
        self.grams = .empty;
        try self.grams.ensureTotalCapacity(a, @intCast(entries.items.len));
        var it = map.iterator();
        while (it.next()) |kv| self.grams.putAssumeCapacity(kv.key_ptr.*, entries.items[kv.value_ptr.*]);
    }

    inline fn keyOf(self: *const Index, t: u32) u32 {
        return if (t < self.vkey.len) self.vkey[t] else none_key;
    }

    inline fn wordyOf(self: *const Index, t: u32) bool {
        return t < self.vwordy.len and self.vwordy[t];
    }

    /// NoteStore.grams: n from n_min to n_max, every start; a unigram of punctuation alone is no key unless
    /// `punct`. Distinct grams, in the order first met (Python keeps a set; see lookup on the order of sums).
    fn gramsOf(self: *const Index, gpa: Allocator, ids: []const u32, keys: []const u32, out: *std.ArrayList(Gram)) Allocator.Error!void {
        const st = self.settings;
        var n: usize = st.n_min;
        while (n <= st.n_max) : (n += 1) {
            if (keys.len < n) break;
            var j: usize = 0;
            while (j + n <= keys.len) : (j += 1) {
                if (n == 1 and !st.punct and !self.wordyOf(ids[j])) continue;
                const g = packGram(keys[j..][0..n]);
                if (std.mem.indexOfScalar(Gram, out.items, g) == null) try out.append(gpa, g);
            }
        }
    }
};

/// Reads a snapshot file and loads it.
pub fn loadFile(gpa: Allocator, io: std.Io, path: []const u8) !*Index {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(text);
    return Index.load(gpa, text);
}

// ------------------------------------------------------------------------------------------- one lookup
/// One note's per-lookup state, kept together so a score update touches one cache line.
const Slot = struct {
    score: f64,
    stamp: u32, // == gen: the note has a score (Python: i in score)
    pstamp: u32, // == gen: the note is in the pool
};

/// What one thread needs for its lookups: a slot per note, reset by a generation number rather than cleared, and
/// the buffers of the result. The lists a note enters at most once are sized to the store up front.
pub const Scratch = struct {
    gpa: Allocator,
    slot: []Slot,
    xstamp: []u32, // == gen: the note is excluded from this lookup (apart from slot: read only when some are)
    gen: u32,
    touched: std.ArrayList(u32) = .empty,
    pool: std.ArrayList(u32) = .empty,
    keys: std.ArrayList(u128) = .empty,
    qkeys: std.ArrayList(u32) = .empty,
    qgrams: std.ArrayList(Gram) = .empty,
    found: std.ArrayList(u32) = .empty,
    spans: std.ArrayList([2]usize) = .empty,
    chosen: std.ArrayList(u32) = .empty,
    order: std.ArrayList([2]u32) = .empty,
    ids: std.ArrayList(u32) = .empty,
    scores: std.ArrayList(f64) = .empty,
    block: std.ArrayList(u8) = .empty,

    const lists = .{ "touched", "pool", "keys", "qkeys", "qgrams", "found", "spans", "chosen", "order", "ids", "scores", "block" };

    pub fn init(gpa: Allocator, idx: *const Index) Allocator.Error!Scratch {
        const n = idx.notes();
        const slot = try gpa.alloc(Slot, n);
        errdefer gpa.free(slot);
        const xstamp = try gpa.alloc(u32, n);
        errdefer gpa.free(xstamp);
        var sc: Scratch = .{ .gpa = gpa, .slot = slot, .xstamp = xstamp, .gen = 0 };
        errdefer inline for (lists) |f| @field(sc, f).deinit(gpa);
        @memset(sc.slot, .{ .score = 0, .stamp = 0, .pstamp = 0 });
        @memset(sc.xstamp, 0);
        try sc.touched.ensureTotalCapacity(gpa, n);
        try sc.pool.ensureTotalCapacity(gpa, n);
        try sc.keys.ensureTotalCapacity(gpa, n);
        return sc;
    }

    pub fn deinit(sc: *Scratch) void {
        sc.gpa.free(sc.slot);
        sc.gpa.free(sc.xstamp);
        inline for (lists) |f| @field(sc, f).deinit(sc.gpa);
    }

    fn next(sc: *Scratch) void {
        sc.gen +%= 1;
        if (sc.gen == 0) {
            @memset(sc.slot, .{ .score = 0, .stamp = 0, .pstamp = 0 });
            @memset(sc.xstamp, 0);
            sc.gen = 1;
        }
        inline for (lists) |f| @field(sc, f).clearRetainingCapacity();
    }

    /// score[i] = score.get(i, 0.0) + w
    inline fn add(sc: *Scratch, i: u32, w: f64) void {
        const x = &sc.slot[i];
        if (x.stamp != sc.gen) {
            x.stamp = sc.gen;
            x.score = 0.0;
            sc.touched.appendAssumeCapacity(i);
        }
        x.score = x.score + w;
    }

    inline fn get(sc: *const Scratch, i: u32) f64 {
        return if (sc.slot[i].stamp == sc.gen) sc.slot[i].score else 0.0;
    }

    inline fn toPool(sc: *Scratch, i: u32) void {
        if (sc.slot[i].pstamp != sc.gen) {
            sc.slot[i].pstamp = sc.gen;
            sc.pool.appendAssumeCapacity(i);
        }
    }
};

pub const Result = struct {
    ids: []const u32, // Python's note ids: the current ones, then the earlier ones, each by told (render order)
    scores: []const f64, // each one's score, for diagnostics
    block: []const u8, // "" when nothing is retrieved
    tokens: u32, // pack()'s sum: the head, the titles, the lines and the line breaks
    candidates: u32,
};

/// Python's order key (-score, -told, id) as one u128 that sorts ascending: the score's bits turned to sort
/// descending, then the note's place in the order (-told, id), which is fixed and so computed once at load, then
/// the note's place itself (to get it back). A float's bits sort as the float does once the sign is folded in.
pub fn orderKey(score: f64, trank: u32, place: u32) u128 {
    const b: u64 = @bitCast(score);
    const up: u64 = if (b >> 63 != 0) ~b else b | (@as(u64, 1) << 63); // ascending with the score
    return (@as(u128, ~up) << 64) | (@as(u128, trank) << 32) | place;
}

inline fn placeOf(k: u128) u32 {
    return @truncate(k);
}

/// Puts the k smallest keys first (in no order), as nth_element does: keys are distinct, so the k first after
/// sorting them are exactly sorted(all)[:k], which is heapq.nsmallest(k, ...) in Python.
fn selectFirst(ks: []u128, k: usize) void {
    if (k >= ks.len) return;
    var lo: usize = 0;
    var hi: usize = ks.len - 1;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2; // median of three, moved to hi
        if (ks[mid] < ks[lo]) std.mem.swap(u128, &ks[mid], &ks[lo]);
        if (ks[hi] < ks[lo]) std.mem.swap(u128, &ks[hi], &ks[lo]);
        if (ks[mid] < ks[hi]) std.mem.swap(u128, &ks[mid], &ks[hi]);
        const p = ks[hi];
        var store = lo;
        var i = lo;
        while (i < hi) : (i += 1) { // Lomuto, without a branch on the comparison
            const x = ks[i];
            ks[i] = ks[store];
            ks[store] = x;
            store += @intFromBool(x < p);
        }
        std.mem.swap(u128, &ks[store], &ks[hi]);
        if (store == k) return;
        if (store < k) lo = store + 1 else hi = store - 1;
    }
}

/// NoteStore.subjects: the entities whose phrase the question's keys contain, longest first, none inside another
/// found one (a list: a phrase found twice counts twice, as in Python).
fn subjects(idx: *const Index, sc: *Scratch) Allocator.Error!void {
    const q = sc.qkeys.items;
    if (idx.phrases.count() == 0) return;
    var n: usize = @min(q.len, max_phrase, idx.phrase_max);
    while (n >= 2) : (n -= 1) {
        var j: usize = 0;
        while (j + n <= q.len) : (j += 1) {
            const e = idx.phrases.get(packPhrase(q[j..][0..n])) orelse continue;
            var inside = false;
            for (sc.spans.items) |sp| {
                if (sp[0] <= j and j + n <= sp[1]) {
                    inside = true;
                    break;
                }
            }
            if (inside) continue;
            try sc.found.append(sc.gpa, e);
            try sc.spans.append(sc.gpa, .{ j, j + n });
        }
    }
}

/// One question's notes and block: NoteStore.retrieve(owner, question, cap, past, exclude) from the question's token
/// ids. `exclude`: note ids never given, neither scored nor brought in by a hop (promotion keeps a note whose fact
/// the neurons answer out of the prompt); as in Python, they still count in the weights' document frequencies, their
/// scores go after the subject boost (before the candidates are counted), and a hop skips them. Ids the snapshot
/// lacks are ignored. The result's slices live in `sc` until its next lookup.
///
/// Sums of floats: Python adds a note's n-gram weights in the order it iterates the question's set of n-grams,
/// which depends on string hashing; here they are added in the order the n-grams are first met. Notes with the same
/// matched n-grams get the same sum either way; a tie between different n-gram sets can come out differently in
/// the last bit (the golden test reports any such case).
pub fn lookup(idx: *const Index, sc: *Scratch, qids: []const u32, past: bool, cap: u32, exclude: []const u32, br: Breaks) Allocator.Error!Result {
    var r = try lookupNotes(idx, sc, qids, past, cap, exclude, br);
    if (br.nudge_score and r.ids.len > 0) sc.scores.items[0] = std.math.nextAfter(f64, sc.scores.items[0], std.math.inf(f64));
    if (br.tokens_plus_one) r.tokens += 1;
    if (br.candidates_plus_one) r.candidates += 1;
    if (br.drop_last_score and r.scores.len > 0) r.scores = r.scores[0 .. r.scores.len - 1];
    return r;
}

fn lookupNotes(idx: *const Index, sc: *Scratch, qids: []const u32, past: bool, cap: u32, exclude: []const u32, br: Breaks) Allocator.Error!Result {
    const st = idx.settings;
    const gpa = sc.gpa;
    sc.next();
    var excluding = false;
    for (exclude) |id| {
        if (idx.placeOfId(id)) |p| {
            sc.xstamp[p] = sc.gen;
            excluding = true;
        }
    }
    for (qids) |t| try sc.qkeys.append(gpa, idx.keyOf(t));
    try idx.gramsOf(gpa, qids, sc.qkeys.items, &sc.qgrams);

    // IDF over the views searched: the current notes, and for a question about the past the earlier ones too
    const N: u32 = idx.n_cur + (if (past) idx.n_hist else 0);
    for (sc.qgrams.items) |g| {
        const p = idx.grams.get(g) orelse continue;
        const df: u32 = p.cur_len + (if (past) p.hist_len else 0);
        if (df == 0) continue;
        const w = idx.log(1.0 + (@as(f64, @floatFromInt(N - df)) + 0.5) / (@as(f64, @floatFromInt(df)) + 0.5));
        for (idx.post[p.cur_off..][0..p.cur_len]) |i| sc.add(i, w);
        if (past) for (idx.post[p.hist_off..][0..p.hist_len]) |i| sc.add(i, w);
    }
    // the subject boost: the top IDF for each note of an entity the question names
    if (st.subject > 0 and N > 0 and !br.no_subject) {
        const top = idx.log(1.0 + (@as(f64, @floatFromInt(N)) - 0.5) / 1.5) * st.subject;
        try subjects(idx, sc);
        for (sc.found.items) |e| {
            for (idx.ent_cur[e]) |i| sc.add(i, top);
            if (past) for (idx.ent_keys[e]) |k| for (idx.pair_notes[k]) |i| sc.add(i, top);
        }
    }
    if (excluding) { // for i in exclude: score.pop(i, None)
        var w: usize = 0;
        for (sc.touched.items) |i| {
            if (sc.xstamp[i] == sc.gen) {
                sc.slot[i].stamp = 0; // gen is never 0: the note has no score now
            } else {
                sc.touched.items[w] = i;
                w += 1;
            }
        }
        sc.touched.shrinkRetainingCapacity(w);
    }
    const cands: u32 = @intCast(sc.touched.items.len);
    if (cands == 0) return .{ .ids = &.{}, .scores = &.{}, .block = "", .tokens = 0, .candidates = 0 };

    // the candidate cut: no more lines than this fit, at 4 tokens a line or more; the 4*room best when there are
    // more than 8*room candidates, else all of them
    const room: usize = @intCast(@max(1, @divFloor(@as(i64, cap) - @as(i64, idx.fx_head), 4)));
    for (sc.touched.items) |i| sc.keys.appendAssumeCapacity(orderKey(sc.slot[i].score, idx.trank[i], i));
    var direct = sc.keys.items;
    if (direct.len > 8 * room) {
        selectFirst(direct, 4 * room);
        direct = direct[0 .. 4 * room];
    }
    // Python sorts `direct` whole, but reads its order only for the best one (the floor) and the first hop_from (the
    // hops); the rest joins the pool as a set. So only those first ones are put in order.
    const head_n = @min(direct.len, @max(st.hop_from, 1));
    selectFirst(direct, head_n);
    std.sort.pdq(u128, direct[0..head_n], {}, std.sort.asc(u128));
    if (st.min_rel > 0) {
        // the kept ones keep their order, so the sorted head stays first (a head note under the floor means every
        // note after it is too)
        const floor = sc.slot[placeOf(direct[0])].score * st.min_rel;
        var w: usize = 0;
        for (direct) |k| {
            if (sc.slot[placeOf(k)].score >= floor) {
                direct[w] = k;
                w += 1;
            }
        }
        direct = direct[0..w];
    }
    for (direct) |k| sc.toPool(placeOf(k));
    // one hop from the best direct matches; a hop note scores hop_weight of the note that brought it in
    if (!br.no_hop) {
        for (direct[0..@min(direct.len, st.hop_from)]) |k| {
            const i = placeOf(k);
            const s = sc.slot[i].score * st.hop_weight;
            var near: [3][]const u32 = undefined;
            var nn: usize = 0;
            if (st.hop != .none) {
                near[nn] = idx.ent_cur[idx.entity[i]];
                nn += 1;
                if (st.hop == .subject_value and idx.value_ent[i] >= 0) {
                    near[nn] = idx.ent_cur[@intCast(idx.value_ent[i])];
                    nn += 1;
                }
            }
            if (past and st.hop_history) {
                near[nn] = idx.pair_notes[idx.pair[i]];
                nn += 1;
            }
            const skip = excluding and !br.no_exclude_in_hop;
            for (near[0..nn]) |js| for (js) |j| {
                if (skip and sc.xstamp[j] == sc.gen) continue;
                if (s > sc.get(j)) {
                    sc.slot[j].stamp = sc.gen;
                    sc.slot[j].score = s;
                }
                sc.toPool(j);
            };
        }
    }

    // pack(): the best notes whose lines fit under the cap with the framing, best first. The pool is put in order a
    // part at a time, and the walk stops once no line could fit in what is left: the same notes as Python's walk
    // over the whole sorted pool, which only skips from there on.
    sc.keys.clearRetainingCapacity();
    for (sc.pool.items) |i| sc.keys.appendAssumeCapacity(orderKey(sc.slot[i].score, idx.trank[i], i));
    var used: u32 = idx.fx_head;
    var has_cur = false;
    var has_hist = false;
    var rest = sc.keys.items;
    var first = true;
    walk: while (rest.len > 0) {
        // the first 64 by selection; if the walk goes on, the rest sorted whole
        const m = if (first) @min(rest.len, 64) else rest.len;
        first = false;
        selectFirst(rest, m);
        std.sort.pdq(u128, rest[0..m], {}, std.sort.asc(u128));
        for (rest[0..m]) |k| {
            if (@as(i64, cap) - @as(i64, used) < idx.min_line) break :walk;
            const i = placeOf(k);
            const cur = idx.state[i] == .current;
            if (!cur and !past) continue;
            const titled = if (cur) has_cur else has_hist;
            const c = idx.cost[i] + idx.fx_nl + (if (titled) 0 else (if (cur) idx.fx_cur else idx.fx_hist) + idx.fx_nl);
            if (@as(u64, used) + c <= cap) {
                used += c;
                try sc.chosen.append(gpa, i);
                if (cur) has_cur = true else has_hist = true;
            }
        }
        rest = rest[m..];
    }
    if (sc.chosen.items.len == 0) return .{ .ids = &.{}, .scores = &.{}, .block = "", .tokens = 0, .candidates = cands };

    // render(): the head, then the current notes and the earlier ones, each by told (stable: ties keep pack order)
    for (sc.chosen.items, 0..) |i, k| try sc.order.append(gpa, .{ i, @intCast(k) });
    const ByTold = struct {
        told: []const i64,
        state: []const State,
        fn less(c: @This(), a: [2]u32, b: [2]u32) bool {
            const ca = c.state[a[0]] == .current;
            const cb = c.state[b[0]] == .current;
            if (ca != cb) return ca;
            if (c.told[a[0]] != c.told[b[0]]) return c.told[a[0]] < c.told[b[0]];
            return a[1] < b[1];
        }
    };
    std.sort.pdq([2]u32, sc.order.items, ByTold{ .told = idx.told, .state = idx.state }, ByTold.less);
    try sc.block.appendSlice(gpa, idx.head);
    var in_cur = false;
    var in_hist = false;
    for (sc.order.items) |x| {
        const i = x[0];
        if (idx.state[i] == .current) {
            if (!in_cur) {
                try sc.block.append(gpa, '\n');
                try sc.block.appendSlice(gpa, idx.cur_title);
                in_cur = true;
            }
        } else if (!in_hist) {
            try sc.block.append(gpa, '\n');
            try sc.block.appendSlice(gpa, idx.hist_title);
            in_hist = true;
        }
        try sc.block.append(gpa, '\n');
        try sc.block.appendSlice(gpa, idx.line[i]);
        try sc.ids.append(gpa, idx.note_id[i]);
        try sc.scores.append(gpa, sc.slot[i].score);
    }
    return .{ .ids = sc.ids.items, .scores = sc.scores.items, .block = sc.block.items, .tokens = used, .candidates = cands };
}

test "packGram keeps lengths apart" {
    const a = packGram(&.{ 1, 2 });
    const b = packGram(&.{ 1, 2, none_key });
    const c = packGram(&.{ 1, 2, 3 });
    try std.testing.expect(a == b); // none_key is never a real key, so this pair cannot occur in a snapshot
    try std.testing.expect(a != c);
    try std.testing.expect(packGram(&.{1}) != packGram(&.{ 1, 0 }));
}

test "the order key sorts as Python's (-score, -told, id), and selectFirst as sorted()[:k]" {
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    const n = 500;
    var score: [n]f64 = undefined;
    var told: [n]i64 = undefined;
    var keys: [n]u128 = undefined;
    var places: [n]u32 = undefined;
    for (0..n) |i| {
        score[i] = @as(f64, @floatFromInt(r.uintLessThan(u32, 20))) * 0.37; // many ties on score
        told[i] = @intCast(r.uintLessThan(u32, 5)); // and on told
        places[i] = @intCast(i);
    }
    // trank as Index.load makes it: (-told, id), with id = place + 1
    const T = struct {
        told: []const i64,
        fn less(c: @This(), x: u32, y: u32) bool {
            if (c.told[x] != c.told[y]) return c.told[x] > c.told[y];
            return x < y;
        }
    };
    var by = places;
    std.sort.pdq(u32, &by, T{ .told = &told }, T.less);
    var trank: [n]u32 = undefined;
    for (by, 0..) |x, k| trank[x] = @intCast(k);
    for (0..n) |i| keys[i] = orderKey(score[i], trank[i], @intCast(i));
    const P = struct {
        score: []const f64,
        told: []const i64,
        fn less(c: @This(), a: u32, b: u32) bool {
            if (c.score[a] != c.score[b]) return c.score[a] > c.score[b];
            if (c.told[a] != c.told[b]) return c.told[a] > c.told[b];
            return a < b;
        }
    };
    std.sort.pdq(u32, &places, P{ .score = &score, .told = &told }, P.less);
    var all = keys;
    std.sort.pdq(u128, &all, {}, std.sort.asc(u128));
    for (all, 0..) |k, i| try std.testing.expectEqual(places[i], placeOf(k));
    for ([_]usize{ 1, 2, 37, 250, 499 }) |k| {
        var ks = keys;
        selectFirst(&ks, k);
        std.sort.pdq(u128, ks[0..k], {}, std.sort.asc(u128));
        try std.testing.expectEqualSlices(u128, all[0..k], ks[0..k]);
    }
    try std.testing.expect(orderKey(0.5, 0, 0) < orderKey(0.25, 0, 0)); // a higher score sorts first
    try std.testing.expect(orderKey(0.5, 0, 9) < orderKey(0.5, 1, 0)); // then the tie order
}

test "the weights' log is the C library's bit for bit, where Zig's own log differs" {
    // Inputs 1 + (N - df + 0.5)/(df + 0.5) of the IDF weight where Zig's own log (compiler_rt, ported from musl) and
    // the C library's differ in the last bit on this Mac: a scan of every N up to 3,000 and every df (4,504,500
    // inputs) found 6,310, one bit either way. `x` is the input's bits; `libc` is CPython's math.log of it (it calls
    // the C library's log), measured on macOS 27.0.1.
    const Case = struct { n: u32, df: u32, x: u64, libc: u64 };
    const cases = [_]Case{
        .{ .n = 5, .df = 5, .x = 0x3ff1745d1745d174, .libc = 0x3fb64660aa8ce621 }, // log(12/11)
        .{ .n = 32, .df = 12, .x = 0x40051eb851eb851e, .libc = 0x3fef109ef29ae864 },
        .{ .n = 57, .df = 48, .x = 0x3ff3224f2cddb0d3, .libc = 0x3fc6e58398083264 },
        .{ .n = 62, .df = 30, .x = 0x4000864b8a7de6d2, .libc = 0x3fe7368aed7f9a61 },
        .{ .n = 62, .df = 54, .x = 0x3ff27ed3604b27ed, .libc = 0x3fc28d32b8214a56 },
        .{ .n = 73, .df = 29, .x = 0x4004115b1e5f7527, .libc = 0x3fed6df9e8a6470e },
    };
    if (!builtin.os.tag.isDarwin()) return error.SkipZigTest; // the bits are macOS's C library's
    const f = libmLog();
    try std.testing.expect(f != &zigLog);
    var own_differs: usize = 0;
    inline for (cases) |c| {
        // the input as lookup() forms it
        const x: f64 = comptime 1.0 + (@as(f64, @floatFromInt(c.n - c.df)) + 0.5) / (@as(f64, @floatFromInt(c.df)) + 0.5);
        try std.testing.expectEqual(c.x, @as(u64, @bitCast(x)));
        // the log the weights use gives the C library's bits exactly
        try std.testing.expectEqual(c.libc, @as(u64, @bitCast(f(x))));
        // negative control: Zig's own log of the same input (evaluated by the compiler, the same code as compiler_rt's,
        // which a plain `extern fn log` was bound to) gives other bits, so this test can fail
        const own: u64 = comptime @bitCast(@log(x));
        if (own != c.libc) own_differs += 1;
    }
    try std.testing.expectEqual(cases.len, own_differs);
}

fn scratchInit(gpa: Allocator, idx: *const Index) !void {
    var sc = try Scratch.init(gpa, idx);
    sc.deinit();
}

fn indexLoad(gpa: Allocator, bytes: []const u8) !void {
    const idx = try Index.load(gpa, bytes);
    idx.deinit();
}

test "a lookup's scratch and the index each free what they took when any one of their allocations fails" {
    const snapshot = @embedFile("testdata/l6_bytes_snapshot.json"); // TensorFold tools/notes/fixture.py: L6-seed1, bytes
    const idx = try Index.load(std.testing.allocator, snapshot);
    defer idx.deinit();
    try std.testing.expect(idx.notes() > 0);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, scratchInit, .{idx});
    try std.testing.checkAllAllocationFailures(std.testing.allocator, indexLoad, .{snapshot});
}
