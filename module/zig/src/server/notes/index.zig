//! Vendored from LLWZ notes-zig a81d1a9 (zig/notes/src/notes.zig; MIT, (c) 2026 ThinkOff): the notes' index.
const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const model_mod = @import("model.zig");
const exact = @import("exact_alloc.zig");
const sa = @import("sa.zig");
const ib = @import("index_build.zig");
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;

/// A log function, called through a pointer.
pub const LogFn = *const fn (f64) callconv(.c) f64;

fn zigLog(x: f64) callconv(.c) f64 {
    return @log(x);
}

/// The C library's log (CPython's math.log calls it), found by name: Zig's own differs in the last bit for some inputs.
pub fn libmLog() LogFn {
    if (comptime builtin.os.tag == .windows) {
        const crt = struct {
            pub extern "c" fn log(x: f64) f64;
        };
        return &crt.log;
    }
    // RTLD_DEFAULT: the handle that searches every loaded image (-2 on Apple systems, 0 elsewhere)
    const default: ?*anyopaque = if (builtin.os.tag.isDarwin()) @ptrFromInt(@as(usize, @bitCast(@as(isize, -2)))) else null;
    if (std.c.dlsym(default, "log")) |p| return @ptrCast(@alignCast(p));
    return &zigLog;
}

pub fn isZigLog(f: LogFn) bool {
    return f == &zigLog;
}

/// A key that is the token id itself has this bit; a fold string's key is its place in the interned list.
pub const key_id: u32 = 0x8000_0000;
/// No key: an unused place in a packed n-gram, and the key of a token id the vocabulary lacks (it matches nothing).
pub const none_key = types.none_key;
/// N-grams are packed into one u128, four keys of 32 bits, so n_max is at most 4.
pub const max_n = types.max_n;
/// Entity phrases are 2 to 8 keys.
pub const max_phrase = 8;
pub const no_note: u32 = 0xFFFF_FFFF;
/// rankAt() by note: a place below 2^24, told in 0..told_span; ids ascend with places, so a place tie is an id tie.
pub const id_bits = 24;
pub const told_span: i64 = (1 << 40) - 1;

/// Whether a note's place and told make its rankAt() key.
pub fn rankFits(place: usize, told: i64) bool {
    return place < (1 << id_bits) and told >= 0 and told <= told_span;
}

pub const State = types.State;
/// The n-gram index: LLWZ's posting lists, or a suffix array over the notes' folded keys.
pub const Kind = enum { postings, sa };

/// What --notes-index and the notes benches build unless told: the suffix array (postings stays selectable).
pub const default_kind: Kind = .sa;

/// Matches over n_max keys, sa only, off by default: all n-grams up to `max`, or each position's longest.
pub const Long = struct { mode: enum { off, all, longest } = .off, max: u8 = 8 };
pub const Hop = types.Hop;
pub const Settings = types.Settings;
pub const LoadError = types.LoadError;

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

pub const GramCtx = struct {
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

pub const Phrase = [max_phrase]u32;
pub const PhraseCtx = struct {
    pub fn hash(_: PhraseCtx, p: Phrase) u64 {
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&p));
    }
    pub fn eql(_: PhraseCtx, a: Phrase, b: Phrase) bool {
        return std.mem.eql(u32, &a, &b);
    }
};

pub fn packPhrase(keys: []const u32) Phrase {
    var p: Phrase = @splat(none_key);
    @memcpy(p[0..keys.len], keys);
    return p;
}

/// Postings of one n-gram: note places in the current view, then in the history view.
pub const Post = struct { cur_off: u32, cur_len: u32, hist_off: u32, hist_len: u32 };

/// Groups of note places (or pairs) by a number, flat: group k is items[off[k]..off[k+1]].
pub const Groups = struct {
    off: []const u32 = &.{},
    items: []const u32 = &.{},

    pub inline fn of(g: Groups, k: usize) []const u32 {
        return g.items[g.off[k]..g.off[k + 1]];
    }

    pub fn len(g: Groups) usize {
        return g.off.len -| 1;
    }
};

/// One group's note places, in order.
pub const Iter = struct {
    a: []const u32,
    i: usize = 0,

    pub inline fn next(it: *Iter) ?u32 {
        if (it.i >= it.a.len) return null;
        it.i += 1;
        return it.a[it.i - 1];
    }
};

/// One note as the snapshot gives it.
pub const Raw = struct { id: u32, state: State, entity: u32, pair: u32, value_ent: i32, told: i64, line: []const u8, cost: u32, tokens: []const u32 };

pub const Index = struct {
    arena: exact.Exact, // every array at its exact size: no arena node grown 1.5 times past the postings
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
    /// The tokenizer.json SHA-256 the snapshot was exported with ("" when it names none).
    tokenizer_sha256: []const u8,
    // the notes by place (ids: Python's)
    own_id: []const u32,
    state: []const State,
    own_told: []const i64,
    own_trank: []const u32, // place in the order (-told, id): Python's tie-break after the score
    own_entity: []const u32,
    own_pair: []const u32,
    own_value: []const i32,
    own_line: []const []const u8,
    own_cost: []const u32,
    min_line: i64, // the least any line can add to a notes block: its cost and a line break
    own_ent_cur: Groups, // entity -> places of its current notes (O.ent_cur)
    own_ent_keys: Groups, // entity -> its pairs (O.ent_keys)
    own_pair_notes: Groups, // pair -> places of its notes, every state, oldest first (O.by_key)
    entity_names: []const []const u8,
    attribute_names: []const []const u8,
    pair_attributes: []const u32,
    kind: Kind = .postings,
    /// sa: the suffix arrays of every note (null: postings only)
    base: ?*sa.Base = null,
    /// postings: every note's (sa: none)
    grams: std.HashMapUnmanaged(Gram, Post, GramCtx, 80),
    post: []const u32,
    own_phrases: std.HashMapUnmanaged(Phrase, u32, PhraseCtx, 80),
    own_phrase_max: usize,
    own_names: std.AutoHashMapUnmanaged(u32, u32), // one-key entity phrases: key -> entity
    // aliases (alias.zig): alias keys -> the entity they name, and every entity they name
    aliases: std.HashMapUnmanaged(Phrase, u32, PhraseCtx, 80),
    alias_entities: std.HashMapUnmanaged(Phrase, std.ArrayList(u32), PhraseCtx, 80),
    alias_max: usize,
    self_entity: ?u32,
    n_cur: u32,
    n_hist: u32,
    owned: ?*Model = null, // the model a loaded snapshot made, freed with the index
    ver: u64 = 0, // the store's version number for this build (0: a loaded snapshot)
    n_notes: u32 = 0,
    max_told: i64 = std.math.minInt(i64), // the latest told of its notes
    n_phrases: u32 = 0, // the model's phrases it was built from
    rank_by_note: bool = false, // rankAt() from each note's told and place (every one fits), else own_trank
    alias_candidates_off: bool = false, // a view that reads no alias candidates

    pub fn notes(self: *const Index) usize {
        return self.n_notes;
    }

    pub inline fn stateAt(self: *const Index, place: u32) State {
        return self.state[place];
    }

    pub fn lineAt(self: *const Index, p: u32) []const u8 {
        return self.own_line[p];
    }

    pub fn costAt(self: *const Index, p: u32) u32 {
        return self.own_cost[p];
    }

    pub inline fn idAt(self: *const Index, p: u32) u32 {
        return self.own_id[p];
    }

    pub inline fn toldAt(self: *const Index, p: u32) i64 {
        return self.own_told[p];
    }

    pub inline fn entityAt(self: *const Index, p: u32) u32 {
        return self.own_entity[p];
    }

    pub inline fn pairAt(self: *const Index, p: u32) u32 {
        return self.own_pair[p];
    }

    pub fn valueEntityAt(self: *const Index, p: u32) i32 {
        return self.own_value[p];
    }

    /// Python's tie order (-told, id) as an ascending key: by the note when it fits (rank_by_note), else own_trank.
    pub inline fn rankAt(self: *const Index, p: u32) u64 {
        if (self.rank_by_note) return (@as(u64, @intCast(told_span - self.toldAt(p))) << id_bits) | p;
        return (@as(u64, self.own_trank[p]) << 32) | p;
    }

    /// The place a rankAt() key was made from.
    pub inline fn placeOfRank(self: *const Index, r: u64) u32 {
        if (self.rank_by_note) return @intCast(r & ((1 << id_bits) - 1));
        return @truncate(r);
    }

    /// The entity an alias names (no_note: several do), or null.
    pub fn aliasOf(self: *const Index, p: Phrase) ?u32 {
        return self.aliases.get(p);
    }

    /// Every entity an alias names, in the order their name facts were written, or null.
    pub fn aliasCandidates(self: *const Index, p: Phrase) ?[]const u32 {
        if (self.alias_candidates_off) return null;
        const l = self.alias_entities.get(p) orelse return null;
        return if (l.items.len == 0) null else l.items;
    }

    /// The longest alias, in keys (a bound for the lookups' loops).
    pub fn aliasMax(self: *const Index) usize {
        return self.alias_max;
    }

    /// The entities this index knows (entity numbers below it).
    pub fn entityCount(self: *const Index) usize {
        return self.own_ent_cur.len();
    }

    /// The attributes this index knows.
    pub fn attributeCount(self: *const Index) usize {
        return self.attribute_names.len;
    }

    pub fn entityName(self: *const Index, e: u32) []const u8 {
        return self.entity_names[e];
    }

    pub fn attributeName(self: *const Index, x: u32) []const u8 {
        return self.attribute_names[x];
    }

    pub fn pairAttribute(self: *const Index, pr: u32) u32 {
        return self.pair_attributes[pr];
    }

    /// A note id's place in the index, if it has that note: own_id ascends, so no table sized by the largest id.
    pub fn placeOfId(self: *const Index, id: u32) ?u32 {
        var lo: usize = 0;
        var hi: usize = self.own_id.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.own_id[mid] == id) return @intCast(mid);
            if (self.own_id[mid] < id) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    /// Frees the index, and the model a loaded snapshot made.
    pub fn deinit(self: *Index) void {
        const gpa = self.arena.child;
        if (self.base) |b| b.deinit();
        self.arena.deinit();
        if (self.owned) |m| m.deinit();
        gpa.destroy(self);
    }

    /// The places of an entity's current notes, by place (O.ent_cur).
    pub fn entCur(self: *const Index, e: u32) Iter {
        return .{ .a = self.own_ent_cur.of(e) };
    }

    /// An entity's pairs (O.ent_keys).
    pub fn entKeys(self: *const Index, e: u32) Iter {
        return .{ .a = self.own_ent_keys.of(e) };
    }

    /// A pair's notes, every state, oldest first (O.by_key).
    pub fn pairNotes(self: *const Index, pr: u32) Iter {
        return .{ .a = self.own_pair_notes.of(pr) };
    }

    /// The entity a phrase of 2 to 8 keys names (a later phrase wins, as the model's list order makes it).
    pub fn phraseOf(self: *const Index, p: Phrase) ?u32 {
        return self.own_phrases.get(p);
    }

    /// The entity a one-key name names.
    pub fn nameOf(self: *const Index, k: u32) ?u32 {
        return self.own_names.get(k);
    }

    pub fn phraseMax(self: *const Index) usize {
        return self.own_phrase_max;
    }

    pub fn phraseCount(self: *const Index) usize {
        return self.own_phrases.count();
    }

    /// Reads a snapshot (notes_export.py's JSON) and builds the index, checking notes and distinct n-grams per view.
    pub fn load(gpa: Allocator, text: []const u8) LoadError!*Index {
        const m = try model_mod.parse(gpa, text);
        errdefer m.deinit();
        const self = try build(gpa, m);
        self.owned = m;
        errdefer {
            self.owned = null;
            self.deinit();
        }
        var gc: u32 = 0;
        var gh: u32 = 0;
        var it = self.grams.valueIterator();
        while (it.next()) |p| {
            if (p.cur_len > 0) gc += 1;
            if (p.hist_len > 0) gh += 1;
        }
        if (gc != m.grams_current or gh != m.grams_history) return error.BadSnapshot;
        return self;
    }

    /// The index of a model's notes, as a snapshot of them would load; the model must outlive it (it reads its vocab).
    pub fn build(gpa: Allocator, m: *const Model) LoadError!*Index {
        return buildKind(gpa, m, .postings);
    }

    /// The index of `kind` over every note (sa: a suffix-array base, no postings).
    pub fn buildKind(gpa: Allocator, m: *const Model, kind: Kind) LoadError!*Index {
        const self = try gpa.create(Index);
        errdefer gpa.destroy(self);
        self.* = undefined;
        self.arena = exact.Exact.init(gpa);
        errdefer self.arena.deinit();
        self.owned = null;
        self.ver = 0;
        self.alias_candidates_off = false;
        self.n_notes = @intCast(m.notes.items.len);
        self.n_phrases = @intCast(m.phrases.items.len);
        self.max_told = std.math.minInt(i64);
        for (m.notes.items) |*n| self.max_told = @max(self.max_told, n.told);
        self.kind = kind;
        self.base = null;
        errdefer if (self.base) |b| b.deinit();
        self.log = libmLog();
        self.settings = m.settings;
        self.head = m.head;
        self.cur_title = m.cur_title;
        self.hist_title = m.hist_title;
        self.fx_head = m.fixed.head;
        self.fx_cur = m.fixed.current;
        self.fx_hist = m.fixed.history;
        self.fx_nl = m.fixed.nl;
        self.tokenizer_sha256 = m.tokenizer_sha256;
        self.vkey = m.vkey;
        self.vwordy = m.vwordy;
        const raw = try gpa.alloc(Raw, m.notes.items.len); // the build's own: freed when it returns
        defer gpa.free(raw);
        for (m.notes.items, raw) |n, *r| r.* = .{ .id = n.id, .state = n.state, .entity = n.entity, .pair = n.pair, .value_ent = n.value_ent, .told = n.told, .line = n.line(), .cost = n.cost, .tokens = n.tokens() };
        try ib.placeNotes(self, gpa, raw);
        try ib.groupsOf(self, m);
        self.own_phrases = .empty;
        self.own_phrase_max = 0;
        self.own_names = .empty;
        try ib.phrasesOf(self, m.phrases.items, m.entities.items.len);
        try ib.namesOf(self, m);
        self.aliases = .empty;
        self.alias_entities = .empty;
        self.alias_max = 0;
        const ve = try @import("value.zig").ofWrites(self.arena.allocator(), self, m);
        if (ve.ptr != self.own_value.ptr) self.arena.allocator().free(self.own_value); // one array, as placed
        self.own_value = ve;
        if (kind == .sa) self.base = try ib.baseOf(gpa, m, raw);
        try ib.buildPostings(self, gpa, if (kind == .sa) &.{} else raw); // sa: an empty postings table
        return self;
    }

    pub inline fn keyOf(self: *const Index, t: u32) u32 {
        return if (t < self.vkey.len) self.vkey[t] else none_key;
    }

    inline fn wordyOf(self: *const Index, t: u32) bool {
        return t < self.vwordy.len and self.vwordy[t];
    }

    /// NoteStore.grams: n from n_min to n_max at every start, distinct, in the order first met (Python keeps a set).
    pub fn gramsOf(self: *const Index, gpa: Allocator, ids: []const u32, keys: []const u32, out: *std.ArrayList(Gram)) Allocator.Error!void {
        const st = self.settings;
        var n: usize = st.n_min;
        while (n <= st.n_max) : (n += 1) {
            if (keys.len < n) break;
            var j: usize = 0;
            while (j + n <= keys.len) : (j += 1) {
                if (n == 1 and !st.punct and !self.wordyOf(ids[j])) continue; // punctuation alone is no key
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

test "packGram keeps lengths apart" {
    const a = packGram(&.{ 1, 2 });
    const b = packGram(&.{ 1, 2, none_key });
    const c = packGram(&.{ 1, 2, 3 });
    try std.testing.expect(a == b); // none_key is never a real key, so this pair cannot occur in a snapshot
    try std.testing.expect(a != c);
    try std.testing.expect(packGram(&.{1}) != packGram(&.{ 1, 0 }));
}
