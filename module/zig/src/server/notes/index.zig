//! Vendored from LLWZ notes-zig a81d1a9 (zig/notes/src/notes.zig; MIT, (c) 2026 ThinkOff): the notes' index.
const std = @import("std");
const builtin = @import("builtin");
const model_mod = @import("model.zig");
const exact = @import("exact_alloc.zig");
const sa = @import("sa.zig");
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;

/// A log function, called through a pointer.
pub const LogFn = *const fn (f64) callconv(.c) f64;

fn zigLog(x: f64) callconv(.c) f64 {
    return @log(x);
}

/// The C library's log (CPython's math.log calls it), found by name: Zig's own differs in the last bit for some inputs.
pub fn libmLog() LogFn {
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
pub const none_key: u32 = 0xFFFF_FFFF;
/// N-grams are packed into one u128, four keys of 32 bits, so n_max is at most 4.
pub const max_n = 4;
/// subjects() looks for entity phrases of 2 to 8 keys.
pub const max_phrase = 8;
pub const no_note: u32 = 0xFFFF_FFFF;

pub const State = enum(u8) { current = 0, superseded = 1, tombstoned = 2 };
/// The n-gram index: LLWZ's posting lists, or a suffix array over the notes' folded keys.
pub const Kind = enum { postings, sa };

/// What --notes-index and the notes benches build unless told: the suffix array (postings stays selectable).
pub const default_kind: Kind = .sa;

/// Matches over n_max keys, sa only, off by default: all n-grams up to `max`, or each position's longest (lookup.zig).
pub const Long = struct { mode: enum { off, all, longest } = .off, max: u8 = 8 };
pub const Hop = enum { subject, subject_value, none };

pub const Settings = struct {
    resolve_subjects: bool = false,
    resolve_self: bool = true,
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

pub const LoadError = error{ BadSnapshot, Unsupported } || Allocator.Error;

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

/// One note as the snapshot gives it.
const Raw = struct { id: u32, state: State, entity: u32, pair: u32, value_ent: i32, told: i64, line: []const u8, cost: u32, tokens: []const u32 };

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
    ent_cur: Groups, // entity -> places of its current notes (O.ent_cur)
    ent_keys: Groups, // entity -> its pairs (O.ent_keys)
    pair_notes: Groups, // pair -> places of its notes, every state, oldest first (O.by_key)
    entity_names: []const []const u8,
    attribute_names: []const []const u8,
    pair_attributes: []const u32,
    kind: Kind = .postings,
    /// sa: the suffix arrays of places 0..base.notes-1, shared by the versions built on it (null: postings only)
    base: ?*sa.Base = null,
    /// sa: the side buffer's notes' keys (place base.notes + k is group k), for matches longer than n_max
    side_keys: Groups = .{},
    /// postings: every note's; sa: the side buffer's, the notes written since the base (places base.notes..)
    grams: std.HashMapUnmanaged(Gram, Post, GramCtx, 80),
    post: []const u32,
    phrases: std.HashMapUnmanaged(Phrase, u32, PhraseCtx, 80),
    phrase_max: usize,
    aliases: std.HashMapUnmanaged(Phrase, u32, PhraseCtx, 80),
    alias_entities: std.HashMapUnmanaged(Phrase, std.ArrayList(u32), PhraseCtx, 80),
    alias_max: usize,
    self_entity: ?u32,
    names: std.AutoHashMapUnmanaged(u32, u32), // one-key entity phrases: key -> entity (the decode check's, not subjects())
    n_cur: u32,
    n_hist: u32,
    owned: ?*Model = null, // the model a loaded snapshot made, freed with the index
    ver: u64 = 0, // the store's version number for this build (0: a loaded snapshot)

    pub fn notes(self: *const Index) usize {
        return self.note_id.len;
    }

    // Version-local metadata seam. Lookups must not consult the mutable
    // writer model when these columns move to a retained base plus delta.
    pub fn stateAt(self: *const Index, place: u32) State {
        return self.state[place];
    }

    pub fn lineAt(self: *const Index, place: u32) []const u8 {
        return self.line[place];
    }

    pub fn costAt(self: *const Index, place: u32) u32 {
        return self.cost[place];
    }

    pub fn idAt(self: *const Index, place: u32) u32 {
        return self.note_id[place];
    }

    pub fn rankAt(self: *const Index, place: u32) u32 {
        return self.trank[place];
    }

    pub fn toldAt(self: *const Index, place: u32) i64 {
        return self.told[place];
    }

    pub fn entityAt(self: *const Index, place: u32) u32 {
        return self.entity[place];
    }

    pub fn pairAt(self: *const Index, place: u32) u32 {
        return self.pair[place];
    }

    pub fn valueEntityAt(self: *const Index, place: u32) i32 {
        return self.value_ent[place];
    }

    /// A note id's place in the index, if the snapshot has that note.
    pub fn placeOfId(self: *const Index, id: u32) ?u32 {
        if (id >= self.id_place.len or self.id_place[id] == no_note) return null;
        return self.id_place[id];
    }

    pub fn deinit(self: *Index) void {
        const gpa = self.arena.child;
        if (self.base) |b| b.release();
        self.arena.deinit();
        if (self.owned) |m| m.deinit();
        gpa.destroy(self);
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

    /// build() with the n-gram index of `kind` (sa: a new base over every note, nothing in the side buffer).
    pub fn buildKind(gpa: Allocator, m: *const Model, kind: Kind) LoadError!*Index {
        return buildWith(gpa, m, kind, null);
    }

    /// The index of a model's notes, as a snapshot of them would load; the model must outlive it (it reads its vocab).
    pub fn build(gpa: Allocator, m: *const Model) LoadError!*Index {
        return buildWith(gpa, m, .postings, null);
    }

    /// The index of `kind`; sa: on `reuse` (a base over the first notes) with the rest as side postings, or a new base.
    pub fn buildWith(gpa: Allocator, m: *const Model, kind: Kind, reuse: ?*sa.Base) LoadError!*Index {
        const self = try gpa.create(Index);
        errdefer gpa.destroy(self);
        self.* = undefined;
        self.arena = exact.Exact.init(gpa);
        errdefer self.arena.deinit();
        self.owned = null;
        self.ver = 0;
        self.kind = kind;
        self.base = null;
        errdefer if (self.base) |b| b.release();
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
        try self.placeNotes(gpa, raw);
        try self.groupsOf(m);
        const meta = self.arena.allocator();
        const entities = try meta.alloc([]const u8, m.entities.items.len);
        for (m.entities.items, entities) |name, *out| out.* = try meta.dupe(u8, name);
        self.entity_names = entities;
        const attributes = try meta.alloc([]const u8, m.attributes.items.len);
        for (m.attributes.items, attributes) |name, *out| out.* = try meta.dupe(u8, name);
        self.attribute_names = attributes;
        const pair_attributes = try meta.alloc(u32, m.pairs.items.len);
        for (m.pairs.items, pair_attributes) |p, *out| out.* = p.attribute;
        self.pair_attributes = pair_attributes;
        self.phrases = .empty;
        self.phrase_max = 0;
        self.aliases = .empty;
        self.alias_entities = .empty;
        self.alias_max = 0;
        self.self_entity = m.entityOf("me");
        self.names = .empty;
        try self.phrasesOf(m);
        const ve = try @import("value.zig").ofWrites(self.arena.allocator(), self, m);
        if (ve.ptr != self.value_ent.ptr) self.arena.allocator().free(self.value_ent); // one array, as placed
        self.value_ent = ve;
        var first: usize = 0;
        self.side_keys = .{};
        if (kind == .sa) {
            self.base = if (reuse) |b| b.retain() else try baseOf(gpa, m, raw);
            first = self.base.?.notes;
            try self.sideKeys(m, raw[first..]);
        }
        try self.buildPostings(gpa, raw, first);
        return self;
    }

    /// The side buffer's notes' keys, flat.
    fn sideKeys(self: *Index, m: *const Model, side: []const Raw) LoadError!void {
        const a = self.arena.allocator();
        const off = try a.alloc(u32, side.len + 1);
        var total: u32 = 0;
        for (side, 0..) |x, k| {
            off[k] = total;
            total += @intCast(x.tokens.len);
        }
        off[side.len] = total;
        const items = try a.alloc(u32, total);
        var at: usize = 0;
        for (side) |x| for (x.tokens) |t| {
            items[at] = m.keyOf(t);
            at += 1;
        };
        self.side_keys = .{ .off = off, .items = items };
    }

    /// A suffix-array base over every note: their folded keys, each view's suffixes sorted (the build's memory freed).
    fn baseOf(gpa: Allocator, m: *const Model, raw: []const Raw) LoadError!*sa.Base {
        try saVocab(gpa, m);
        var total: usize = 0;
        for (raw) |x| total += x.tokens.len;
        const keys = try gpa.alloc(u32, total);
        defer gpa.free(keys);
        const src = try gpa.alloc(sa.Source, raw.len);
        defer gpa.free(src);
        var at: usize = 0;
        for (raw, src) |x, *o| {
            for (x.tokens, keys[at..][0..x.tokens.len]) |t, *k| k.* = m.keyOf(t);
            o.* = .{ .keys = keys[at..][0..x.tokens.len], .ids = x.tokens, .cur = x.state == .current };
            at += x.tokens.len;
        }
        return sa.build(gpa, src, m.vwordy);
    }

    /// Key sequences match as packed n-grams do only with no none_key key and one wordiness per key: else Unsupported.
    fn saVocab(gpa: Allocator, m: *const Model) LoadError!void {
        var max: u32 = 0;
        for (m.vkey) |k| {
            if (k == none_key) return error.Unsupported;
            max = @max(max, k & ~key_id);
        }
        const seen = try gpa.alloc(u8, 2 * (@as(usize, max) + 1)); // by key: 0 unseen, 1 a word, 2 not
        defer gpa.free(seen);
        @memset(seen, 0);
        for (m.vkey, 0..) |k, t| {
            const at = 2 * @as(usize, k & ~key_id) + @intFromBool(k & key_id != 0);
            const w: u8 = if (t < m.vwordy.len and m.vwordy[t]) 1 else 2;
            if (seen[at] == 0) seen[at] = w else if (seen[at] != w) return error.Unsupported;
        }
    }

    /// The notes by place, with Python's tie order (-told, id) and the least a line can add.
    fn placeNotes(self: *Index, gpa: Allocator, raw: []const Raw) LoadError!void {
        const a = self.arena.allocator();
        const n = raw.len;
        const note_id = try a.alloc(u32, n);
        const state = try a.alloc(State, n);
        const told = try a.alloc(i64, n);
        const entity = try a.alloc(u32, n);
        const pair = try a.alloc(u32, n);
        const value_ent = try a.alloc(i32, n);
        const line = try a.alloc([]const u8, n);
        const cost = try a.alloc(u32, n);
        var max_id: u32 = 0;
        for (raw) |x| max_id = @max(max_id, x.id);
        const id_place = try a.alloc(u32, @as(usize, max_id) + 1);
        @memset(id_place, no_note);
        var n_cur: u32 = 0;
        var n_hist: u32 = 0;
        for (raw, 0..) |x, k| {
            if (id_place[x.id] != no_note) return error.BadSnapshot;
            id_place[x.id] = @intCast(k);
            note_id[k] = x.id;
            state[k] = x.state;
            if (x.state == .current) n_cur += 1 else n_hist += 1;
            told[k] = x.told;
            entity[k] = x.entity;
            pair[k] = x.pair;
            value_ent[k] = x.value_ent;
            line[k] = x.line; // the model's own: a write frees a replaced line only once no version can read it
            cost[k] = x.cost;
        }
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
        self.note_id = note_id;
        self.state = state;
        self.told = told;
        self.trank = trank;
        self.entity = entity;
        self.pair = pair;
        self.value_ent = value_ent;
        self.line = line;
        self.cost = cost;
        var least: i64 = std.math.maxInt(i64);
        for (cost) |c| least = @min(least, @as(i64, c) + self.fx_nl);
        self.min_line = least;
        self.id_place = id_place;
        self.n_cur = n_cur;
        self.n_hist = n_hist;
    }

    /// O.ent_cur, O.ent_keys and O.by_key as note places, flat: one pass to count, one to fill (place, pair order).
    fn groupsOf(self: *Index, m: *const Model) LoadError!void {
        const a = self.arena.allocator();
        const ne = m.entities.items.len;
        const np = m.pairs.items.len;
        for (self.pair) |p| if (p >= np) return error.BadSnapshot;
        for (self.entity) |e| if (e >= ne) return error.BadSnapshot;
        for (m.pairs.items) |p| if (p.entity >= ne) return error.BadSnapshot;
        const cur_off = try a.alloc(u32, ne + 1);
        const key_off = try a.alloc(u32, ne + 1);
        @memset(cur_off, 0);
        @memset(key_off, 0);
        for (self.state, self.entity) |st, e| if (st == .current) {
            cur_off[e + 1] += 1;
        };
        for (m.pairs.items) |p| key_off[p.entity + 1] += 1;
        for (1..ne + 1) |e| {
            cur_off[e] += cur_off[e - 1];
            key_off[e] += key_off[e - 1];
        }
        const cur_items = try a.alloc(u32, cur_off[ne]);
        const key_items = try a.alloc(u32, key_off[ne]);
        // each group fills from its start: off[e] is moved along it, then back
        for (self.state, self.entity, 0..) |st, e, k| if (st == .current) {
            cur_items[cur_off[e]] = @intCast(k);
            cur_off[e] += 1;
        };
        for (m.pairs.items, 0..) |p, k| {
            key_items[key_off[p.entity]] = @intCast(k);
            key_off[p.entity] += 1;
        }
        var e = ne;
        while (e > 0) : (e -= 1) {
            cur_off[e] = cur_off[e - 1];
            key_off[e] = key_off[e - 1];
        }
        cur_off[0] = 0;
        key_off[0] = 0;
        var total: usize = 0;
        for (m.pairs.items) |p| total += p.ids.items.len;
        const pair_off = try a.alloc(u32, np + 1);
        const pair_items = try a.alloc(u32, total);
        var at: u32 = 0;
        for (m.pairs.items, 0..) |p, k| {
            pair_off[k] = at;
            for (p.ids.items) |id| {
                pair_items[at] = self.placeOfId(id) orelse return error.BadSnapshot;
                at += 1;
            }
        }
        pair_off[np] = at;
        self.ent_cur = .{ .off = cur_off, .items = cur_items };
        self.ent_keys = .{ .off = key_off, .items = key_items };
        self.pair_notes = .{ .off = pair_off, .items = pair_items };
    }

    /// Entity phrases: only 2 to 8 keys can ever match (subjects() looks at those lengths).
    fn phrasesOf(self: *Index, m: *const Model) LoadError!void {
        const a = self.arena.allocator();
        var count: u32 = 0;
        for (m.phrases.items) |p| count += @intFromBool(p.keys.len >= 2 and p.keys.len <= max_phrase);
        try self.phrases.ensureTotalCapacity(a, count); // once: no outgrown table left behind
        for (m.phrases.items) |p| {
            if (p.entity >= m.entities.items.len) return error.BadSnapshot;
            if (p.keys.len == 1) try self.names.put(a, p.keys[0], p.entity);
            if (p.keys.len < 2 or p.keys.len > max_phrase) continue;
            self.phrases.putAssumeCapacity(packPhrase(p.keys), p.entity);
            self.phrase_max = @max(self.phrase_max, p.keys.len);
        }
    }

    /// The postings of both views, from each note's token ids by the same grams() as a question's.
    fn buildPostings(self: *Index, gpa: Allocator, all: []const Raw, first: usize) LoadError!void {
        const raw = all[first..]; // the notes from place `first` on (sa: the side buffer's)
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
        for (raw, 0..) |x, k| {
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
                if (self.state[first + k] == .current) e.cur_len += 1 else e.hist_len += 1;
                try hits.append(gpa, .{ .entry = r.value_ptr.*, .note = @intCast(first + k) });
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
        try self.grams.ensureTotalCapacity(a, @intCast(entries.items.len)); // counted first: one table, no regrowth
        var it = map.iterator();
        while (it.next()) |kv| self.grams.putAssumeCapacity(kv.key_ptr.*, entries.items[kv.value_ptr.*]);
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
