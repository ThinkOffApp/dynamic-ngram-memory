//! The suffix-array index (--notes-index sa): LLWZ Python's answers, and its n-grams and notes equal the postings'.
const std = @import("std");
const server = @import("root.zig");
const notes_route = @import("notes_route.zig");
const index = @import("notes/index.zig");
const sa = @import("notes/sa.zig");
const model_mod = @import("notes/model.zig");
const notes_test = @import("notes_test.zig");
const Bytes = notes_test.Bytes;
const testing = std.testing;
const Allocator = std.mem.Allocator;

test "subject aliases contain only current names and first person words" {
    var text: Bytes = .{};
    const store = try openSa(text.text(), notes_test.snapshot);
    defer store.deinit();
    const idx = store.current.idx;
    try testing.expectEqual(store.model.entities.items.len, idx.entity_names.len);
    try testing.expectEqualStrings(store.model.entities.items[0], idx.entity_names[0]);
    try testing.expect(store.model.entities.items[0].ptr != idx.entity_names[0].ptr);
    try testing.expectEqual(store.model.attributes.items.len, idx.attribute_names.len);
    for (store.model.pairs.items, idx.pair_attributes) |p, attribute| try testing.expectEqual(p.attribute, attribute);
    for ([_][]const u8{ "Ana", "Bruno", "Milo", "Otto", "my", "me", "I" }) |name| {
        const spaced = try std.mem.concat(testing.allocator, u8, &.{ " ", name });
        defer testing.allocator.free(spaced);
        const ids = try text.text().encode(testing.allocator, spaced, false);
        defer testing.allocator.free(ids);
        var keys: [index.max_phrase]u32 = @splat(index.none_key);
        for (ids, keys[0..ids.len]) |id, *key| key.* = idx.keyOf(id);
        try testing.expect(idx.aliases.get(keys) != null);
    }
    const old_name = try text.text().encode(testing.allocator, " Luna", false);
    defer testing.allocator.free(old_name);
    var keys: [index.max_phrase]u32 = @splat(index.none_key);
    for (old_name, keys[0..old_name.len]) |id, *key| key.* = idx.keyOf(id);
    try testing.expect(idx.aliases.get(keys) == null);
    try testing.expect(!idx.settings.resolve_subjects);
}

test "subject aliases reject ambiguous names and recover after take back" {
    const rig = @import("notes_rig_test.zig");
    const r = try rig.Rig.initKind(testing.allocator, .sa, 1024);
    defer r.deinit();
    const store = r.store();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try store.text.encode(a, " Ana", false);
    var keys: [index.max_phrase]u32 = @splat(index.none_key);
    for (ids, keys[0..ids.len]) |id, *key| key.* = store.current.idx.keyOf(id);
    const sister = store.current.idx.aliases.get(keys).?;
    const reply = try rig.request(r.srv, a, "PUT", "/v1/notes/16", "{\"text\":\"My brother's name is Ana.\"}");
    try testing.expectEqual(@as(u16, 201), reply.status);
    try testing.expectEqual(index.no_note, store.current.idx.aliases.get(keys).?);
    const candidates = store.current.idx.alias_entities.get(keys).?.items;
    try testing.expectEqual(@as(usize, 2), candidates.len);
    try testing.expect(std.mem.indexOfScalar(u32, candidates, sister) != null);
    try testing.expect(std.mem.indexOfScalar(u32, candidates, store.model.entityOf("my brother").?) != null);
    const lookup_mod = @import("notes/lookup.zig");
    var view = store.current.idx.*;
    var scratch = try lookup_mod.Scratch.init(testing.allocator, &view);
    defer scratch.deinit();
    _ = try lookup_mod.lookup(&view, &scratch, ids, false, 512, &.{}, .{});
    try testing.expectEqual(@as(usize, 0), scratch.found.items.len);
    view.settings.resolve_subjects = true;
    _ = try lookup_mod.lookup(&view, &scratch, ids, false, 512, &.{}, .{});
    try testing.expectEqual(@as(usize, 2), scratch.found.items.len);
    try testing.expectEqualSlices(u32, candidates, scratch.found.items);
    for (scratch.found_weights.items) |weight| {
        try testing.expect(weight > 0 and weight < 1);
        try testing.expectApproxEqAbs(scratch.found_weights.items[0], weight, 1e-12);
    }
    const decode = @import("notes_decode.zig");
    var live: decode.Live = .{ .a = a, .on = true, .store = store, .ver = store.current.idx.ver, .question = "", .past = false, .cap = 512, .exclude = &.{}, .place = .first, .echo = false, .shape = .{ .resolve_subjects = true }, .block = "", .block_tokens = 0 };
    try testing.expect(!live.nameJustWritten(ids)); // unresolved collisions must not expand live fetch
    // Once all candidates' current notes are held, the same name cannot
    // trigger another fetch. A collision never chooses an arbitrary entity.
    for (candidates) |e| for (store.current.idx.ent_cur.of(e)) |p| try live.held.append(a, store.current.idx.idAt(p));
    try testing.expect(!live.nameJustWritten(ids));
    const newest = store.model.notes.items[store.model.notes.items.len - 1].id;
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}", .{newest});
    const withdrawn = try rig.request(r.srv, a, "DELETE", path, "{}");
    try testing.expectEqual(@as(u16, 201), withdrawn.status);
    try testing.expectEqual(sister, store.current.idx.aliases.get(keys).?);
    try testing.expectEqualSlices(u32, &.{sister}, store.current.idx.alias_entities.get(keys).?.items);
}

test "subject aliases are consulted only by the opt in lookup view" {
    var text: Bytes = .{};
    const store = try openSa(text.text(), notes_test.snapshot);
    defer store.deinit();
    const lookup_mod = @import("notes/lookup.zig");
    var view = store.current.idx.*;
    var scratch = try lookup_mod.Scratch.init(testing.allocator, &view);
    defer scratch.deinit();
    const qids = try text.text().encode(testing.allocator, " Ana", false);
    defer testing.allocator.free(qids);
    _ = try lookup_mod.lookup(&view, &scratch, qids, false, 512, &.{}, .{});
    try testing.expectEqual(@as(usize, 0), scratch.found.items.len);
    view.settings.resolve_subjects = true;
    _ = try lookup_mod.lookup(&view, &scratch, qids, false, 512, &.{}, .{});
    try testing.expectEqual(@as(usize, 1), scratch.found.items.len);
    try testing.expectEqual(store.model.entityOf("my sister").?, scratch.found.items[0]);
    const self_ids = try text.text().encode(testing.allocator, " my", false);
    defer testing.allocator.free(self_ids);
    view.settings.resolve_self = false;
    _ = try lookup_mod.lookup(&view, &scratch, self_ids, false, 512, &.{}, .{});
    try testing.expectEqual(@as(usize, 0), scratch.found.items.len);
}

test "subject aliases survive erasing an unrelated note" {
    const rig = @import("notes_rig_test.zig");
    const r = try rig.Rig.initKind(testing.allocator, .sa, 1024);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try r.store().text.encode(a, " Ana", false);
    var keys: [index.max_phrase]u32 = @splat(index.none_key);
    for (ids, keys[0..ids.len]) |id, *key| key.* = r.store().current.idx.keyOf(id);
    const before = r.store().current.idx.aliases.get(keys).?;
    const added = try rig.request(r.srv, a, "POST", "/v1/notes", "{\"text\":\"An unrelated bicycle is silver.\"}");
    try testing.expectEqual(@as(u16, 201), added.status);
    const id = r.store().model.notes.items[r.store().model.notes.items.len - 1].id;
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id});
    const reply = try rig.request(r.srv, a, "DELETE", path, "{}");
    try testing.expectEqual(@as(u16, 200), reply.status);
    try testing.expectEqual(before, r.store().current.idx.aliases.get(keys).?);
}

test "version metadata accessors keep a held reader unchanged across a replacement" {
    const rig = @import("notes_rig_test.zig");
    const r = try rig.Rig.initKind(testing.allocator, .sa, 1024);
    defer r.deinit();
    const store = r.store();
    const held = store.acquire();
    defer store.release(held);
    var place: u32 = 0;
    while (held.idx.stateAt(place) != .current) : (place += 1) {}
    const id = held.idx.idAt(place);
    const line = try testing.allocator.dupe(u8, held.idx.lineAt(place));
    defer testing.allocator.free(line);
    const cost = held.idx.costAt(place);
    const rank = held.idx.rankAt(place);
    const told = held.idx.toldAt(place);
    const entity = held.idx.entityAt(place);
    const pair = held.idx.pairAt(place);
    const value_entity = held.idx.valueEntityAt(place);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}", .{id});
    const reply = try rig.request(r.srv, a, "PUT", path, "{\"text\":\"A uniquely new green Saab is parked in Tampere.\"}");
    try testing.expectEqual(@as(u16, 201), reply.status);
    try testing.expectEqual(index.State.current, held.idx.stateAt(place));
    try testing.expectEqualStrings(line, held.idx.lineAt(place));
    try testing.expectEqual(cost, held.idx.costAt(place));
    try testing.expectEqual(rank, held.idx.rankAt(place));
    try testing.expectEqual(told, held.idx.toldAt(place));
    try testing.expectEqual(entity, held.idx.entityAt(place));
    try testing.expectEqual(pair, held.idx.pairAt(place));
    try testing.expectEqual(value_entity, held.idx.valueEntityAt(place));
    try testing.expectEqual(index.State.superseded, store.current.idx.stateAt(place));
}

fn openSa(text: server.model_text.Text, bytes: []const u8) !*notes_route.Store {
    var why: []const u8 = "";
    return notes_route.Store.loadIndexed(testing.allocator, testing.io, bytes, text, .sa, &why);
}

test "the suffix-array index gives LLWZ Python's ids, tokens, candidates and block for every fixture case" {
    var text: Bytes = .{};
    const store = try openSa(text.text(), notes_test.snapshot);
    defer store.deinit();
    try testing.expect(store.current.idx.base != null);
    var answered: usize = 0;
    try testing.expectEqual(@as(usize, 0), try notes_test.differing(store, text.text(), &answered));
    try testing.expect(answered >= 60);
}

test "negative control: the suffix-array index with the hop off answers otherwise than Python" {
    var text: Bytes = .{};
    const broken = try std.mem.replaceOwned(u8, testing.allocator, notes_test.snapshot, "\"hop\":\"subject\"", "\"hop\":\"none\"");
    defer testing.allocator.free(broken);
    const store = try openSa(text.text(), broken);
    defer store.deinit();
    var answered: usize = 0;
    try testing.expect(try notes_test.differing(store, text.text(), &answered) > 0);
}

/// One view's notes holding `g` by the suffix array (each once, by place), against the postings' list.
fn notesOf(a: Allocator, b: *const sa.Base, arr: sa.Packed, g: index.Gram, punct: bool) ![]u32 {
    var keys: [index.max_n]u32 = undefined;
    var len: usize = 0;
    while (len < index.max_n) : (len += 1) {
        keys[len] = @truncate(g >> @intCast(32 * len));
        if (keys[len] == index.none_key) break;
    }
    var out: std.ArrayList(u32) = .empty;
    var buf: [index.max_n]u32 = undefined;
    const pat = b.codes(keys[0..len], &buf) orelse return out.items;
    if (len == 1 and !punct and !b.wordyCode(pat[0])) return out.items;
    const r = b.range(arr, pat);
    for (r[0]..r[1]) |k| {
        const p = b.placeOf(arr.get(k));
        if (std.mem.indexOfScalar(u32, out.items, p) == null) try out.append(a, p);
    }
    std.sort.pdq(u32, out.items, {}, std.sort.asc(u32));
    return out.items;
}

/// Every n-gram of the postings index: the suffix array's notes in each view are its postings exactly.
fn sameAsPostings(a: Allocator, post: *const index.Index, sx: *const index.Index) !usize {
    const b = sx.base.?;
    var bad: usize = 0;
    var it = post.grams.iterator();
    while (it.next()) |kv| {
        const p = kv.value_ptr.*;
        const views = [2][]const u32{ post.post[p.cur_off..][0..p.cur_len], post.post[p.hist_off..][0..p.hist_len] };
        for (views, [2]sa.Packed{ b.sa_cur, b.sa_hist }) |want, arr| {
            const sorted = try a.dupe(u32, want);
            std.sort.pdq(u32, sorted, {}, std.sort.asc(u32));
            if (!std.mem.eql(u32, sorted, try notesOf(a, b, arr, kv.key_ptr.*, post.settings.punct))) bad += 1;
        }
    }
    return bad;
}

test "every n-gram's notes by the suffix array are its postings, view by view, and the distinct n-grams the snapshot's" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try model_mod.parse(testing.allocator, notes_test.snapshot);
    defer m.deinit();
    const post = try index.Index.build(testing.allocator, m);
    defer post.deinit();
    const sx = try index.Index.buildKind(testing.allocator, m, .sa);
    defer sx.deinit();
    try testing.expect(post.grams.count() > 100);
    try testing.expectEqual(@as(usize, 0), try sameAsPostings(a, post, sx));
    const b = sx.base.?;
    const Keep = struct {
        fn wordy(idx: *const index.Index, key: u32) bool {
            for (idx.vkey, 0..) |k, t| if (k == key) return idx.vwordy[t];
            return false;
        }
    };
    try testing.expectEqual(@as(usize, m.grams_current), sa.distinctGrams(b, b.sa_cur, 1, 3, sx, Keep.wordy));
    try testing.expectEqual(@as(usize, m.grams_history), sa.distinctGrams(b, b.sa_hist, 1, 3, sx, Keep.wordy));
    // negative control: the 1-gram rule off (punctuation alone counted) gives more n-grams than the snapshot's
    const All = struct {
        fn any(_: *const index.Index, _: u32) bool {
            return true;
        }
    };
    try testing.expect(sa.distinctGrams(b, b.sa_cur, 1, 3, sx, All.any) > m.grams_current);
}

test "a note holding an n-gram twice counts once in its document frequency" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try model_mod.parse(testing.allocator, notes_test.snapshot);
    defer m.deinit();
    const post = try index.Index.build(testing.allocator, m);
    defer post.deinit();
    const sx = try index.Index.buildKind(testing.allocator, m, .sa);
    defer sx.deinit();
    const b = sx.base.?;
    // the n-grams some current note holds more than once: their ranges are longer than their notes
    var twice: usize = 0;
    var it = post.grams.iterator();
    while (it.next()) |kv| {
        const p = kv.value_ptr.*;
        if (p.cur_len == 0) continue;
        var keys: [index.max_n]u32 = undefined;
        var len: usize = 0;
        while (len < index.max_n) : (len += 1) {
            keys[len] = @truncate(kv.key_ptr.* >> @intCast(32 * len));
            if (keys[len] == index.none_key) break;
        }
        var buf: [index.max_n]u32 = undefined;
        const r = b.range(b.sa_cur, b.codes(keys[0..len], &buf).?);
        if (r[1] - r[0] > p.cur_len) {
            twice += 1;
            try testing.expectEqual(@as(usize, p.cur_len), (try notesOf(a, b, b.sa_cur, kv.key_ptr.*, false)).len);
        }
    }
    try testing.expect(twice > 10);
    // and the lookup counts notes: the fixture's answers above hold with these n-grams in the questions
    var text: Bytes = .{};
    const store = try openSa(text.text(), notes_test.snapshot);
    defer store.deinit();
    const got = try store.retrieve(a, text.text(), try text.text().encode(a, "My dog Luna and my cat: my my my.", false), false, 512, &.{});
    const want = blk: {
        const ps = try notes_route.Store.load(testing.allocator, testing.io, notes_test.snapshot, text.text(), @constCast(&@as([]const u8, "")));
        defer ps.deinit();
        const f = try ps.retrieve(a, text.text(), try text.text().encode(a, "My dog Luna and my cat: my my my.", false), false, 512, &.{});
        break :blk f;
    };
    try testing.expectEqualSlices(u32, want.ids, got.ids);
    try testing.expectEqualStrings(want.block, got.block);
}

test "a vocabulary the suffix array cannot match as the postings do is refused" {
    var text: Bytes = .{};
    // byte 'a' made a punctuation id of the same key as 'A': one key's ids disagree on being a word
    const s = notes_test.snapshot;
    const at = std.mem.indexOf(u8, s, "\"wordy\":\"").? + "\"wordy\":\"".len;
    const bad = try testing.allocator.dupe(u8, s);
    defer testing.allocator.free(bad);
    try testing.expect(bad[at + 'a'] == '1');
    bad[at + 'a'] = '0';
    try testing.expectError(error.Unsupported, openSa(text.text(), bad));
    // the postings index takes it
    var why: []const u8 = "";
    const ok = try notes_route.Store.load(testing.allocator, testing.io, bad, text.text(), &why);
    ok.deinit();
}

test "longer matches (off by default) weigh n-grams over n_max, from the base and from the side buffer" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rig = @import("notes_rig_test.zig");
    const ops = @import("notes_ops.zig");
    const r = try rig.Rig.initKind(testing.allocator, .sa, 1024);
    defer r.deinit();
    const s = r.store();
    const text = r.text.text();
    var why: ops.Refusal = .{};
    _ = try ops.perform(s, a, .{ .op = .add, .text = "Quentin keeps a xylophone in the attic." }, &why);
    const ix = s.current.idx;
    try testing.expect(ix.notes() > ix.base.?.notes); // the new note is in the side buffer
    const q = try text.encode(a, "Does Quentin keep a xylophone in the attic?", false);
    const off = try s.retrieve(a, text, q, false, 512, &.{});
    try testing.expectEqual(@as(u32, 0), off.long);
    for ([_]index.Long{ .{ .mode = .longest }, .{ .mode = .all, .max = 6 } }) |long| {
        s.long = long;
        const on = try s.retrieve(a, text, q, false, 512, &.{});
        try testing.expect(on.long > 0); // the byte tokenizer's n-grams: " xylophone in the attic" holds many over 3
        try testing.expectEqual(on.ids[on.ids.len - 1], off.ids[off.ids.len - 1]); // the written note is in both
        // the same after a merge, the note now in the base
        try s.merge();
        const merged = try s.retrieve(a, text, q, false, 512, &.{});
        try testing.expectEqual(on.long, merged.long);
        try testing.expectEqualStrings(on.block, merged.block);
    }
    s.long = .{}; // off again: the block it gave before (the merges changed no answer)
    const again = try s.retrieve(a, text, q, false, 512, &.{});
    try testing.expectEqual(@as(u32, 0), again.long);
    try testing.expectEqualStrings(off.block, again.block);
}
