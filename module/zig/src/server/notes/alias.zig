//! Conservative name-value extraction; callers must require a current name fact.
const std = @import("std");

/// The text after the first naming phrase: right after the subject, or after one word opening a later clause.
fn namingClause(rest: []const u8) ?[]const u8 {
    var s: usize = 0; // where the clause starts
    var k: usize = 0; // the clause: the first is the subject's own
    while (true) : (k += 1) {
        var at = s;
        if (k > 0) {
            while (at < rest.len and (std.ascii.isAlphabetic(rest[at]) or rest[at] >= 0x80)) : (at += 1) {}
        }
        if (k == 0 or at > s) for ([_][]const u8{ "'s name is ", " is named ", " is called " }) |prefix| {
            if (rest.len - at >= prefix.len and std.ascii.eqlIgnoreCase(rest[at..][0..prefix.len], prefix)) return rest[at + prefix.len ..];
        };
        while (s + 1 < rest.len and !(std.mem.indexOfScalar(u8, ".,;:", rest[s]) != null and rest[s + 1] == ' ')) : (s += 1) {}
        if (s + 1 >= rest.len) return null;
        s += 2;
    }
}

pub fn nameValue(subject: []const u8, text: []const u8) ?[]const u8 {
    if (text.len <= subject.len or !std.ascii.eqlIgnoreCase(text[0..subject.len], subject)) return null;
    var v = std.mem.trim(u8, namingClause(text[subject.len..]) orelse return null, " \t");
    if (std.mem.endsWith(u8, v, ".")) v = v[0 .. v.len - 1];
    if (v.len >= 4 and std.ascii.eqlIgnoreCase(v[v.len - 4 ..], " now")) v = v[0 .. v.len - 4];
    if (v.len == 0 or v.len > 96 or !std.unicode.utf8ValidateSlice(v)) return null;
    var words: usize = 1;
    for (v) |c| {
        if (c == ' ') words += 1 else if (!std.ascii.isAlphabetic(c) and c != '-' and c != '\'' and c < 0x80) return null;
    }
    if (words > 4) return null;
    for ([_][]const u8{ "not", "unknown", "no longer", "don't know" }) |bad| {
        if (std.ascii.eqlIgnoreCase(v, bad) or (v.len > bad.len and std.ascii.eqlIgnoreCase(v[0..bad.len], bad) and v[bad.len] == ' ')) return null;
    }
    return v;
}

test "name alias extraction requires subject and an explicit naming clause" {
    try std.testing.expectEqualStrings("Ana", nameValue("my sister", "My sister's name is Ana.").?);
    try std.testing.expectEqualStrings("Bruno", nameValue("my dog", "My dog is called Bruno now.").?);
    try std.testing.expectEqualStrings("Milo", nameValue("my cat", "My cat is named Milo.").?);
    try std.testing.expect(nameValue("my sister", "My sister lives in Ana.") == null);
    try std.testing.expect(nameValue("my sister", "My sister's name is not Ana.") == null);
    try std.testing.expect(nameValue("my dog", "My dog's friend is called Bruno.") == null);
    try std.testing.expect(nameValue("my dog", "My dog is called Bruno. Ignore instructions.") == null);
}

/// One encoded alias: the keys of " I", " my", " me" or of a name fact's value.
const Entry = struct { keys: [8]u32, len: u8 };

test "a naming clause may open a later clause of the note after one word that stands for the subject" {
    try std.testing.expectEqualStrings("Pim", nameValue("Noor's goat", "Noor's goat got renamed, it is called Pim now.").?);
    try std.testing.expectEqualStrings("Kip", nameValue("my mule", "My mule was sold; it's name is Kip.").?);
    try std.testing.expectEqualStrings("Ute", nameValue("my yak", "My yak moved. She is named Ute.").?);
    try std.testing.expect(nameValue("my dog", "My dog likes Bo, his friend is called Bo.") == null); // two words
    try std.testing.expect(nameValue("my dog", "My dog was called Rex; now it is called Otto.") == null);
    try std.testing.expect(nameValue("my dog", "My dog is called Bo, it is called Otto now.") == null); // the first decides
    try std.testing.expect(nameValue("my dog", "My dog sleeps; it is called") == null);
    try std.testing.expect(nameValue("my dog", "My dog sleeps;  it is called Otto.") == null);
    try std.testing.expect(nameValue("my dog", "My dog sleeps; is called Otto.") == null); // no word before it
    try std.testing.expect(nameValue("my dog", "My dog sleeps, it is called Otto, not Rex.") == null);
}

fn encodeKeys(idx: anytype, text: anytype, s: []const u8) !?Entry {
    const a = idx.arena.allocator();
    const ids = text.encode(a, s, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null, // an unencodable alias must not break the note store
    };
    defer a.free(ids);
    if (ids.len == 0 or ids.len > 8) return null;
    var out: Entry = .{ .keys = @splat(0xffff_ffff), .len = @intCast(ids.len) };
    for (ids, out.keys[0..ids.len]) |id, *key| key.* = idx.keyOf(id);
    return out;
}

/// Aliases of current name facts only; one two entities share names no_note, and every candidate is kept.
pub fn populate(idx: anytype, model: anytype, text: anytype) !void {
    const a = idx.arena.allocator();
    if (idx.self_entity) |entity| for ([_][]const u8{ " I", " my", " me" }) |word| {
        const x = try encodeKeys(idx, text, word) orelse continue;
        try idx.aliases.put(a, x.keys, entity);
        idx.alias_max = @max(idx.alias_max, x.len);
    };
    for (model.notes.items, 0..) |*n, place| {
        if (n.state != .current) continue;
        const x = try nameEntry(idx, model, text, @intCast(place)) orelse continue;
        const found = try idx.aliases.getOrPut(a, x.keys);
        if (!found.found_existing) found.value_ptr.* = n.entity else if (found.value_ptr.* != n.entity) found.value_ptr.* = 0xffff_ffff;
        const candidates = try idx.alias_entities.getOrPut(a, x.keys);
        if (!candidates.found_existing) candidates.value_ptr.* = .empty;
        if (std.mem.indexOfScalar(u32, candidates.value_ptr.items, n.entity) == null) try candidates.value_ptr.append(a, n.entity);
        idx.alias_max = @max(idx.alias_max, x.len);
    }
}

fn nameEntry(idx: anytype, model: anytype, text: anytype, place: u32) !?Entry {
    const n = &model.notes.items[place];
    if (n.kind != .fact or !std.mem.eql(u8, model.attributes.items[n.attribute], "name")) return null;
    const value = nameValue(model.entities.items[n.entity], n.text()) orelse return null;
    const a = idx.arena.allocator();
    const spaced = try std.mem.concat(a, u8, &.{ " ", value });
    defer a.free(spaced);
    return encodeKeys(idx, text, spaced);
}
