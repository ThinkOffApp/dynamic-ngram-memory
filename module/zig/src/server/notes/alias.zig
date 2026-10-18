//! Conservative name-value extraction; callers must require a current name fact.
const std = @import("std");

pub fn nameValue(subject: []const u8, text: []const u8) ?[]const u8 {
    if (text.len <= subject.len or !std.ascii.eqlIgnoreCase(text[0..subject.len], subject)) return null;
    const rest = text[subject.len..];
    var value: ?[]const u8 = null;
    for ([_][]const u8{ "'s name is ", " is named ", " is called " }) |prefix| {
        if (rest.len >= prefix.len and std.ascii.eqlIgnoreCase(rest[0..prefix.len], prefix)) {
            value = rest[prefix.len..];
            break;
        }
    }
    var v = std.mem.trim(u8, value orelse return null, " \t");
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

/// Build only from current structured name attributes. Ambiguous aliases are
/// disabled in the legacy unique map rather than selecting the last entity.
/// Retain all distinct candidates separately for the weighted resolver.
pub fn populate(idx: anytype, model: anytype, text: anytype) !void {
    const a = idx.arena.allocator();
    if (model.entityOf("me")) |entity| {
        for ([_][]const u8{ " I", " my", " me" }) |word| {
            const ids = text.encode(a, word, false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => continue,
            };
            defer a.free(ids);
            if (ids.len == 0 or ids.len > 8) continue;
            var keys: [8]u32 = @splat(0xffff_ffff);
            for (ids, keys[0..ids.len]) |id, *key| key.* = idx.keyOf(id);
            try idx.aliases.put(a, keys, entity);
            idx.alias_max = @max(idx.alias_max, ids.len);
        }
    }
    for (model.notes.items) |*n| {
        if (n.state != .current or n.kind != .fact) continue;
        if (!std.mem.eql(u8, model.attributes.items[n.attribute], "name")) continue;
        const value = nameValue(model.entities.items[n.entity], n.text()) orelse continue;
        const spaced = try std.mem.concat(a, u8, &.{ " ", value });
        defer a.free(spaced);
        const ids = text.encode(a, spaced, false) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => continue, // an unencodable alias must not break the note store
        };
        defer a.free(ids);
        if (ids.len == 0 or ids.len > 8) continue;
        var keys: [8]u32 = @splat(0xffff_ffff);
        for (ids, keys[0..ids.len]) |id, *key| key.* = idx.keyOf(id);
        const found = try idx.aliases.getOrPut(a, keys);
        if (!found.found_existing) found.value_ptr.* = n.entity else if (found.value_ptr.* != n.entity) found.value_ptr.* = 0xffff_ffff;
        const candidates = try idx.alias_entities.getOrPut(a, keys);
        if (!candidates.found_existing) candidates.value_ptr.* = .empty;
        if (std.mem.indexOfScalar(u32, candidates.value_ptr.items, n.entity) == null) try candidates.value_ptr.append(a, n.entity);
        idx.alias_max = @max(idx.alias_max, ids.len);
    }
}
