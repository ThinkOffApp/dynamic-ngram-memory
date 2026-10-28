//! Adapter-only v4 intake types. Not wired into the durable journal yet.
//! Validation is read-only; replay must validate the entire record before apply.
const std = @import("std");

pub const format = "tf-notes-journal/4";
pub const max_references = 100_000;
pub const Error = error{ OutOfMemory, InvalidSource, InvalidRecord, DuplicateReference, MissingSource, WrongOwner, WrongPair, NotCurrent, TooManyReferences };

pub const Verdict = enum { NEW, REPLACES, ALREADY_KNOWN, IMPLIED };
pub const Effect = union(enum) {
    learned: struct { id: u32, retire: []const []const u8 = &.{} },
    noop,
};

pub const Record = struct {
    source: []const u8,
    told: i64,
    subject: []const u8,
    attribute: []const u8,
    text: []const u8, // exact raw intake, including no-op intakes
    verdict: Verdict,
    depends: []const []const u8 = &.{},
    effect: Effect,
};

/// Resolver data belongs to the caller's prepared state, not this wire record.
pub const Known = struct {
    source: []const u8,
    owner: []const u8,
    subject: []const u8,
    attribute: []const u8,
    current: bool,
};

pub const Source = struct {
    kind: enum { seed, write },
    number: u64,

    pub fn parse(text: []const u8) Error!Source {
        if (text.len < 2 or text[1] == '0') return error.InvalidSource;
        const kind: @FieldType(Source, "kind") = switch (text[0]) {
            's' => .seed,
            'w' => .write,
            else => return error.InvalidSource,
        };
        for (text[1..]) |c| if (c < '0' or c > '9') return error.InvalidSource;
        const number = std.fmt.parseInt(u64, text[1..], 10) catch return error.InvalidSource;
        return .{ .kind = kind, .number = number };
    }
};

fn references(a: std.mem.Allocator, r: Record, refs: []const []const u8, owner: []const u8, known: *const std.StringHashMap(Known), retiring: bool) Error!void {
    if (refs.len > max_references) return error.TooManyReferences;
    const own = try Source.parse(r.source);
    var seen = std.StringHashMap(void).init(a);
    defer seen.deinit();
    for (refs) |source| {
        const ref = try Source.parse(source);
        // A write can depend only on an earlier source; seeds predate writes.
        if (ref.kind == .write and ref.number >= own.number) return error.InvalidSource;
        const inserted = try seen.getOrPut(source);
        if (inserted.found_existing) return error.DuplicateReference;
        const item = known.get(source) orelse return error.MissingSource;
        if (!std.mem.eql(u8, owner, item.owner)) return error.WrongOwner;
        if (retiring) {
            if (!item.current) return error.NotCurrent;
            if (!std.mem.eql(u8, r.subject, item.subject) or !std.mem.eql(u8, r.attribute, item.attribute)) return error.WrongPair;
        }
    }
}

pub fn validateWithAllocator(a: std.mem.Allocator, r: Record, owner: []const u8, known: []const Known) Error!void {
    const source = try Source.parse(r.source);
    if (source.kind != .write or owner.len == 0 or r.subject.len == 0 or r.attribute.len == 0 or r.text.len == 0) return error.InvalidRecord;
    var sources = std.StringHashMap(Known).init(a);
    defer sources.deinit();
    for (known) |item| {
        _ = try Source.parse(item.source);
        const inserted = try sources.getOrPut(item.source);
        if (inserted.found_existing) return error.DuplicateReference;
        inserted.value_ptr.* = item;
    }
    if (sources.contains(r.source)) return error.InvalidSource;
    switch (r.effect) {
        .learned => |learned| {
            if (learned.id == 0 or (r.verdict != .NEW and r.verdict != .REPLACES)) return error.InvalidRecord;
            if ((r.verdict == .REPLACES) != (learned.retire.len != 0)) return error.InvalidRecord;
            try references(a, r, learned.retire, owner, &sources, true);
        },
        .noop => {
            if (r.verdict != .ALREADY_KNOWN and r.verdict != .IMPLIED) return error.InvalidRecord;
            if (r.verdict == .ALREADY_KNOWN and r.depends.len == 0) return error.InvalidRecord;
        },
    }
    try references(a, r, r.depends, owner, &sources, false);
}

fn validate(r: Record, owner: []const u8, known: []const Known) Error!void {
    return validateWithAllocator(std.testing.allocator, r, owner, known);
}

const fixtures = [_]Known{
    .{ .source = "w1", .owner = "alice", .subject = "alice", .attribute = "likes", .current = true },
    .{ .source = "w2", .owner = "alice", .subject = "alice", .attribute = "likes", .current = true },
};

test "v4 explicit atomic retirement shape validates all targets without mutation" {
    var r: Record = .{ .source = "w3", .told = 3, .subject = "alice", .attribute = "likes", .text = "I now only like pears.", .verdict = .REPLACES, .effect = .{ .learned = .{ .id = 3, .retire = &.{ "w1", "w2" } } } };
    try validate(r, "alice", &fixtures);
    r.effect.learned.retire = &.{ "w1", "w9" };
    try std.testing.expectError(error.InvalidSource, validate(r, "alice", &fixtures));
    r.effect.learned.retire = &.{ "w1", "s9" };
    try std.testing.expectError(error.MissingSource, validate(r, "alice", &fixtures));
    r.effect.learned.retire = &.{ "w1", "w1" };
    try std.testing.expectError(error.DuplicateReference, validate(r, "alice", &fixtures));
    try std.testing.expect(fixtures[0].current and fixtures[1].current);
}

test "v4 no-op retains raw text and explicit erase dependencies without a note id" {
    var r: Record = .{ .source = "w3", .told = 3, .subject = "alice", .attribute = "likes", .text = "Yes, I still like apples!", .verdict = .ALREADY_KNOWN, .depends = &.{"w1"}, .effect = .noop };
    try validate(r, "alice", &fixtures);
    try std.testing.expectEqualStrings("Yes, I still like apples!", r.text);
    try std.testing.expectError(error.WrongOwner, validate(r, "bob", &fixtures));
    r.depends = &.{};
    try std.testing.expectError(error.InvalidRecord, validate(r, "alice", &fixtures));
    r.verdict = .IMPLIED;
    try validate(r, "alice", &fixtures);
}

test "v4 source and retirement constraints fail closed" {
    for ([_][]const u8{ "", "w0", "w01", "s0", "w+1", "w18446744073709551616", "x1" }) |bad| {
        try std.testing.expectError(error.InvalidSource, Source.parse(bad));
    }
    var r: Record = .{ .source = "w3", .told = 3, .subject = "alice", .attribute = "lives", .text = "Paris", .verdict = .REPLACES, .effect = .{ .learned = .{ .id = 3, .retire = &.{"w1"} } } };
    try std.testing.expectError(error.WrongPair, validate(r, "alice", &fixtures));
    var historical = fixtures;
    historical[0].current = false;
    r.attribute = "likes";
    try std.testing.expectError(error.NotCurrent, validate(r, "alice", &historical));
    r.effect.learned.id = 0;
    try std.testing.expectError(error.InvalidRecord, validate(r, "alice", &fixtures));
}
