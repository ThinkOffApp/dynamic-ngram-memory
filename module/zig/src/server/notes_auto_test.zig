//! A "notes": true chat request writes its last message through the intake when it is the user's statement, only then.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const rig = @import("notes_rig_test.zig");
const testing = std.testing;
const Value = json.Value;

/// A chat body from (role, content) pairs, with `extra` fields after "notes": true.
fn chat(a: std.mem.Allocator, msgs: []const [2][]const u8, extra: []const u8) ![]const u8 {
    var list: std.ArrayList(u8) = .empty;
    for (msgs, 0..) |m, k| try list.print(a, "{s}{{\"role\":\"{s}\",\"content\":{s}}}", .{ if (k > 0) "," else "", m[0], try json.quote(a, m[1], .{}) });
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":4,\"notes\":true{s},\"messages\":[{s}]}}", .{ extra, list.items });
}

fn ask(r: *rig.Rig, a: std.mem.Allocator, msgs: []const [2][]const u8, extra: []const u8) !Value {
    const got = try rig.request(r.srv, a, "POST", "/v1/chat/completions", try chat(a, msgs, extra));
    errdefer std.debug.print("{d} {s}\n", .{ got.status, got.body });
    try testing.expectEqual(@as(u16, 200), got.status);
    return got.value.get("notes").?;
}

/// Whether any note, or the journal, holds `text`.
fn stored(r: *rig.Rig, a: std.mem.Allocator, text: []const u8) !bool {
    for (r.store().model.notes.items) |*n| if (std.mem.indexOf(u8, n.text(), text) != null) return true;
    const j = r.journal(a) catch return false;
    return std.mem.indexOf(u8, j, text) != null;
}

test "a user's statement is written before the reply, and the next request's lookup finds it" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const said = try ask(r, a, &.{.{ "user", "My favourite colour is green." }}, ",\"notes_echo\":true");
    const w = said.get("write").?;
    try testing.expectEqualStrings("NEW", w.strField("verdict").?);
    try testing.expectEqual(@as(?i64, 30), w.get("id").?.int64());
    // its own lookup ran first: the block it got is the store before it
    try testing.expect(std.mem.indexOf(u8, said.strField("block") orelse "", "My favourite colour is green.") == null);
    try testing.expect(try stored(r, a, "My favourite colour is green."));
    const next = try ask(r, a, &.{ .{ "user", "My favourite colour is green." }, .{ "assistant", "Noted." }, .{ "user", "What is my favourite colour?" } }, ",\"notes_echo\":true");
    try testing.expect(std.mem.indexOf(u8, next.strField("block").?, "[30] \"My favourite colour is green.\"") != null);
    try testing.expectEqualStrings("a question: it ends in a question mark", next.get("write").?.strField("skipped").?);
    // said again: already known, nothing written
    const again = try ask(r, a, &.{.{ "user", "my favourite colour is green" }}, "");
    try testing.expectEqualStrings("ALREADY_KNOWN", again.get("write").?.strField("verdict").?);
    try testing.expectEqual(@as(usize, 30), r.store().model.notes.items.len);
    // a confirmation is the user's input too
    const yes = try ask(r, a, &.{ .{ "user", "Is my favourite colour green?" }, .{ "assistant", "Yes, it is green." }, .{ "user", "Yes, that's right." } }, "");
    try testing.expectEqualStrings("NEW", yes.get("write").?.strField("verdict").?);
    try r.restart();
    try testing.expect(try stored(r, a, "Yes, that's right."));
}

test "an assistant's message never becomes a note, nor a tool's, nor any message but the last user one" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const wrong = "Your favourite colour is purple.";
    // the conversation ends with the model's (wrong) answer: nothing is written
    const ends = try ask(r, a, &.{ .{ "user", "My cat is called Milo." }, .{ "assistant", wrong } }, "");
    try testing.expectEqualStrings("the last message is not the user's: only the user's own words are written", ends.get("write").?.strField("skipped").?);
    // the model's answer between two user messages: only the last user message is written
    const between = try ask(r, a, &.{ .{ "user", "My cat is called Milo." }, .{ "assistant", wrong }, .{ "user", "My dog is called Rex." } }, "");
    try testing.expectEqualStrings("NEW", between.get("write").?.strField("verdict").?);
    // a system message that states something is not the user's either
    _ = try ask(r, a, &.{ .{ "system", "The user's favourite colour is purple." }, .{ "user", "Tell me about colours." } }, "");
    for ([_][]const u8{ wrong, "purple", "My cat is called Milo." }) |t| try testing.expect(!try stored(r, a, t));
    try testing.expect(try stored(r, a, "My dog is called Rex."));
    try testing.expect(!try stored(r, a, "Tell me about colours.")); // a request, not a statement
    // control: the same check finds a text that was written, so it can fail
    try testing.expect(try stored(r, a, "Rex"));
}

test "no question or request a user asks in chat is written, and every statement still is" {
    const question = @import("notes/question.zig");
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const before = r.store().model.notes.items.len;
    for (question.asks) |q| {
        errdefer std.debug.print("written: {s}\n", .{q});
        const got = try ask(r, a, &.{.{ "user", q }}, "");
        try testing.expect(got.get("write").?.strField("skipped") != null);
    }
    // none of them: a revert to the trailing-"?" rule writes all but one of them here
    try testing.expectEqual(before, r.store().model.notes.items.len);
    try testing.expectError(error.FileNotFound, r.journal(a));
    var written: usize = 0;
    for (question.tells) |t| {
        errdefer std.debug.print("not taken: {s}\n", .{t});
        const got = try ask(r, a, &.{.{ "user", t }}, "");
        const v = got.get("write").?.strField("verdict") orelse return error.TestUnexpectedResult;
        if (std.mem.eql(u8, v, "NEW")) written += 1;
    }
    // every statement went through the intake: all new but the confirmation said twice (the same normal form)
    try testing.expectEqual(question.tells.len - 1, written);
    try testing.expectEqual(before + written, r.store().model.notes.items.len);
}

test "the off switches: --no-notes-write and \"notes_write\": false write nothing; the field needs notes on" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const off = try ask(r, a, &.{.{ "user", "My favourite colour is green." }}, ",\"notes_write\":false");
    try testing.expect(std.mem.indexOf(u8, off.get("write").?.strField("skipped").?, "notes_write") != null);
    r.store().auto_write = false;
    const flag = try ask(r, a, &.{.{ "user", "My favourite colour is green." }}, "");
    try testing.expect(std.mem.indexOf(u8, flag.get("write").?.strField("skipped").?, "--no-notes-write") != null);
    try testing.expect(!try stored(r, a, "My favourite colour is green."));
    r.store().auto_write = true;
    const bad = try rig.request(r.srv, a, "POST", "/v1/chat/completions", "{\"model\":\"m\",\"notes_write\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
    try testing.expectEqual(@as(u16, 400), bad.status);
    try testing.expectEqualStrings("notes_write needs \"notes\": true", rig.message(bad));
    // without "notes": true nothing is looked up or written
    const plain = try rig.request(r.srv, a, "POST", "/v1/chat/completions", "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"My favourite colour is green.\"}]}");
    try testing.expect(plain.value.get("notes") == null);
    try testing.expect(!try stored(r, a, "My favourite colour is green."));
}
