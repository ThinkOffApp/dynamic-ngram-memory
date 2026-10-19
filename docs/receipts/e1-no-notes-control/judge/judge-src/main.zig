//! E1 judge: the branch's own pure functions (3bf4e54f) over a JSONL file; one output line per input line.
const std = @import("std");
const judge = @import("notes/promote_judge.zig");
const wording = @import("slide_words.zig");

pub fn main(init: std.process.Init) !void {
    const aa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(aa);
    const input = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], aa, .limited(1 << 30));
    var out: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, input, '\n');
    const Rec = struct { fact: []const u8, q: []const u8, ref: []const u8, short: []const u8, reply: []const u8, others: []const []const u8 };
    while (it.next()) |l| {
        if (l.len == 0) continue;
        const r = try std.json.parseFromSliceLeaky(Rec, aa, l, .{});
        const a = judge.agreesWith(r.fact, r.q, r.ref, r.reply, r.others);
        const b = wording.recalls(r.fact, r.q, r.ref, r.reply);
        const c = wording.recallsShort(r.fact, r.q, r.short, r.reply);
        try out.print(aa, "{d} {d} {d}\n", .{ @intFromBool(a), @intFromBool(b), @intFromBool(c) });
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = out.items });
}
