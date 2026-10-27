//! This library owns no inference engine. Store erase cannot attest upstream cache clearance.
//! No engine token spans are retained in its journal: the proxy separately tracks child restart completion.
const std = @import("std");
const Text = @import("../server/model_text.zig").Text;
pub const Spans = struct { items: []const []const u32, short: bool };

pub fn discover(_: std.mem.Allocator, _: Text, _: []const []const u8) !Spans {
    return .{ .items = &.{}, .short = true };
}
