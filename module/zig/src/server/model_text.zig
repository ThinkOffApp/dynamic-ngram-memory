//! Standalone tokenizer seam: the caller supplies real token ids, never an engine or fallback tokenizer.
const std = @import("std");
pub const Error = error{ OutOfMemory, Tokenizer };
pub const Text = struct {
    ctx: *anyopaque,
    encode_fn: *const fn (*anyopaque, std.mem.Allocator, []const u8, bool) Error![]u32,
    vocabulary_size: u32,

    pub fn encode(t: Text, a: std.mem.Allocator, bytes: []const u8, special: bool) Error![]u32 {
        return t.encode_fn(t.ctx, a, bytes, special);
    }

    pub fn vocabSize(t: Text) u32 {
        return t.vocabulary_size;
    }
};
