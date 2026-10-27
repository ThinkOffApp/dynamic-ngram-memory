//! Narrow POSIX-only replacement for TensorFold's platform module.
const std = @import("std");
const builtin = @import("builtin");
comptime {
    if (builtin.os.tag == .windows) @compileError("standalone Notes POSIX journal is not qualified on Windows");
}
pub const c = std.c;
