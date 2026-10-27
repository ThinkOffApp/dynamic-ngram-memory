//! Keep the pristine engine export separate from the standalone adapter/build glue.
const std = @import("std");
test "standalone core declarations" {
    std.testing.refAllDecls(@import("server/notes/model.zig"));
    std.testing.refAllDecls(@import("server/notes/index.zig"));
    std.testing.refAllDecls(@import("server/notes/journal.zig"));
}
