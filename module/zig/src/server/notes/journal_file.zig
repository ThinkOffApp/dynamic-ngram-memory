//! The journal's files as found on disk: this server's user must own them, and no one else may reach them.
const std = @import("std");
const builtin = @import("builtin");
const libc = @import("sys").c;
const Allocator = std.mem.Allocator;

/// Tests only: the uid a check expects instead of this process's effective uid (an owner mismatch without root).
pub var test_euid: ?u32 = null;

/// Where a refusal's words are kept: one startup reads them before any other journal opens.
var said: [std.fs.max_path_bytes + 256]u8 = undefined;

/// Why the file open at `fd` (named `path`) is refused, or null. Windows has no POSIX owner or mode: never refused.
pub fn unsafe(fd: c_int, path: []const u8) ?[]const u8 {
    if (builtin.os.tag == .windows) return null;
    const st = ownerMode(fd) orelse return say("{s} cannot be inspected (fstat failed); refused", .{path});
    if (st.mode & std.c.S.IFMT != std.c.S.IFREG) return say("{s} is not a regular file; refused", .{path});
    const euid: u32 = test_euid orelse @intCast(std.c.geteuid());
    if (st.uid != euid)
        return say("{s} is owned by uid {d}, not by this server's uid {d}; refused: another local user could read or change notes", .{ path, st.uid, euid });
    if (st.mode & 0o077 != 0)
        return say("{s} has mode {o:0>3}, which lets group or other users reach it; refused (chmod 600 it if it is yours)", .{ path, st.mode & 0o777 });
    return null;
}

fn say(comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(&said, fmt, args) catch "a notes journal file is unsafe and its path too long to name; refused";
}

const OwnerMode = struct { uid: u32, mode: u32 };

/// Zig 0.17's std.c has no fstat on Linux: statx of the descriptor there, fstat elsewhere.
fn ownerMode(fd: c_int) ?OwnerMode {
    if (builtin.os.tag == .linux) {
        var x: std.os.linux.Statx = undefined;
        const want: std.os.linux.STATX = .{ .TYPE = true, .MODE = true, .UID = true };
        if (std.c.statx(fd, "", std.c.AT.EMPTY_PATH, want, &x) != 0 or !x.mask.TYPE or !x.mask.MODE or !x.mask.UID) return null;
        return .{ .uid = x.uid, .mode = x.mode };
    }
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return null;
    return .{ .uid = @intCast(st.uid), .mode = @intCast(st.mode) };
}

/// The whole file at `fd`, read with pread; past `limit` bytes it is refused as JournalTooLarge.
pub fn readAll(gpa: Allocator, fd: c_int, limit: u64, problem: *[]const u8) error{ OutOfMemory, JournalIo, JournalTooLarge }![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var buf: [65536]u8 = undefined;
    var at: i64 = 0;
    while (true) {
        const n = libc.pread(fd, &buf, buf.len, at);
        if (n < 0) {
            problem.* = "the notes journal cannot be read";
            return error.JournalIo;
        }
        if (n == 0) break;
        if (out.items.len + @as(usize, @intCast(n)) > limit) {
            problem.* = @import("limits.zig").journal_too_large;
            return error.JournalTooLarge;
        }
        try out.appendSlice(gpa, buf[0..@intCast(n)]);
        at += n;
    }
    return out.toOwnedSlice(gpa);
}
