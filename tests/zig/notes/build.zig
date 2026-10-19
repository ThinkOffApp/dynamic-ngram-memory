//! zig build -Doptimize=ReleaseFast   builds zig-out/bin/notes-zig (the CLI) and the `notes` module
//! zig build test                       runs the unit tests
//! Zig 0.17.0. libc is linked: the weights use the C library's log, as CPython's math.log does.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("notes", .{
        .root_source_file = b.path("src/notes.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const exe = b.addExecutable(.{
        .name = "notes-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "notes", .module = mod }},
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run notes-zig").dependOn(&run_cmd.step);

    const tests = b.addTest(.{ .root_module = mod });
    b.step("test", "Run the unit tests").dependOn(&b.addRunArtifact(tests).step);
}
