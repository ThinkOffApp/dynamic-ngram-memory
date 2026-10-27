//! Standalone CPU-only Notes core. No TensorFold host, engine, GPU, or model download.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const json = b.createModule(.{
        .root_source_file = b.path("src/server/json.zig"),
        .target = target,
        .optimize = optimize,
    });
    const sys = b.createModule(.{
        .root_source_file = b.path("standalone/sys.zig"),
        .target = target,
        .optimize = optimize,
    });
    const core = b.createModule(.{
        .root_source_file = b.path("src/core_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "json", .module = json }, .{ .name = "sys", .module = sys } },
    });
    const tests = b.addTest(.{ .root_module = core });
    b.step("test", "Run the standalone model/index/journal core tests").dependOn(&b.addRunArtifact(tests).step);
}
