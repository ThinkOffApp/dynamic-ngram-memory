//! Promotion's judge without an answer key: does a reply give the fact as its reference answer does.
const std = @import("std");
const wording = @import("../slide_words.zig");

/// Whether `reply` gives the fact as `ref` does: the reference's answer words, or a notable fact word ("Grey").
pub fn agreesWith(fact: []const u8, question: []const u8, ref: []const u8, reply: []const u8, others: []const []const u8) bool {
    if (hedged(reply)) return false; // "Not grey.", "Maybe grey.", "I don't know whether grey."
    if (wording.recalls(fact, question, ref, reply)) return true;
    // a terse reply ("Grey") only: a few words, and no word another note gives that this fact and question do not
    var words: usize = 0;
    var wi = std.mem.tokenizeAny(u8, reply, delims);
    while (wi.next()) |w| {
        words += 1;
        if (w.len < 3 or wording.isFiller(w) or hasStem(fact, w) or hasStem(question, w)) continue;
        for (others) |o| if (hasStem(o, w)) return false; // "grey or brown", "brown, not grey"
    }
    if (words > terse_words) return false;
    // value words (in reference and fact, not fillers, not the question's): names and numbers said, half the rest
    var n: usize = 0;
    var said: usize = 0;
    var first = true;
    var it = std.mem.tokenizeAny(u8, ref, delims);
    while (it.next()) |w| : (first = false) {
        if (w.len < 2 or wording.isFiller(w) or !hasStem(fact, w) or hasStem(question, w)) continue;
        const there = hasStem(reply, w);
        if ((!first and std.ascii.isUpper(w[0])) or std.ascii.isDigit(w[0])) if (!there) return false;
        n += 1;
        said += @intFromBool(there);
    }
    return n > 0 and 2 * said >= n;
}

/// Whether `reply` gives a value neither `before` nor the question has: a name, a number, or a stored note's word.
pub fn newValue(question: []const u8, before: []const u8, reply: []const u8, notes: []const []const u8) bool {
    // clause by clause: a denying clause asserts nothing; "I don't know, but your rabbit is Floppy." asserts its second
    var clauses = std.mem.tokenizeAny(u8, reply, ".!?;,");
    while (clauses.next()) |raw| {
        var parts = std.mem.splitSequence(u8, raw, " but ");
        while (parts.next()) |clause| {
            if (denies(clause)) continue;
            var it = std.mem.tokenizeAny(u8, clause, delims);
            var first = true;
            while (it.next()) |w| : (first = false) {
                if (std.ascii.isDigit(w[0]) and !hasStem(before, w) and !hasStem(question, w)) return true;
                if (w.len < 2 or wording.isFiller(w) or hasStem(before, w) or hasStem(question, w)) continue;
                if (!first and std.ascii.isUpper(w[0]) and !std.mem.eql(u8, w, "I")) return true;
                if (w.len >= 3) for (notes) |n| if (hasStem(n, w)) return true;
            }
        }
    }
    return false;
}

/// Whether `now` says every word of `before` (fillers aside, by stem): the same reply, at more length.
pub fn extends(before: []const u8, now: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, before, delims);
    var n: usize = 0;
    while (it.next()) |w| {
        if (w.len < 3 or wording.isFiller(w)) continue;
        n += 1;
        if (!hasStem(now, w)) return false;
    }
    return n > 0;
}

const delims = " \t\r\n.,;:!?\"'()*_";

/// A reply at most this many words long may agree by its value words alone.
const terse_words = 8;

/// Words that deny, doubt or offer a choice: a reply holding one does not give the fact as told.
const hedges = [_][]const u8{ "not", "no", "isn", "aren", "don", "doesn", "didn", "never", "maybe", "perhaps", "possibly", "probably", "might", "whether", "unsure", "or", "either", "neither", "nor" };

/// Words that deny: a clause holding one asserts no value (a doubt, "maybe green", still offers one).
const denials = [_][]const u8{ "not", "no", "isn", "aren", "don", "doesn", "didn", "never", "neither", "nor", "without" };

fn denies(clause: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, clause, delims);
    while (it.next()) |w| for (denials) |d| if (std.ascii.eqlIgnoreCase(w, d)) return true;
    return false;
}

fn hedged(reply: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, reply, delims);
    while (it.next()) |w| for (hedges) |h| if (std.ascii.eqlIgnoreCase(w, h)) return true;
    return false;
}

/// Whether `text` has a word of `w`'s stem (case folded; a plural s and the last letters of a long word aside).
fn hasStem(text: []const u8, w: []const u8) bool {
    var buf: [64]u8 = undefined;
    if (w.len > buf.len) return std.ascii.findIgnoreCase(text, w) != null;
    const low = std.ascii.lowerString(&buf, w);
    const base = if (low.len > 3 and low[low.len - 1] == 's') low[0 .. low.len - 1] else low;
    const stem = base[0..@min(base.len, @max(4, base.len -| 2))];
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n.,;:!?\"'()*_");
    while (it.next()) |t| if (t.len >= stem.len and std.ascii.eqlIgnoreCase(t[0..stem.len], stem)) return true;
    return false;
}
