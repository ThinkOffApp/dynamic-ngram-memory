//! Attribute-name-only question matching; no benchmark vocabulary or external schema cues.
const std = @import("std");

/// Whether a question contains a whole word from the stored attribute name.
pub fn matches(attribute: []const u8, question: []const u8) bool {
    var names = std.mem.tokenizeAny(u8, attribute, "_ -");
    while (names.next()) |name| {
        if (name.len == 0) continue;
        var words = std.mem.tokenizeAny(u8, question, " \t\r\n.,;:!?\"'()*_-=");
        while (words.next()) |word| if (std.ascii.eqlIgnoreCase(name, word)) return true;
    }
    return false;
}

test "question attributes use stored name words only and keep ambiguity" {
    try std.testing.expect(matches("eye_colour", "What is her eye colour?"));
    try std.testing.expect(matches("fav_colour", "What COLOUR?"));
    try std.testing.expect(matches("eye_colour", "What COLOUR?"));
    try std.testing.expect(!matches("eye_colour", "What color?")); // no unstored synonym
    try std.testing.expect(!matches("name", "Was it renamed?"));
    try std.testing.expect(!matches("home_city", "Where does she live?"));
}
