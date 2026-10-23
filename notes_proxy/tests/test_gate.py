"""The chat-write gate against the TensorFold server's own cases (notes/question.zig and notes/intake.zig)."""
import unittest

from notes_proxy import gate

ASKS = [
    "What car does my sister drive", "\"Where do I live?\"", "(what's my name?)", "[Is Anna my sister?]",
    "My sister drives what car？", "My sister drives what car؟", "My sister drives what car⸮",
    "My sister drives what car;", "My sister drives what car?   \t", "My sister drives what car? \U0001F697",
    "My sister drives what car?\U0001F44D\U0001F3FD", "My sister drives what car ❓", "My sister drives what car?!",
    "¿Anna drives a Volvo", "tell me my sister's name", "Tell me about colours.", "Please tell me where I live.",
    "do you know where I live", "Remind me what car my sister drives", "Show me my notes.", "List my sister's cars",
    "Explain my schedule.", "who is my sister", "Which car is mine", "how old is Anna", "Is Anna my sister",
    "Are you sure", "Am I late", "Was it red", "Were we there", "Do I live in Helsinki", "Does Anna drive a Volvo",
    "Did I tell you about my car.", "Can you remember my car", "Could it be blue", "Will it rain",
    "Would my sister like it", "Should I buy a Volvo", "Shall we go", "May I ask", "Might it work",
    "Have I told you about Anna", "Has Anna called", "Had I known", "Whose car is that", "Whom did I meet",
    "When is my birthday", "Why did I sell my car", "Where's my car", "Isn't my car red", "Don't I live in Helsinki",
    "Doesn't Anna drive a Volvo", "Won't she come", "Can't I change it", "\U0001F697 what car does my sister drive",
]
TELLS = [
    "My sister is Anna.", "I live in Helsinki", "Anna drives a red Volvo", "yes, that's right", "Yes, that's right.",
    "This is my car.", "My favourite colour is green.", "Whoever calls, I am out.", "Listening to jazz relaxes me.",
    "Explanations bore me.", "Telling me twice helps.", "Showers wake me up.", "Isabel is my sister.",
    "Dogs are my favourite animals!", "I said \"why not\" to the dog.", "My car is red \U0001F697",
    "Anna asked me where I live.", "Whatever happens, I live in Helsinki.",
]


class Gate(unittest.TestCase):
    def test_counts_match_the_server(self):
        self.assertEqual((len(ASKS), len(TELLS)), (55, 18))   # question.zig's lists

    def test_every_question_and_request_is_caught(self):
        self.assertEqual([t for t in ASKS if gate.reason(t) is None], [])

    def test_every_statement_is_taken(self):
        self.assertEqual([(t, gate.reason(t)) for t in TELLS if gate.reason(t) is not None], [])

    def test_the_documented_limits(self):
        for t in ["Missä asun", "Wo wohne ich", "I wonder what my sister drives", "my sister drives what car",
                  "Give me my sister's name", "Help me remember my car"]:
            self.assertIsNone(gate.reason(t), t)
        for t in ["Will is my brother.", "May is my sister's name.", "Do remember that I like tea.",
                  "Have a look: my car is red.", "He asked \"why?\""]:
            self.assertIsNotNone(gate.reason(t), t)

    def test_reasons(self):
        self.assertEqual(gate.reason("\"Where do I live?\""), "a question: it ends in a question mark")
        self.assertEqual(gate.reason("What car does my sister drive"), "a question: it opens as one")
        self.assertEqual(gate.reason("Please, tell me my name"), "a request: it asks to be told something")

    def test_statement(self):
        self.assertEqual(gate.statement(" My favourite colour is green.\n"), ("My favourite colour is green.", None))
        self.assertEqual(gate.statement("Yes, that's right."), ("Yes, that's right.", None))
        self.assertEqual(gate.statement("What colour do I like?")[1], "a question: it ends in a question mark")
        self.assertEqual(gate.statement("My eyes are green? 　\n")[1], "a question: it ends in a question mark")
        self.assertEqual(gate.statement("What colour do I like")[1], "a question: it opens as one")
        self.assertEqual(gate.statement(" ... ")[1], "no words")
        self.assertEqual(gate.statement("\U0001F44D\U0001F3FD ❤️")[1], "no words")
        self.assertEqual(gate.statement("a\nb")[1], "more than one line: a note is one sentence")
        self.assertEqual(gate.statement("x" * 400), ("x" * 400, None))
        self.assertEqual(gate.statement("x" * 401)[1], "over 400 bytes: a note is one sentence")

    def test_control_the_old_rule_misses_them(self):
        """The trailing-"?" rule this gate replaces catches only one of the 55, so the lists can tell them apart."""
        old = lambda t: t.rstrip().endswith("?")
        self.assertEqual(sum(1 for t in ASKS if old(t)), 1)


if __name__ == "__main__":
    unittest.main()
