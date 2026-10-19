/-
  LwLogic.lean: the logic of Living Weights tests, in Lean 4 (core library only, no Mathlib).

  STATUS: CHECKED with Lean 4.35.0 (pinned in lean-toolchain), 13 Oct 2026. The file compiles with no errors,
  no warnings and no `sorry`, and every theorem rests only on Lean's three standard axioms (propext,
  Classical.choice, Quot.sound). The definitions mirror zengine/lwlogic.py name for name. The theorems are about
  these Lean definitions; that the Python copy gives the same answers is checked on test sets (1216 checks for the
  nine example sets, 750 of them on level 6), not proved. At its first compile one proof needed a fix (dup_lesson:
  name the two writes given to dup_write); every other proof compiled as written. The level 6 part (two-step
  questions, 30 theorems) was added on 13 Oct 2026; it compiled at the first try, with four warnings about the
  renamed if_pos and if_neg, which were then cleared.

  To check, with elan installed (the toolchain installs itself on first use), from the repository root:
      python3 zengine/lwlogic.py leancheck
  which runs, and reads the output of:
      lean lean/LwLogic.lean
      cat lean/LwLogic.lean examples/L3-seed1.check.lean > /tmp/c.lean && lean /tmp/c.lean
  The second command makes Lean's kernel recompute every expected answer of a generated test set.

  93 theorems, all proved. The main ones:
    delete_is_never_taught   DELETE(i) on a log = the log without lesson i (+ delete_answers, delete_comm)
    locality                 lessons about other (entity, attribute) pairs never change an answer
    swap_independent         two adjacent lessons, at most one on the asked key, can be swapped
    step_idem                every effect, hints included, is idempotent
    dup_lesson               a plain present-tense lesson taught twice in a row = once
    dup_change, dup_retract  the same for "now v" and "not v anymore", which carry a hint (told later)
    dated_screens_off        a DATED lesson answers its own questions whatever was taught (or deleted) before it
    previous_screened        the same for PREVIOUS and ALWAYS, which also read the boundary list
    dup_dated                teaching a DATED lesson twice in a row = once
    add_comm                 two "I like X" lessons on a multi-valued attribute commute (state level)
    adds_commute_store       the same in a store (via adds_commute_stateAt: at every point, the whole state)
    functional_stateAt       a single-valued attribute never holds two values at once, hints included
    yesNo_from_summary       YES_NO follows from CURRENT
    eitherOr_from_summary    EITHER_OR follows from CURRENT
    whoHas_from_current      WHO_HAS follows from CURRENT
    empty_store_unknown      nothing taught: every value unknown
  Level 6, two-step questions (chainSum, linkedState, kAnd, kOr):
    kAnd_no_iff, kOr_yes_iff, kAnd_yes_iff, kOr_no_iff   Kleene's truth tables: an unknown part never hides a no in
                             "and" or a yes in "or"
    deMorgan_and, deMorgan_or, neg_neg, kleene_classical, kleene_mono   De Morgan; ordinary logic on known parts;
                             an unknown part that becomes known never turns a yes or a no of the whole around
    chain_known_iff          a two-step chain is known exactly when both links are known
    chain_value_iff          its value is the second link's value, read on the entity the first link names
    chain_link_unknown       a first link that names no entity of the set gives "don't know"
    delete_link_unknown      deleting the only lesson on a chain's first link makes the chain "don't know"
    named_chain_unknown, namedEnt_untouched   a name nobody holds (all name lessons deleted) points at nobody
    deep_delete_implied      deleting what was said outright keeps the answer the kept lessons imply through a link
    linked_fills_gap, own_wins   a link fills gaps only; what is said outright wins
    functional_linked        a linked single-valued answer never holds two values, contradictions included
    natSum_value_iff         the natural step gives a country only from a known city
  The rest are helper lemmas. Three `example`s after adds_commute_store are controls: Lean's kernel evaluates
  small cases that show the hypotheses of previous_screened and adds_commute_store matter. Four more at the end
  evaluate small chains: the owner's eyes, the same after a change of owner, "don't know" without the owner
  lesson, and a link that fills a gap.
  Not stated here, checked in Python only: idempotence of dated-interval asserts ("since D", "until D"),
  and that the fold equals an independently written timeline.

  Assumptions, stated once:
    A1  A store is the list of lessons still in force, in the order they were taught.
    A2  Time is an integer tick. A date D is the tick D * 10^6; a lesson told on day T gets a tick inside T,
        later than every date its text mentions. Questions about dates use the start of the day.
    A3  "until D" means up to the start of D; "since D" means from the start of D.
    A4  Where lessons overlap, the later lesson wins (it is applied later in the fold).
    A5  A present-tense lesson holds from the moment it is told. It says nothing about earlier dates.
    A6  Each attribute is single-valued or multi-valued, fixed by the schema, with a finite value list.
    A7  Truth is three-valued: yes, no, unknown. Nothing ever said is unknown, never no.
    A8  Wordings that describe a change also say what held on its other side: "now v" (single-valued) = not v
        just before it was said; "not v anymore" = v just before; "since D" = not v just before D;
        "until D" = not v from D on. These hints only fill gaps: a hint acts only where the value is still
        unknown, so it never overrides anything a lesson says outright.
    A9  Level 6. A chain asks about now. A name points at the one entity called that now. A linking fact
        ("My cat's eyes are the same colour as mine.") is a lesson on its own attribute, read one way and one link
        deep; like a hint it only fills gaps. Compound yes/no questions use Kleene's strong three-valued logic.
        The natural step (the country of a city) is fixed general knowledge in the schema.
-/

namespace LwLogic

abbrev Entity := Nat
abbrev Attr := Nat
abbrev Val := Nat

inductive Truth where
  | yes | no | unk
  deriving DecidableEq, Repr

def Truth.neg : Truth → Truth
  | .yes => .no
  | .no => .yes
  | .unk => .unk

/-- Half-open interval [lo, hi) of ticks; `none` means unbounded on that side. -/
structure Interval where
  lo : Option Int
  hi : Option Int
  deriving DecidableEq, Repr

/-- A point in time: a tick, or "now" (after every tick any lesson mentions). -/
inductive TPoint where
  | pt (t : Int)
  | now
  deriving DecidableEq, Repr

def lowerOk : Option Int → Int → Bool
  | none, _ => true
  | some l, t => decide (l ≤ t)

def upperOk : Option Int → Int → Bool
  | none, _ => true
  | some h, t => decide (t < h)

def Interval.covers (iv : Interval) : TPoint → Bool
  | .pt t => lowerOk iv.lo t && upperOk iv.hi t
  | .now => iv.hi.isNone

inductive Effect where
  | set (v : Val)         -- the value is v and nothing else
  | add (v : Val)         -- v is one of the values (multi-valued); on a single-valued attribute = set
  | retract (v : Val)     -- v no longer holds
  | clear                 -- no value holds
  | presume (v : Val)     -- hint: v held here, if nothing says otherwise (fills a gap only)
  | presumeNot (v : Val)  -- hint: v did not hold here, if nothing says otherwise (fills a gap only)
  deriving DecidableEq, Repr

structure Write where
  e : Entity
  a : Attr
  iv : Interval
  eff : Effect
  deriving DecidableEq, Repr

/-- The lesson language. `assert` with `none` is present tense; with an interval it is "since", "until",
    "between" or "always". `change` is "now it is v" (on a multi-valued attribute: "now only v").
    Hints (A8): "now v" also says it was not v just before; "not v anymore" says it was v just before;
    "since D" says it was not v just before D; "until D" says it is not v from D on. -/
inductive Op where
  | assert (e : Entity) (a : Attr) (v : Val) (iv : Option Interval)
  | change (e : Entity) (a : Attr) (v : Val)
  | retract (e : Entity) (a : Attr) (v : Option Val)
  | dated (e : Entity) (a : Attr) (v0 : Val) (d : Int) (v1 : Val)
  deriving DecidableEq, Repr

structure Lesson where
  id : Nat
  told : Int
  op : Op
  deriving DecidableEq, Repr

structure Schema where
  multi : Attr → Bool
  dom : Attr → List Val
  /-- Level 6 only; the defaults switch each one off, so the sets of levels 1 to 5 never set them.
      valEnt: the entity a value names (the owner "my sister" is the entity my sister). -/
  valEnt : Val → Option Entity := fun _ => none
  /-- The linking attribute of an attribute: "My cat's eyes are the same colour as mine." is the lesson
      (my cat, eyesLike, me), and linkOf eyeColour = some eyesLike. -/
  linkOf : Attr → Option Attr := fun _ => none
  /-- Fixed general knowledge, the natural step of a chain: the country of each city. -/
  natMap : Val → Option Val := fun _ => none
  /-- The attribute that holds names, for "What colour are Nora's eyes?". -/
  nameAttr : Option Attr := none

abbrev State := Val → Truth

def blank : State := fun _ => .unk

def Op.writes (σ : Schema) (told : Int) : Op → List Write
  | .assert e a v none => [⟨e, a, ⟨some told, none⟩, if σ.multi a then .add v else .set v⟩]
  | .assert e a v (some iv) =>
      ⟨e, a, iv, if σ.multi a then .add v else .set v⟩ ::
        ((match iv.lo with
          | some d => [⟨e, a, ⟨some (d - 1), some d⟩, .presumeNot v⟩]
          | none => []) ++
         (match iv.hi with
          | some d => [⟨e, a, ⟨some d, none⟩, .presumeNot v⟩]
          | none => []))
  | .change e a v =>
      (if σ.multi a then [] else [⟨e, a, ⟨some (told - 1), some told⟩, .presumeNot v⟩])
        ++ [⟨e, a, ⟨some told, none⟩, .set v⟩]
  | .retract e a (some v) =>
      [⟨e, a, ⟨some (told - 1), some told⟩, .presume v⟩, ⟨e, a, ⟨some told, none⟩, .retract v⟩]
  | .retract e a none => [⟨e, a, ⟨some told, none⟩, .clear⟩]
  | .dated e a v0 d v1 => [⟨e, a, ⟨none, some d⟩, .set v0⟩, ⟨e, a, ⟨some d, none⟩, .set v1⟩]

def Lesson.writes (σ : Schema) (l : Lesson) : List Write := l.op.writes σ l.told

def step (multi : Bool) : Effect → State → State
  | .set v, _ => fun x => if x = v then .yes else .no
  | .add v, s =>
      if multi then (fun x => if x = v then .yes else s x) else (fun x => if x = v then .yes else .no)
  | .retract v, s =>
      if multi then (fun x => if x = v then .no else s x)
      else if s v = .yes then (fun x => if x = v then .no else .unk)
      else (fun x => if x = v then .no else s x)
  | .clear, _ => fun _ => .no
  | .presume v, s =>
      if s v = .unk then
        (if multi then (fun x => if x = v then .yes else s x) else (fun x => if x = v then .yes else .no))
      else s
  | .presumeNot v, s => if s v = .unk then (fun x => if x = v then .no else s x) else s

def keyMatch (e : Entity) (a : Attr) (w : Write) : Bool := w.e == e && w.a == a

def hit (e : Entity) (a : Attr) (p : TPoint) (w : Write) : Bool := keyMatch e a w && w.iv.covers p

/-- The writes of a store that bear on (e, a) at p, in teaching order. -/
def relevant (σ : Schema) (e : Entity) (a : Attr) (p : TPoint) : List Lesson → List Write
  | [] => []
  | l :: S => (l.writes σ).filter (hit e a p) ++ relevant σ e a p S

def fold (m : Bool) (s : State) (ws : List Write) : State := ws.foldl (fun s w => step m w.eff s) s

/-- The oracle's core: the truth of every value of (e, a) at p. -/
def stateAt (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) (p : TPoint) : State :=
  fold (σ.multi a) blank (relevant σ e a p S)

-- ------------------------------------------------------------------------------------------- answers

def memN (i : Nat) : List Nat → Bool
  | [] => false
  | x :: xs => i == x || memN i xs

def firstYes (s : State) : List Val → Option Val
  | [] => none
  | v :: vs => if s v = .yes then some v else firstYes s vs

def allNo (s : State) : List Val → Bool
  | [] => true
  | v :: vs => decide (s v = .no) && allNo s vs

def withTruth (s : State) (t : Truth) : List Val → List Val
  | [] => []
  | v :: vs => if s v = t then v :: withTruth s t vs else withTruth s t vs

inductive Summary where
  | value (v : Val)
  | none
  | unknown (excl : List Val)
  | set (yes no : List Val)
  deriving DecidableEq, Repr

def summarize (multi : Bool) (dom : List Val) (s : State) : Summary :=
  match firstYes s dom with
  | some v => if multi then .set (withTruth s .yes dom) (withTruth s .no dom) else .value v
  | none => if allNo s dom then .none else .unknown (withTruth s .no dom)

def sumAt (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) (p : TPoint) : Summary :=
  summarize (σ.multi a) (σ.dom a) (stateAt σ S e a p)

/-- The yes/no answer about one value that a summary gives. -/
def deriveYN : Summary → Val → Truth
  | .value w, v => if v = w then .yes else .no
  | .none, _ => .no
  | .unknown ex, v => if memN v ex then .no else .unk
  | .set ys ns, v => if memN v ys then .yes else if memN v ns then .no else .unk

inductive Choice where
  | pick (v : Val)
  | neither
  | unknown
  deriving DecidableEq, Repr

def eitherOr (s : State) (v1 v2 : Val) : Choice :=
  if s v1 = .yes then .pick v1
  else if s v2 = .yes then .pick v2
  else if s v1 = .no ∧ s v2 = .no then .neither
  else .unknown

def boundsOf : List Write → List Int
  | [] => []
  | w :: ws => (match w.iv.lo with | some l => [l] | none => [])
      ++ (match w.iv.hi with | some h => [h] | none => []) ++ boundsOf ws

def boundaries (σ : Schema) (e : Entity) (a : Attr) : List Lesson → List Int
  | [] => []
  | l :: S => boundsOf ((l.writes σ).filter (keyMatch e a)) ++ boundaries σ e a S

def maxOpt : List Int → Option Int
  | [] => none
  | x :: xs => match maxOpt xs with
    | none => some x
    | some m => some (max x m)

def Summary.known : Summary → Bool
  | .unknown _ => false
  | _ => true

/-- "What was it before it changed?": the latest piece below the current run that differs from now and is
    known (a value, none or a set), skipping unknown gaps ("blue; not blue anymore; red now" gives blue).
    If no earlier piece is known, the piece just below the current run (for example "not green"). -/
def previous (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) : Summary :=
  let cur := sumAt σ S e a .now
  let below := fun (b : Int) => sumAt σ S e a (.pt (b - 1))
  let cands := (boundaries σ e a S).filter (fun b => decide (below b ≠ cur))
  match maxOpt (cands.filter (fun b => (below b).known)) with
  | some b => below b
  | none =>
    match maxOpt cands with
    | some b => below b
    | none => .unknown []

def conflicts : Summary → Summary → Bool
  | .value v, .value w => !(v == w)
  | .value _, .none => true
  | .value v, .unknown ex => memN v ex
  | .none, .value _ => true
  | .unknown ex, .value v => memN v ex
  | _, _ => false

def always (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) : Truth :=
  let cur := sumAt σ S e a .now
  let ps := cur :: (boundaries σ e a S).map (fun b => sumAt σ S e a (.pt (b - 1)))
  let allSame : Bool := match cur with
    | .value _ => ps.all (fun q => decide (q = cur))
    | _ => false
  if allSame then .yes
  else if ps.any (fun p => ps.any (fun q => conflicts p q)) then .no
  else .unk

def whoHas (σ : Schema) (S : List Lesson) (ents : List Entity) (a : Attr) (v : Val) : List Entity :=
  ents.filter (fun e => decide (stateAt σ S e a .now v = .yes))

-- --------------------------------------------------------------------- two-step questions (level 6)
-- A chain walks from a start to an answer: "What colour are the eyes of the owner of the boat?" reads the boat's
-- owner, then that person's eyes. A chain asks about now. A name points at the one entity called that now. A link is
-- read one way and one link deep, and only fills gaps, like a hint (A8): what is said outright wins.

/-- Kleene's strong three-valued logic: a no decides "and", a yes decides "or", whatever the other part. -/
def kAnd : Truth → Truth → Truth
  | .no, _ => .no
  | _, .no => .no
  | .yes, .yes => .yes
  | _, _ => .unk

def kOr : Truth → Truth → Truth
  | .yes, _ => .yes
  | _, .yes => .yes
  | .no, .no => .no
  | _, _ => .unk

/-- The entity a known value names, if it names one of the set. -/
def entOf (σ : Schema) : Summary → Option Entity
  | .value v => σ.valEnt v
  | _ => none

/-- A link fills the gaps of e's own state from the linked entity's state; a value known about e itself wins. -/
def fill (own src : State) : State := fun x => if own x = .unk then src x else own x

/-- e's state of a now, with the gaps filled through e's link ("My cat's eyes are the same colour as mine."). -/
def linkedState (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) : State :=
  match σ.linkOf a with
  | none => stateAt σ S e a .now
  | some la =>
    match entOf σ (sumAt σ S e la .now) with
    | none => stateAt σ S e a .now
    | some t => fill (stateAt σ S e a .now) (stateAt σ S t a .now)

/-- Where a chain starts: an entity ("the boat"), or a name ("Nora", the one entity called Nora now). -/
inductive Start where
  | ent (e : Entity)
  | named (v : Val)
  deriving DecidableEq, Repr

/-- The one entity called v now; nobody, or more than one, gives none. -/
def namedEnt (σ : Schema) (S : List Lesson) (ents : List Entity) (v : Val) : Option Entity :=
  match σ.nameAttr with
  | none => none
  | some na =>
    match ents.filter (fun e => decide (stateAt σ S e na .now v = .yes)) with
    | [e] => some e
    | _ => none

def startEnt (σ : Schema) (S : List Lesson) (ents : List Entity) : Start → Option Entity
  | .ent e => some e
  | .named v => namedEnt σ S ents v

/-- Follow the hops (attributes whose values name entities, such as the owner) from an entity, now. -/
def hopsEnt (σ : Schema) (S : List Lesson) : Option Entity → List Attr → Option Entity
  | none, _ => none
  | some e, [] => some e
  | some e, h :: hs => hopsEnt σ S (entOf σ (sumAt σ S e h .now)) hs

/-- The natural step on a summary ("Oslo" gives "Norway"); anything but a known value stays unknown. -/
def natSum (σ : Schema) : Summary → Summary
  | .value w =>
    match σ.natMap w with
    | some c => .value c
    | none => .unknown []
  | .none => .none
  | _ => .unknown []

/-- The answer to a chain question: start, every hop, then the final attribute (through its link), and the
    natural step when `nat` is set. -/
def chainSum (σ : Schema) (S : List Lesson) (ents : List Entity) (st : Start) (hops : List Attr) (a : Attr)
    (nat : Bool) : Summary :=
  match hopsEnt σ S (startEnt σ S ents st) hops with
  | none => .unknown []
  | some e =>
    if nat then natSum σ (summarize (σ.multi a) (σ.dom a) (linkedState σ S e a))
    else summarize (σ.multi a) (σ.dom a) (linkedState σ S e a)

inductive Question where
  | current (e : Entity) (a : Attr)
  | atT (e : Entity) (a : Attr) (t : Int)        -- AT(date): t is the date's tick
  | before (e : Entity) (a : Attr) (t : Int)     -- BEFORE(date): evaluated at t - 1
  | previous (e : Entity) (a : Attr)
  | always (e : Entity) (a : Attr)
  | changed (e : Entity) (a : Attr)
  | yesNo (e : Entity) (a : Attr) (v : Val)
  | eitherOr (e : Entity) (a : Attr) (v1 v2 : Val)
  | whoHas (a : Attr) (v : Val)
  | chain (st : Start) (hops : List Attr) (a : Attr) (nat : Bool)              -- level 6: a two-step question
  | chainYN (st : Start) (hops : List Attr) (a : Attr) (nat : Bool) (v : Val)  -- its yes/no form: is it v?
  | kand (p q : Question)                                                      -- compound yes/no questions
  | kor (p q : Question)
  | knot (p : Question)
  deriving DecidableEq, Repr

inductive Answer where
  | summary (s : Summary)
  | yn (t : Truth)
  | choice (c : Choice)
  | ents (es : List Entity)
  deriving DecidableEq, Repr

/-- The truth a yes/no answer gives; any other answer counts as unknown. -/
def Answer.truth : Answer → Truth
  | .yn t => t
  | _ => .unk

/-- The oracle: the one right answer to a question, given the lessons in force. -/
def answer (σ : Schema) (ents : List Entity) (S : List Lesson) : Question → Answer
  | .current e a => .summary (sumAt σ S e a .now)
  | .atT e a t => .summary (sumAt σ S e a (.pt t))
  | .before e a t => .summary (sumAt σ S e a (.pt (t - 1)))
  | .previous e a => .summary (previous σ S e a)
  | .always e a => .yn (always σ S e a)
  | .changed e a => .yn (always σ S e a).neg
  | .yesNo e a v => .yn (stateAt σ S e a .now v)
  | .eitherOr e a v1 v2 => .choice (eitherOr (stateAt σ S e a .now) v1 v2)
  | .whoHas a v => .ents (whoHas σ S ents a v)
  | .chain st hops a nat => .summary (chainSum σ S ents st hops a nat)
  | .chainYN st hops a nat v => .yn (deriveYN (chainSum σ S ents st hops a nat) v)
  | .kand p q => .yn (kAnd (answer σ ents S p).truth (answer σ ents S q).truth)
  | .kor p q => .yn (kOr (answer σ ents S p).truth (answer σ ents S q).truth)
  | .knot p => .yn (answer σ ents S p).truth.neg

-- ------------------------------------------------------------------------------------------- the log

inductive Entry where
  | teach (l : Lesson)
  | delete (id : Nat)

def lessonsOf : List Entry → List Lesson
  | [] => []
  | .teach l :: r => l :: lessonsOf r
  | .delete _ :: r => lessonsOf r

def deletedOf : List Entry → List Nat
  | [] => []
  | .teach _ :: r => deletedOf r
  | .delete i :: r => i :: deletedOf r

/-- The lessons in force after a log of teaches and deletes. -/
def effective (log : List Entry) : List Lesson :=
  (lessonsOf log).filter (fun l => !(memN l.id (deletedOf log)))

-- =========================================================================================== lemmas

theorem filter_filter' {α : Type} (p q : α → Bool) :
    ∀ (l : List α), (l.filter q).filter p = l.filter (fun x => q x && p x)
  | [] => rfl
  | x :: xs => by
    cases hq : q x <;> cases hp : p x <;> simp [hq, hp, filter_filter' p q xs]

theorem fold_append (m : Bool) (s : State) (xs ys : List Write) :
    fold m s (xs ++ ys) = fold m (fold m s xs) ys := by
  simp [fold, List.foldl_append]

theorem relevant_append (σ : Schema) (e : Entity) (a : Attr) (p : TPoint) :
    ∀ (S T : List Lesson), relevant σ e a p (S ++ T) = relevant σ e a p S ++ relevant σ e a p T
  | [], T => by simp [relevant]
  | l :: S, T => by simp [relevant, relevant_append σ e a p S T, List.append_assoc]

def touches (σ : Schema) (e : Entity) (a : Attr) (l : Lesson) : Bool := (l.writes σ).any (keyMatch e a)

theorem filter_hit_nil (e : Entity) (a : Attr) (p : TPoint) :
    ∀ (ws : List Write), ws.any (keyMatch e a) = false → ws.filter (hit e a p) = []
  | [], _ => rfl
  | w :: ws, h => by
    simp only [List.any_cons, Bool.or_eq_false_iff] at h
    have h2 : hit e a p w = false := by simp [hit, h.1]
    simp [h2, filter_hit_nil e a p ws h.2]

theorem relevant_nil_of (σ : Schema) (e : Entity) (a : Attr) (p : TPoint) :
    ∀ (T : List Lesson), (∀ l ∈ T, touches σ e a l = false) → relevant σ e a p T = []
  | [], _ => rfl
  | l :: T, h => by
    have h1 : (l.writes σ).filter (hit e a p) = [] := filter_hit_nil e a p (l.writes σ) (h l (by simp))
    have h2 := relevant_nil_of σ e a p T (fun x hx => h x (List.mem_cons_of_mem _ hx))
    simp only [relevant, h1, h2, List.nil_append]

-- ======================================================================================== invariants

theorem lessonsOf_append : ∀ (x y : List Entry), lessonsOf (x ++ y) = lessonsOf x ++ lessonsOf y
  | [], y => rfl
  | .teach l :: x, y => by simp [lessonsOf, lessonsOf_append x y]
  | .delete _ :: x, y => by simp [lessonsOf, lessonsOf_append x y]

theorem deletedOf_append : ∀ (x y : List Entry), deletedOf (x ++ y) = deletedOf x ++ deletedOf y
  | [], y => rfl
  | .teach _ :: x, y => by simp [deletedOf, deletedOf_append x y]
  | .delete i :: x, y => by simp [deletedOf, deletedOf_append x y]

theorem memN_append (i : Nat) : ∀ (xs ys : List Nat), memN i (xs ++ ys) = (memN i xs || memN i ys)
  | [], ys => by simp [memN]
  | x :: xs, ys => by simp [memN, memN_append i xs ys, Bool.or_assoc]

/-- DELETE(i) appended to a log gives exactly the lessons of the log without lesson i:
    deleting a lesson is the same as never having taught it. -/
theorem delete_is_never_taught (log : List Entry) (i : Nat) :
    effective (log ++ [.delete i]) = (effective log).filter (fun l => !(l.id == i)) := by
  simp only [effective, lessonsOf_append, deletedOf_append, lessonsOf, deletedOf, List.append_nil,
    memN_append, memN, Bool.or_false, Bool.not_or, filter_filter']

/-- Corollary: any question gets the same answer after DELETE(i) as from the store without lesson i. -/
theorem delete_answers (σ : Schema) (ents : List Entity) (log : List Entry) (i : Nat) (q : Question) :
    answer σ ents (effective (log ++ [.delete i])) q
      = answer σ ents ((effective log).filter (fun l => !(l.id == i))) q := by
  rw [delete_is_never_taught]

theorem delete_comm (log : List Entry) (i j : Nat) :
    effective (log ++ [.delete i, .delete j]) = effective (log ++ [.delete j, .delete i]) := by
  have hi := delete_is_never_taught (log ++ [.delete i]) j
  have hj := delete_is_never_taught (log ++ [.delete j]) i
  simp only [List.append_assoc, List.singleton_append] at hi hj
  rw [hi, hj, delete_is_never_taught, delete_is_never_taught, filter_filter', filter_filter']
  congr 1
  funext l
  exact Bool.and_comm _ _

/-- Locality: only lessons that touch (e, a) matter for (e, a). -/
theorem relevant_filter_touches (σ : Schema) (e : Entity) (a : Attr) (p : TPoint) :
    ∀ (S : List Lesson), relevant σ e a p (S.filter (touches σ e a)) = relevant σ e a p S
  | [] => rfl
  | l :: S => by
    cases ht : touches σ e a l
    · have hnil : (l.writes σ).filter (hit e a p) = [] := filter_hit_nil e a p (l.writes σ) ht
      simp [ht, relevant, hnil, relevant_filter_touches σ e a p S]
    · simp [ht, relevant, relevant_filter_touches σ e a p S]

theorem locality (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) (p : TPoint) :
    stateAt σ (S.filter (touches σ e a)) e a p = stateAt σ S e a p := by
  simp only [stateAt, relevant_filter_touches]

/-- Order: two adjacent lessons can be swapped when at most one of them touches the asked key. -/
theorem swap_independent (σ : Schema) (S T : List Lesson) (l1 l2 : Lesson) (e : Entity) (a : Attr)
    (p : TPoint) (h : touches σ e a l1 = false ∨ touches σ e a l2 = false) :
    stateAt σ (S ++ l1 :: l2 :: T) e a p = stateAt σ (S ++ l2 :: l1 :: T) e a p := by
  simp only [stateAt, relevant_append, relevant]
  rcases h with h | h
  · simp [filter_hit_nil e a p (l1.writes σ) h]
  · simp [filter_hit_nil e a p (l2.writes σ) h]

/-- Every effect is idempotent. -/
theorem step_idem (m : Bool) (eff : Effect) (s : State) : step m eff (step m eff s) = step m eff s := by
  cases eff with
  | set v => rfl
  | add v =>
    cases m
    · rfl
    · funext x; by_cases hx : x = v <;> simp [step, hx]
  | retract v =>
    cases m
    · by_cases hs : s v = .yes
      · funext x; by_cases hx : x = v <;> simp [step, hs, hx]
      · funext x; by_cases hx : x = v <;> simp [step, hs, hx]
    · funext x; by_cases hx : x = v <;> simp [step, hx]
  | clear => rfl
  | presume v =>
    by_cases hs : s v = .unk
    · cases m <;> simp [step, hs]
    · simp [step, hs]
  | presumeNot v =>
    by_cases hs : s v = .unk
    · simp [step, hs]
    · simp [step, hs]

/-- Two "I like X" lessons on a multi-valued attribute commute. -/
theorem add_comm (v w : Val) (s : State) :
    step true (.add v) (step true (.add w) s) = step true (.add w) (step true (.add v) s) := by
  funext x
  by_cases hv : x = v <;> by_cases hw : x = w <;> simp [step, hv, hw]

theorem fold_skip (m : Bool) (s : State) (A X Y B : List Write)
    (h : ∀ s, fold m (fold m s X) Y = fold m s X) :
    fold m s (A ++ (X ++ (Y ++ B))) = fold m s (A ++ (X ++ B)) := by
  simp only [fold_append, h]

theorem dup_write (m : Bool) (e : Entity) (a : Attr) (p : TPoint) (w w' : Write)
    (heff : w'.eff = w.eff) (hkey : keyMatch e a w' = keyMatch e a w)
    (hcov : w'.iv.covers p = true → w.iv.covers p = true) (s : State) :
    fold m (fold m s ([w].filter (hit e a p))) ([w'].filter (hit e a p)) = fold m s ([w].filter (hit e a p)) := by
  cases h' : hit e a p w' with
  | false => simp [List.filter_cons, h', fold]
  | true =>
    have hk : keyMatch e a w' = true ∧ w'.iv.covers p = true := by simpa [hit] using h'
    have h : hit e a p w = true := by
      simp only [hit, Bool.and_eq_true]
      exact ⟨hkey ▸ hk.1, hcov hk.2⟩
    simp [h, h', fold, heff, step_idem]

/-- Present-tense lessons that make exactly one write, from the moment they are told (no hint). -/
def Op.plain (σ : Schema) : Op → Bool
  | .assert _ _ _ none => true
  | .retract _ _ none => true
  | .change _ a _ => σ.multi a
  | _ => false

theorem plain_writes (σ : Schema) (op : Op) (hp : op.plain σ = true) :
    ∃ (e : Entity) (a : Attr) (eff : Effect), ∀ told, op.writes σ told = [⟨e, a, ⟨some told, none⟩, eff⟩] := by
  cases op with
  | assert e a v iv =>
    cases iv with
    | none => exact ⟨e, a, if σ.multi a then .add v else .set v, fun _ => rfl⟩
    | some iv => simp [Op.plain] at hp
  | change e a v =>
    have hm : σ.multi a = true := by simpa [Op.plain] using hp
    exact ⟨e, a, .set v, fun _ => by simp [Op.writes, hm]⟩
  | retract e a v =>
    cases v with
    | none => exact ⟨e, a, .clear, fun _ => rfl⟩
    | some v => simp [Op.plain] at hp
  | dated e a v0 d v1 => simp [Op.plain] at hp

theorem cover_mono (t t' : Int) (ht : t ≤ t') (p : TPoint) :
    (Interval.mk (some t') none).covers p = true → (Interval.mk (some t) none).covers p = true := by
  cases p with
  | pt x =>
    simp only [Interval.covers, lowerOk, upperOk, Bool.and_true, decide_eq_true_eq]
    intro hx
    exact Int.le_trans ht hx
  | now => intro h; exact h

/-- Idempotence: the same plain present-tense lesson taught again right after itself changes nothing. -/
theorem dup_lesson (σ : Schema) (S T : List Lesson) (l l' : Lesson) (e : Entity) (a : Attr) (p : TPoint)
    (hop : l'.op = l.op) (hplain : l.op.plain σ = true) (ht : l.told ≤ l'.told) :
    stateAt σ (S ++ l :: l' :: T) e a p = stateAt σ (S ++ l :: T) e a p := by
  obtain ⟨e0, a0, eff, hw⟩ := plain_writes σ l.op hplain
  have hw1 : l.writes σ = [⟨e0, a0, ⟨some l.told, none⟩, eff⟩] := hw l.told
  have hw2 : l'.writes σ = [⟨e0, a0, ⟨some l'.told, none⟩, eff⟩] := by
    show l'.op.writes σ l'.told = _
    rw [hop]
    exact hw l'.told
  simp only [stateAt, relevant_append, relevant, hw1, hw2]
  apply fold_skip
  intro s
  exact dup_write _ e a p ⟨e0, a0, ⟨some l.told, none⟩, eff⟩ ⟨e0, a0, ⟨some l'.told, none⟩, eff⟩ rfl rfl
    (cover_mono _ _ ht p) s

/-- A hint never acts where the value is already known. -/
theorem presume_noop (m : Bool) (v : Val) (s : State) (h : s v ≠ .unk) : step m (.presume v) s = s := by
  simp [step, h]

theorem presumeNot_noop (m : Bool) (v : Val) (s : State) (h : s v ≠ .unk) : step m (.presumeNot v) s = s := by
  simp [step, h]

/-- After "not v anymore" v is no, so the hint "it was v just before" of the same lesson said again is idle. -/
theorem presume_after_retract (m : Bool) (v : Val) (s : State) :
    step m (.presume v) (step m (.retract v) s) = step m (.retract v) s := by
  apply presume_noop
  have h : step m (.retract v) s v = .no := by
    cases m
    · by_cases hs : s v = .yes <;> simp [step, hs]
    · simp [step]
  rw [h]
  decide

/-- After "now v" v is yes, so the hint "it was not v just before" of the same lesson said again is idle. -/
theorem presumeNot_after_set (m : Bool) (v : Val) (s : State) :
    step m (.presumeNot v) (step m (.set v) s) = step m (.set v) s := by
  apply presumeNot_noop
  simp [step]

/-- The two writes of a lesson that carries a hint: the hint on the tick just before t, then the main write. -/
def hinted (e0 : Entity) (a0 : Attr) (h eff : Effect) (t : Int) : List Write :=
  [⟨e0, a0, ⟨some (t - 1), some t⟩, h⟩, ⟨e0, a0, ⟨some t, none⟩, eff⟩]

/-- Where the second copy (told at t' > t) writes anything, the first copy's main write already covers the
    point, and that main write makes the second copy's hint idle and its main write a repeat. -/
theorem dup_hinted_writes (m : Bool) (e : Entity) (a : Attr) (p : TPoint) (e0 : Entity) (a0 : Attr)
    (h eff : Effect) (t t' : Int) (htt : t < t')
    (hhm : ∀ s, step m h (step m eff s) = step m eff s) (s : State) :
    fold m (fold m s ((hinted e0 a0 h eff t).filter (hit e a p))) ((hinted e0 a0 h eff t').filter (hit e a p))
      = fold m s ((hinted e0 a0 h eff t).filter (hit e a p)) := by
  by_cases he : e0 = e
  · by_cases ha : a0 = a
    · subst he
      subst ha
      cases p with
      | now =>
        simp [hinted, hit, keyMatch, Interval.covers, fold, step_idem]
      | pt x =>
        by_cases h1 : t' ≤ x
        · have h2 : t ≤ x := by omega
          have h3 : ¬ x < t := by omega
          have h4 : ¬ x < t' := by omega
          have h5 : t - 1 ≤ x := by omega
          have h6 : t' - 1 ≤ x := by omega
          simp [hinted, hit, keyMatch, Interval.covers, lowerOk, upperOk, fold,
            h1, h2, h3, h4, h5, h6, step_idem]
        · by_cases h7 : t' - 1 ≤ x
          · have h2 : t ≤ x := by omega
            have h3 : ¬ x < t := by omega
            have h4 : x < t' := by omega
            have h5 : t - 1 ≤ x := by omega
            simp [hinted, hit, keyMatch, Interval.covers, lowerOk, upperOk, fold,
              h1, h2, h3, h4, h5, h7, hhm]
          · simp [hinted, List.filter_cons, hit, keyMatch, Interval.covers, lowerOk, upperOk, fold, h1, h7]
    · simp [hinted, hit, keyMatch, fold, ha]
  · simp [hinted, hit, keyMatch, fold, he]

/-- Idempotence for lessons with a hint, told again later (l.told < l'.told). -/
theorem dup_hinted (σ : Schema) (S T : List Lesson) (l l' : Lesson) (e : Entity) (a : Attr) (p : TPoint)
    (e0 : Entity) (a0 : Attr) (h eff : Effect)
    (hw : ∀ t, l.op.writes σ t = hinted e0 a0 h eff t) (hop : l'.op = l.op) (ht : l.told < l'.told)
    (hhm : ∀ s, step (σ.multi a) h (step (σ.multi a) eff s) = step (σ.multi a) eff s) :
    stateAt σ (S ++ l :: l' :: T) e a p = stateAt σ (S ++ l :: T) e a p := by
  have hw1 : l.writes σ = hinted e0 a0 h eff l.told := hw l.told
  have hw2 : l'.writes σ = hinted e0 a0 h eff l'.told := by
    show l'.op.writes σ l'.told = _
    rw [hop]
    exact hw l'.told
  simp only [stateAt, relevant_append, relevant, hw1, hw2]
  apply fold_skip
  intro s
  exact dup_hinted_writes _ e a p e0 a0 h eff _ _ ht hhm s

/-- "Now it is v" said twice on a single-valued attribute = once. -/
theorem dup_change (σ : Schema) (S T : List Lesson) (l l' : Lesson) (e : Entity) (a : Attr) (p : TPoint)
    (e0 : Entity) (a0 : Attr) (v : Val) (hl : l.op = .change e0 a0 v) (hm : σ.multi a0 = false)
    (hop : l'.op = l.op) (ht : l.told < l'.told) :
    stateAt σ (S ++ l :: l' :: T) e a p = stateAt σ (S ++ l :: T) e a p :=
  dup_hinted σ S T l l' e a p e0 a0 (.presumeNot v) (.set v)
    (fun t => by rw [hl]; simp [Op.writes, hm, hinted]) hop ht (fun s => presumeNot_after_set _ v s)

/-- "v not anymore" said twice = once. -/
theorem dup_retract (σ : Schema) (S T : List Lesson) (l l' : Lesson) (e : Entity) (a : Attr) (p : TPoint)
    (e0 : Entity) (a0 : Attr) (v : Val) (hl : l.op = .retract e0 a0 (some v))
    (hop : l'.op = l.op) (ht : l.told < l'.told) :
    stateAt σ (S ++ l :: l' :: T) e a p = stateAt σ (S ++ l :: T) e a p :=
  dup_hinted σ S T l l' e a p e0 a0 (.presume v) (.retract v)
    (fun t => by rw [hl]; rfl) hop ht (fun s => presume_after_retract _ v s)

theorem fold_const_of_sets (m : Bool) : ∀ (F : List Write) (s s' : State),
    F ≠ [] → (∀ w ∈ F, ∃ v, w.eff = .set v) → fold m s F = fold m s' F
  | [], _, _, hne, _ => absurd rfl hne
  | [w], s, s', _, hs => by
    obtain ⟨v, hv⟩ := hs w (List.mem_singleton.mpr rfl)
    simp [fold, hv, step]
  | w :: w2 :: ws, s, s', _, hs => by
    have ih := fold_const_of_sets m (w2 :: ws) (step m w.eff s) (step m w.eff s') (List.cons_ne_nil _ _)
      (fun x hx => hs x (List.mem_cons_of_mem _ hx))
    simp only [fold, List.foldl_cons] at ih ⊢
    exact ih

theorem dated_sets (σ : Schema) (told : Int) (e : Entity) (a : Attr) (v0 : Val) (d : Int) (v1 : Val)
    (p : TPoint) : ∀ w ∈ ((Op.dated e a v0 d v1).writes σ told).filter (hit e a p), ∃ v, w.eff = .set v := by
  intro w hw
  have hw' := (List.mem_filter.mp hw).1
  simp [Op.writes] at hw'
  rcases hw' with rfl | rfl
  · exact ⟨v0, rfl⟩
  · exact ⟨v1, rfl⟩

theorem dated_nonempty (σ : Schema) (told : Int) (e : Entity) (a : Attr) (v0 : Val) (d : Int) (v1 : Val)
    (p : TPoint) : ((Op.dated e a v0 d v1).writes σ told).filter (hit e a p) ≠ [] := by
  cases p with
  | now => simp [Op.writes, hit, keyMatch, Interval.covers]
  | pt t =>
    by_cases h : t < d
    · simp [Op.writes, hit, keyMatch, Interval.covers, lowerOk, upperOk, h]
    · simp [Op.writes, hit, keyMatch, Interval.covers, lowerOk, upperOk, h,
        Int.not_lt.mp h]

/-- The #678 property. A DATED lesson ("were v0 until d; now v1") fixes every state of its key, so its
    answers do not depend on anything taught before it: deleting fact 4 must not break fact 5.
    S1 and S2 are any two histories (for example S and S with some lessons deleted); T holds the
    lessons taught after it, none on the same key. -/
theorem dated_screens_off (σ : Schema) (S1 S2 T : List Lesson) (l : Lesson)
    (e : Entity) (a : Attr) (v0 v1 : Val) (d : Int) (p : TPoint)
    (hl : l.op = .dated e a v0 d v1) (hT : ∀ l' ∈ T, touches σ e a l' = false) :
    stateAt σ (S1 ++ l :: T) e a p = stateAt σ (S2 ++ l :: T) e a p := by
  have hTnil : relevant σ e a p T = [] := relevant_nil_of σ e a p T hT
  have hlw : l.writes σ = (Op.dated e a v0 d v1).writes σ l.told := by
    simp only [Lesson.writes, hl]
  simp only [stateAt, relevant_append, relevant, hTnil, List.append_nil, fold_append]
  rw [hlw]
  exact fold_const_of_sets _ _ _ _ (dated_nonempty σ l.told e a v0 d v1 p) (dated_sets σ l.told e a v0 d v1 p)

/-- Idempotence for DATED lessons, from the screening property. -/
theorem dup_dated (σ : Schema) (S T : List Lesson) (l l' : Lesson) (e : Entity) (a : Attr)
    (v0 v1 : Val) (d : Int) (p : TPoint) (hl : l.op = .dated e a v0 d v1) (hop : l'.op = l.op)
    (hT : ∀ l'' ∈ T, touches σ e a l'' = false) :
    stateAt σ (S ++ l :: l' :: T) e a p = stateAt σ (S ++ l :: T) e a p := by
  have h1 := dated_screens_off σ (S ++ [l]) S T l' e a v0 v1 d p (hop.trans hl) hT
  have hsame : ∀ q : TPoint, relevant σ e a q (l' :: T) = relevant σ e a q (l :: T) := by
    intro q
    simp only [relevant, Lesson.writes, hop, hl, Op.writes]
  simp only [List.append_assoc, List.singleton_append] at h1
  rw [h1]
  simp only [stateAt, relevant_append, hsame]

-- ------------------------------------------------------------- PREVIOUS and ALWAYS behind a DATED lesson

/-- `previous` with its two inputs made parameters: the summary at each point, and the boundary list.
    The body is the same, so `previous_eq` holds by `rfl`. -/
def previousOf (f : TPoint → Summary) (B : List Int) : Summary :=
  let cur := f .now
  let below := fun (b : Int) => f (.pt (b - 1))
  let cands := B.filter (fun b => decide (below b ≠ cur))
  match maxOpt (cands.filter (fun b => (below b).known)) with
  | some b => below b
  | none =>
    match maxOpt cands with
    | some b => below b
    | none => .unknown []

theorem previous_eq (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) :
    previous σ S e a = previousOf (sumAt σ S e a) (boundaries σ e a S) := rfl

/-- `always` with the same two inputs made parameters. -/
def alwaysOf (f : TPoint → Summary) (B : List Int) : Truth :=
  let cur := f .now
  let ps := cur :: B.map (fun b => f (.pt (b - 1)))
  let allSame : Bool := match cur with
    | .value _ => ps.all (fun q => decide (q = cur))
    | _ => false
  if allSame then .yes
  else if ps.any (fun p => ps.any (fun q => conflicts p q)) then .no
  else .unk

theorem always_eq (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) :
    always σ S e a = alwaysOf (sumAt σ S e a) (boundaries σ e a S) := rfl

theorem maxOpt_mem : ∀ (xs : List Int) (m : Int), maxOpt xs = some m → m ∈ xs
  | [], m, h => by simp [maxOpt] at h
  | x :: xs, m, h => by
    simp only [maxOpt] at h
    split at h
    · cases h
      exact List.mem_cons_self
    · rename_i m' hm'
      cases h
      rcases Int.le_total x m' with hle | hle
      · rw [Int.max_eq_right hle]
        exact List.mem_cons_of_mem _ (maxOpt_mem xs m' hm')
      · rw [Int.max_eq_left hle]
        exact List.mem_cons_self

theorem maxOpt_none : ∀ (xs : List Int), maxOpt xs = none → xs = []
  | [], _ => rfl
  | _ :: _, h => by
    simp only [maxOpt] at h
    split at h <;> cases h

/-- PREVIOUS returns the piece below some boundary that differs from now, or unknown when none does.
    So if every piece that differs from now is the same R, and one exists, PREVIOUS is R. -/
theorem previousOf_some (f : TPoint → Summary) (B : List Int) (R : Summary)
    (hR : ∀ b ∈ B, f (.pt (b - 1)) ≠ f .now → f (.pt (b - 1)) = R)
    (hex : ∃ b ∈ B, f (.pt (b - 1)) ≠ f .now) : previousOf f B = R := by
  have hc : ∀ b, b ∈ B.filter (fun b => decide (f (.pt (b - 1)) ≠ f .now)) → f (.pt (b - 1)) = R := by
    intro b hb
    have hb' := List.mem_filter.mp hb
    exact hR b hb'.1 (of_decide_eq_true hb'.2)
  simp only [previousOf]
  split
  · rename_i b hb
    exact hc b (List.mem_filter.mp (maxOpt_mem _ b hb)).1
  · split
    · rename_i b hb
      exact hc b (maxOpt_mem _ b hb)
    · rename_i hn
      obtain ⟨b, hb, hne⟩ := hex
      have hmem : b ∈ B.filter (fun b => decide (f (.pt (b - 1)) ≠ f .now)) :=
        List.mem_filter.mpr ⟨hb, decide_eq_true hne⟩
      rw [maxOpt_none _ hn] at hmem
      cases hmem

/-- ... and if every piece equals now, PREVIOUS is unknown. -/
theorem previousOf_none (f : TPoint → Summary) (B : List Int)
    (hall : ∀ b ∈ B, f (.pt (b - 1)) = f .now) : previousOf f B = .unknown [] := by
  have hnil : B.filter (fun b => decide (f (.pt (b - 1)) ≠ f .now)) = [] := by
    rw [List.filter_eq_nil_iff]
    intro b hb h
    exact of_decide_eq_true h (hall b hb)
  simp only [previousOf, hnil, List.filter_nil, maxOpt]

theorem all_congr' {α : Type} (xs ys : List α) (h : ∀ x, x ∈ xs ↔ x ∈ ys) (p : α → Bool) :
    xs.all p = ys.all p := by
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true]
  exact ⟨fun H x hx => H x ((h x).mpr hx), fun H x hx => H x ((h x).mp hx)⟩

theorem any_congr' {α : Type} (xs ys : List α) (h : ∀ x, x ∈ xs ↔ x ∈ ys) (p : α → Bool) :
    xs.any p = ys.any p := by
  apply Bool.eq_iff_iff.mpr
  simp only [List.any_eq_true]
  exact ⟨fun ⟨x, hx, hp⟩ => ⟨x, (h x).mp hx, hp⟩, fun ⟨x, hx, hp⟩ => ⟨x, (h x).mpr hx, hp⟩⟩

/-- ALWAYS looks only at which pieces occur, not at how often or in which order. -/
theorem alwaysOf_congr (f : TPoint → Summary) (B1 B2 : List Int)
    (h : ∀ x, x ∈ f .now :: B1.map (fun b => f (.pt (b - 1))) ↔ x ∈ f .now :: B2.map (fun b => f (.pt (b - 1)))) :
    alwaysOf f B1 = alwaysOf f B2 := by
  have hall := all_congr' _ _ h
  have hany := any_congr' _ _ h
  simp only [alwaysOf, hall, hany]

theorem boundaries_append (σ : Schema) (e : Entity) (a : Attr) :
    ∀ (S T : List Lesson), boundaries σ e a (S ++ T) = boundaries σ e a S ++ boundaries σ e a T
  | [], T => by simp [boundaries]
  | l :: S, T => by simp [boundaries, boundaries_append σ e a S T, List.append_assoc]

/-- A DATED lesson on its own, at a tick before its date: the same state as just before the date. -/
theorem dated_before (σ : Schema) (l : Lesson) (e : Entity) (a : Attr) (v0 v1 : Val) (d x : Int)
    (hl : l.op = .dated e a v0 d v1) (hx : x < d) :
    stateAt σ [l] e a (.pt x) = stateAt σ [l] e a (.pt (d - 1)) := by
  have h1 : d - 1 < d := by omega
  have h2 : ¬ d ≤ x := by omega
  have h3 : ¬ d ≤ d - 1 := by omega
  simp [stateAt, relevant, Lesson.writes, hl, Op.writes, hit, keyMatch, Interval.covers, lowerOk, upperOk,
    hx, h1, h2, h3]

/-- ... and at a tick on or after its date: the same state as now. -/
theorem dated_after (σ : Schema) (l : Lesson) (e : Entity) (a : Attr) (v0 v1 : Val) (d x : Int)
    (hl : l.op = .dated e a v0 d v1) (hx : ¬ x < d) :
    stateAt σ [l] e a (.pt x) = stateAt σ [l] e a .now := by
  have h2 : d ≤ x := by omega
  simp [stateAt, relevant, Lesson.writes, hl, Op.writes, hit, keyMatch, Interval.covers, lowerOk, upperOk,
    hx, h2]

/-- The screening property also holds for PREVIOUS and ALWAYS. They read the boundary list too, and
    boundaries from earlier lessons differ between S1 and S2. But every piece they look at is one of the
    DATED lesson's two pieces (v0 before d, v1 from d on), and d itself is a boundary, so both stores see the
    same pieces and give the same answers. -/
theorem previous_screened (σ : Schema) (S1 S2 T : List Lesson) (l : Lesson)
    (e : Entity) (a : Attr) (v0 v1 : Val) (d : Int)
    (hl : l.op = .dated e a v0 d v1) (hT : ∀ l' ∈ T, touches σ e a l' = false) :
    previous σ (S1 ++ l :: T) e a = previous σ (S2 ++ l :: T) e a ∧
    always σ (S1 ++ l :: T) e a = always σ (S2 ++ l :: T) e a := by
  -- every summary of the key, at every point, is the DATED lesson's own
  let g := sumAt σ [l] e a
  have hs : ∀ S, sumAt σ (S ++ l :: T) e a = g := by
    intro S
    funext p
    have h := dated_screens_off σ S [] T l e a v0 v1 d p hl hT
    simp only [List.nil_append] at h
    show summarize _ _ (stateAt σ (S ++ l :: T) e a p) = summarize _ _ (stateAt σ [l] e a p)
    rw [h]
    simp only [stateAt, relevant, relevant_nil_of σ e a p T hT]
  -- every piece below a boundary is the piece before d or the piece from d on
  let P0 := g (.pt (d - 1))
  let P1 := g .now
  have hpiece : ∀ b : Int, g (.pt (b - 1)) = P0 ∨ g (.pt (b - 1)) = P1 := by
    intro b
    by_cases hb : b - 1 < d
    · left
      simp only [g, P0, sumAt, dated_before σ l e a v0 v1 d (b - 1) hl hb]
    · right
      simp only [g, P1, sumAt, dated_after σ l e a v0 v1 d (b - 1) hl hb]
  have hd : ∀ S, d ∈ boundaries σ e a (S ++ l :: T) := by
    intro S
    simp [boundaries_append, boundaries, Lesson.writes, hl, Op.writes, keyMatch, boundsOf]
  have hprev : ∀ S, previous σ (S ++ l :: T) e a = if P0 = P1 then .unknown [] else P0 := by
    intro S
    rw [previous_eq, hs S]
    by_cases h01 : P0 = P1
    · rw [ite_eq_left h01]
      apply previousOf_none
      intro b _
      rcases hpiece b with h | h
      · rw [h, h01]
      · exact h
    · rw [ite_eq_right h01]
      apply previousOf_some
      · intro b _ hne
        rcases hpiece b with h | h
        · exact h
        · exact absurd h hne
      · exact ⟨d, hd S, by show P0 ≠ P1; exact h01⟩
  have hmem : ∀ S x, x ∈ g .now :: (boundaries σ e a (S ++ l :: T)).map (fun b => g (.pt (b - 1)))
      ↔ x = P0 ∨ x = P1 := by
    intro S x
    simp only [List.mem_cons, List.mem_map]
    constructor
    · rintro (h | ⟨b, _, hb⟩)
      · exact Or.inr h
      · rw [← hb]
        exact hpiece b
    · rintro (h | h)
      · exact Or.inr ⟨d, hd S, h.symm⟩
      · exact Or.inl h
  refine ⟨by rw [hprev S1, hprev S2], ?_⟩
  rw [always_eq, always_eq, hs S1, hs S2]
  exact alwaysOf_congr g _ _ (fun x => (hmem S1 x).trans (hmem S2 x).symm)

-- -------------------------------------------------------------------- "I like X" lessons in any order

theorem fold_swap (m : Bool) (s : State) (A X Y B : List Write)
    (h : ∀ s, fold m (fold m s X) Y = fold m (fold m s Y) X) :
    fold m s (A ++ (X ++ (Y ++ B))) = fold m s (A ++ (Y ++ (X ++ B))) := by
  simp only [fold_append, h]

theorem add_writes (σ : Schema) (l : Lesson) (e : Entity) (a : Attr) (v : Val) (hm : σ.multi a = true)
    (h : l.op = .assert e a v none) : l.writes σ = [⟨e, a, ⟨some l.told, none⟩, .add v⟩] := by
  simp [Lesson.writes, h, Op.writes, hm]

/-- add_comm lifted to stores, at every point and for the whole state: two adjacent "I like X" lessons on a
    multi-valued attribute can be swapped. Each one either applies at the point or not, whatever the order. -/
theorem adds_commute_stateAt (σ : Schema) (S T : List Lesson) (l1 l2 : Lesson) (e : Entity) (a : Attr)
    (v w : Val) (hm : σ.multi a = true) (h1 : l1.op = .assert e a v none) (h2 : l2.op = .assert e a w none)
    (p : TPoint) : stateAt σ (S ++ l1 :: l2 :: T) e a p = stateAt σ (S ++ l2 :: l1 :: T) e a p := by
  simp only [stateAt, relevant_append, relevant, add_writes σ l1 e a v hm h1, add_writes σ l2 e a w hm h2]
  apply fold_swap
  intro s
  rw [hm]
  cases hx : hit e a p ⟨e, a, ⟨some l1.told, none⟩, .add v⟩ <;>
    cases hy : hit e a p ⟨e, a, ⟨some l2.told, none⟩, .add w⟩ <;>
    simp only [List.filter_cons, hx, hy, List.filter_nil, fold, List.foldl_cons, List.foldl_nil,
      Bool.false_eq_true, ↓reduceIte]
  exact (add_comm v w s).symm

/-- The form first stated: CURRENT now, and the summary at a tick before both lessons. (The tick condition
    turned out not to be needed; adds_commute_stateAt holds at every point.) -/
theorem adds_commute_store (σ : Schema) (S T : List Lesson) (l1 l2 : Lesson) (e : Entity) (a : Attr)
    (v w : Val) (hm : σ.multi a = true) (h1 : l1.op = .assert e a v none) (h2 : l2.op = .assert e a w none)
    (t : Int) (_ht : t < l1.told ∧ t < l2.told) :
    sumAt σ (S ++ l1 :: l2 :: T) e a .now = sumAt σ (S ++ l2 :: l1 :: T) e a .now ∧
    sumAt σ (S ++ l1 :: l2 :: T) e a (.pt t) = sumAt σ (S ++ l2 :: l1 :: T) e a (.pt t) := by
  have h := adds_commute_stateAt σ S T l1 l2 e a v w hm h1 h2
  simp only [sumAt, h, and_self]

/-- Controls: the hypotheses of the two theorems above matter. Lean's kernel evaluates both cases.
    Without a DATED lesson last, PREVIOUS does depend on what came before: "0", then "now 1" gives 0, while
    "now 1" alone gives "not 1". And two plain lessons on a single-valued attribute do not commute. -/
def ctlSchema : Schema := {
  multi := fun a => a == 1,
  dom := fun _ => [0, 1, 2] }

example : previous ctlSchema ([⟨1, 100, .assert 0 0 0 none⟩] ++ [⟨2, 200, .change 0 0 1⟩]) 0 0
    ≠ previous ctlSchema ([] ++ [⟨2, 200, .change 0 0 1⟩]) 0 0 := by decide

example : sumAt ctlSchema [⟨1, 100, .assert 0 0 0 none⟩, ⟨2, 200, .assert 0 0 1 none⟩] 0 0 .now
    ≠ sumAt ctlSchema [⟨2, 200, .assert 0 0 1 none⟩, ⟨1, 100, .assert 0 0 0 none⟩] 0 0 .now := by decide

-- and on the multi-valued attribute 1 the same swap changes nothing, as adds_commute_store says
example : sumAt ctlSchema [⟨1, 100, .assert 0 1 0 none⟩, ⟨2, 200, .assert 0 1 1 none⟩] 0 1 .now
    = sumAt ctlSchema [⟨2, 200, .assert 0 1 1 none⟩, ⟨1, 100, .assert 0 1 0 none⟩] 0 1 .now := by decide

-- ------------------------------------------------------------- single-valued attributes and derivability

def Functional (s : State) : Prop := ∀ v w, s v = .yes → w ≠ v → s w = .no

theorem functional_blank : Functional blank := by
  intro v w h _
  simp [blank] at h

theorem functional_step (eff : Effect) (s : State) (hs : Functional s) : Functional (step false eff s) := by
  intro u w hu hw
  cases eff with
  | set v =>
    simp only [step] at hu ⊢
    by_cases huv : u = v
    · subst huv
      simp [hw]
    · simp [huv] at hu
  | add v =>
    simp only [step, Bool.false_eq_true, ↓reduceIte] at hu ⊢
    by_cases huv : u = v
    · subst huv
      simp [hw]
    · simp [huv] at hu
  | retract v =>
    simp only [step, Bool.false_eq_true, ↓reduceIte] at hu ⊢
    by_cases hsv : s v = .yes
    · rw [ite_eq_left hsv] at hu
      by_cases huv : u = v <;> simp [huv] at hu
    · rw [ite_eq_right hsv] at hu ⊢
      by_cases huv : u = v
      · simp [huv] at hu
      · simp only [huv, ↓reduceIte] at hu
        by_cases hwv : w = v
        · simp [hwv]
        · simp only [hwv, ↓reduceIte]
          exact hs u w hu hw
  | clear =>
    simp [step] at hu
  | presume v =>
    simp only [step, Bool.false_eq_true, ↓reduceIte] at hu ⊢
    by_cases hsv : s v = .unk
    · rw [ite_eq_left hsv] at hu ⊢
      by_cases huv : u = v
      · subst huv
        simp [hw]
      · simp [huv] at hu
    · rw [ite_eq_right hsv] at hu ⊢
      exact hs u w hu hw
  | presumeNot v =>
    simp only [step] at hu ⊢
    by_cases hsv : s v = .unk
    · rw [ite_eq_left hsv] at hu ⊢
      by_cases huv : u = v
      · simp [huv] at hu
      · simp only [huv, ↓reduceIte] at hu
        by_cases hwv : w = v
        · simp [hwv]
        · simp only [hwv, ↓reduceIte]
          exact hs u w hu hw
    · rw [ite_eq_right hsv] at hu ⊢
      exact hs u w hu hw

theorem functional_fold : ∀ (ws : List Write) (s : State), Functional s → Functional (fold false s ws)
  | [], _, h => h
  | w :: ws, s, h => functional_fold ws (step false w.eff s) (functional_step w.eff s h)

/-- A single-valued attribute never holds two values at once, at any point, after any lessons. -/
theorem functional_stateAt (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) (p : TPoint)
    (h : σ.multi a = false) : Functional (stateAt σ S e a p) := by
  unfold stateAt
  rw [h]
  exact functional_fold _ _ functional_blank

def deriveEO (sm : Summary) (v1 v2 : Val) : Choice :=
  if deriveYN sm v1 = .yes then .pick v1
  else if deriveYN sm v2 = .yes then .pick v2
  else if deriveYN sm v1 = .no ∧ deriveYN sm v2 = .no then .neither
  else .unknown

theorem firstYes_some (s : State) : ∀ (l : List Val) (w : Val), firstYes s l = some w → s w = .yes
  | [], w, h => by simp [firstYes] at h
  | x :: xs, w, h => by
    by_cases hx : s x = .yes
    · simp only [firstYes, hx, ↓reduceIte] at h
      cases h
      exact hx
    · simp only [firstYes, hx, ↓reduceIte] at h
      exact firstYes_some s xs w h

theorem firstYes_none (s : State) : ∀ (l : List Val), firstYes s l = none → ∀ v, memN v l = true → s v ≠ .yes
  | [], _, v, hv => by simp [memN] at hv
  | x :: xs, h, v, hv => by
    by_cases hx : s x = .yes
    · simp [firstYes, hx] at h
    · simp only [firstYes, hx, ↓reduceIte] at h
      simp only [memN, Bool.or_eq_true, beq_iff_eq] at hv
      rcases hv with rfl | hv
      · exact hx
      · exact firstYes_none s xs h v hv

theorem allNo_true (s : State) : ∀ (l : List Val), allNo s l = true → ∀ v, memN v l = true → s v = .no
  | [], _, v, hv => by simp [memN] at hv
  | x :: xs, h, v, hv => by
    simp only [allNo, Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [memN, Bool.or_eq_true, beq_iff_eq] at hv
    rcases hv with rfl | hv
    · exact h.1
    · exact allNo_true s xs h.2 v hv

theorem memN_withTruth (s : State) (t : Truth) (v : Val) : ∀ (l : List Val),
    memN v (withTruth s t l) = true ↔ (memN v l = true ∧ s v = t)
  | [] => by simp [withTruth, memN]
  | x :: xs => by
    have ih := memN_withTruth s t v xs
    by_cases hx : s x = t
    · simp only [withTruth, hx, ↓reduceIte, memN, Bool.or_eq_true, beq_iff_eq, ih]
      constructor
      · rintro (rfl | ⟨h1, h2⟩)
        · exact ⟨Or.inl rfl, hx⟩
        · exact ⟨Or.inr h1, h2⟩
      · rintro ⟨rfl | h1, h2⟩
        · exact Or.inl rfl
        · exact Or.inr ⟨h1, h2⟩
    · simp only [withTruth, hx, ↓reduceIte, memN, Bool.or_eq_true, beq_iff_eq, ih]
      constructor
      · rintro ⟨h1, h2⟩
        exact ⟨Or.inr h1, h2⟩
      · rintro ⟨rfl | h1, h2⟩
        · exact absurd h2 hx
        · exact ⟨h1, h2⟩

/-- YES_NO is derivable from CURRENT on a single-valued attribute. -/
theorem yesNo_from_summary (dom : List Val) (s : State) (hs : Functional s) (v : Val)
    (hv : memN v dom = true) : deriveYN (summarize false dom s) v = s v := by
  unfold summarize
  cases hf : firstYes s dom with
  | some w =>
    have hw : s w = .yes := firstYes_some s dom w hf
    simp only [Bool.false_eq_true, ↓reduceIte, deriveYN]
    by_cases hvw : v = w
    · subst hvw
      simp [hw]
    · simp only [hvw, ↓reduceIte]
      exact (hs w v hw hvw).symm
  | none =>
    have hny := firstYes_none s dom hf v hv
    cases ha : allNo s dom with
    | true =>
      simp only [↓reduceIte, deriveYN]
      exact (allNo_true s dom ha v hv).symm
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte, deriveYN]
      by_cases hno : s v = .no
      · have hm : memN v (withTruth s .no dom) = true := (memN_withTruth s .no v dom).mpr ⟨hv, hno⟩
        simp only [hm, ↓reduceIte]
        exact hno.symm
      · have hm : memN v (withTruth s .no dom) = false := by
          cases h : memN v (withTruth s .no dom)
          · rfl
          · exact absurd ((memN_withTruth s .no v dom).mp h).2 hno
        simp only [hm, Bool.false_eq_true, ↓reduceIte]
        cases hsv : s v
        · exact absurd hsv hny
        · exact absurd hsv hno
        · rfl

/-- EITHER_OR is derivable from CURRENT on a single-valued attribute. -/
theorem eitherOr_from_summary (dom : List Val) (s : State) (hs : Functional s) (v1 v2 : Val)
    (h1 : memN v1 dom = true) (h2 : memN v2 dom = true) :
    deriveEO (summarize false dom s) v1 v2 = eitherOr s v1 v2 := by
  simp only [deriveEO, eitherOr, yesNo_from_summary dom s hs v1 h1, yesNo_from_summary dom s hs v2 h2]

/-- WHO_HAS is derivable from everyone's CURRENT on a single-valued attribute. -/
theorem whoHas_from_current (σ : Schema) (S : List Lesson) (ents : List Entity) (a : Attr) (v : Val)
    (h : σ.multi a = false) (hv : memN v (σ.dom a) = true) :
    whoHas σ S ents a v = ents.filter (fun e => decide (deriveYN (sumAt σ S e a .now) v = .yes)) := by
  unfold whoHas sumAt
  congr 1
  funext e
  rw [h, yesNo_from_summary _ _ (functional_stateAt σ S e a .now h) v hv]

/-- Nothing ever said is unknown. -/
theorem empty_store_unknown (σ : Schema) (e : Entity) (a : Attr) (p : TPoint) :
    stateAt σ [] e a p = blank := rfl

-- ===================================================================== two-step questions (level 6)

-- Kleene's strong three-valued logic, the logic of compound questions. Unknown is not no (A7).

theorem kAnd_yes_iff (p q : Truth) : kAnd p q = .yes ↔ p = .yes ∧ q = .yes := by
  cases p <;> cases q <;> decide

/-- An unknown part does not hide a no: "and" is no as soon as one part is no. -/
theorem kAnd_no_iff (p q : Truth) : kAnd p q = .no ↔ p = .no ∨ q = .no := by
  cases p <;> cases q <;> decide

/-- An unknown part does not hide a yes: "or" is yes as soon as one part is yes. -/
theorem kOr_yes_iff (p q : Truth) : kOr p q = .yes ↔ p = .yes ∨ q = .yes := by
  cases p <;> cases q <;> decide

theorem kOr_no_iff (p q : Truth) : kOr p q = .no ↔ p = .no ∧ q = .no := by
  cases p <;> cases q <;> decide

theorem kAnd_comm (p q : Truth) : kAnd p q = kAnd q p := by
  cases p <;> cases q <;> rfl

theorem kOr_comm (p q : Truth) : kOr p q = kOr q p := by
  cases p <;> cases q <;> rfl

theorem neg_neg (p : Truth) : p.neg.neg = p := by
  cases p <;> rfl

theorem deMorgan_and (p q : Truth) : (kAnd p q).neg = kOr p.neg q.neg := by
  cases p <;> cases q <;> rfl

theorem deMorgan_or (p q : Truth) : (kOr p q).neg = kAnd p.neg q.neg := by
  cases p <;> cases q <;> rfl

def Truth.ofBool : Bool → Truth
  | true => .yes
  | false => .no

/-- On known parts, Kleene's logic is ordinary logic. -/
theorem kleene_classical (a b : Bool) :
    kAnd (.ofBool a) (.ofBool b) = .ofBool (a && b) ∧ kOr (.ofBool a) (.ofBool b) = .ofBool (a || b)
      ∧ (Truth.ofBool a).neg = .ofBool (!a) := by
  cases a <;> cases b <;> decide

/-- p refines to q: p is unknown, or the same as q. -/
def Truth.refines (p q : Truth) : Bool := p == .unk || p == q

/-- When unknown parts of a compound become known, its yes or no stays (about the connectives, not lessons). -/
theorem kleene_mono (p q p' q' : Truth) (hp : p.refines p' = true) (hq : q.refines q' = true) :
    (kAnd p q).refines (kAnd p' q') = true ∧ (kOr p q).refines (kOr p' q') = true
      ∧ p.neg.refines p'.neg = true := by
  revert hp hq
  cases p <;> cases q <;> cases p' <;> cases q' <;> decide

/-- A compound question with a part that is no is no, whatever the other part. -/
theorem kand_no_part (σ : Schema) (ents : List Entity) (S : List Lesson) (p q : Question)
    (h : (answer σ ents S p).truth = .no) : answer σ ents S (.kand p q) = .yn .no := by
  simp only [answer, h]
  rfl

-- Chains.

theorem hopsEnt_none (σ : Schema) (S : List Lesson) : ∀ (hs : List Attr), hopsEnt σ S none hs = none
  | [] => rfl
  | _ :: _ => rfl

/-- A chain whose first link names no entity of the set (nothing taught, taken back, or "my neighbour", of whom
    the set knows nothing) is "don't know". -/
theorem chain_link_unknown (σ : Schema) (S : List Lesson) (ents : List Entity) (e : Entity) (h : Attr)
    (hs : List Attr) (a : Attr) (nat : Bool) (hn : entOf σ (sumAt σ S e h .now) = none) :
    chainSum σ S ents (.ent e) (h :: hs) a nat = .unknown [] := by
  simp only [chainSum, startEnt, hopsEnt, hn, hopsEnt_none]

/-- A two-step chain has a known answer exactly when both links are known: the first link names an entity, and
    that entity's answer is known. -/
theorem chain_known_iff (σ : Schema) (S : List Lesson) (ents : List Entity) (e : Entity) (h a : Attr) :
    (chainSum σ S ents (.ent e) [h] a false).known = true ↔
      ∃ t, entOf σ (sumAt σ S e h .now) = some t ∧
        (summarize (σ.multi a) (σ.dom a) (linkedState σ S t a)).known = true := by
  cases hx : entOf σ (sumAt σ S e h .now) with
  | none => simp [chainSum, startEnt, hopsEnt, hx, Summary.known]
  | some t => simp [chainSum, startEnt, hopsEnt, hx]

/-- The value of a two-step chain is the value of its second link, read on the entity its first link names. -/
theorem chain_value_iff (σ : Schema) (S : List Lesson) (ents : List Entity) (e : Entity) (h a : Attr) (v : Val) :
    chainSum σ S ents (.ent e) [h] a false = .value v ↔
      ∃ t, entOf σ (sumAt σ S e h .now) = some t ∧ summarize (σ.multi a) (σ.dom a) (linkedState σ S t a) = .value v := by
  cases hx : entOf σ (sumAt σ S e h .now) with
  | none => simp [chainSum, startEnt, hopsEnt, hx]
  | some t => simp [chainSum, startEnt, hopsEnt, hx]

theorem firstYes_blank : ∀ (l : List Val), firstYes blank l = none
  | [] => rfl
  | _ :: l => by simp [firstYes, blank, firstYes_blank l]

theorem entOf_blank (σ : Schema) (m : Bool) (dom : List Val) : entOf σ (summarize m dom blank) = none := by
  simp only [summarize, firstYes_blank]
  split <;> rfl

/-- With no lesson in force about (e, a), e's a is what nothing taught gives. -/
theorem stateAt_untouched (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) (p : TPoint)
    (h : ∀ l ∈ S, touches σ e a l = false) : stateAt σ S e a p = blank := by
  simp [stateAt, relevant_nil_of σ e a p S h, fold]

/-- After DELETE(i), where i was the only lesson in force about (e, a), no lesson in force touches (e, a). -/
theorem untouched_after_delete (σ : Schema) (log : List Entry) (i : Nat) (e : Entity) (a : Attr)
    (honly : ∀ l ∈ effective log, touches σ e a l = true → l.id = i) :
    ∀ l ∈ effective (log ++ [.delete i]), touches σ e a l = false := by
  rw [delete_is_never_taught]
  intro l hl
  rw [List.mem_filter] at hl
  cases ht : touches σ e a l
  · rfl
  · have hid := honly l hl.1 ht
    simp [hid] at hl

/-- Deleting a link: when lesson i was the only lesson in force on the first link of a chain, the chain is
    "don't know" after DELETE(i), however many links follow. -/
theorem delete_link_unknown (σ : Schema) (ents : List Entity) (log : List Entry) (i : Nat) (e : Entity)
    (h : Attr) (hs : List Attr) (a : Attr) (nat : Bool)
    (honly : ∀ l ∈ effective log, touches σ e h l = true → l.id = i) :
    chainSum σ (effective (log ++ [.delete i])) ents (.ent e) (h :: hs) a nat = .unknown [] := by
  apply chain_link_unknown
  simp only [sumAt, stateAt_untouched σ _ e h .now (untouched_after_delete σ log i e h honly), entOf_blank]

/-- A name that no entity holds now points at nobody: the chain is "don't know". -/
theorem named_chain_unknown (σ : Schema) (S : List Lesson) (ents : List Entity) (v : Val) (hs : List Attr)
    (a : Attr) (nat : Bool) (hn : namedEnt σ S ents v = none) :
    chainSum σ S ents (.named v) hs a nat = .unknown [] := by
  simp only [chainSum, startEnt, hn, hopsEnt_none]

/-- With no name lesson in force, no name points at anyone (deleting the alias makes alias chains unknown). -/
theorem namedEnt_untouched (σ : Schema) (S : List Lesson) (ents : List Entity) (v : Val) (na : Attr)
    (hna : σ.nameAttr = some na) (h : ∀ e, ∀ l ∈ S, touches σ e na l = false) : namedEnt σ S ents v = none := by
  have hf : ents.filter (fun e => decide (stateAt σ S e na .now v = .yes)) = [] := by
    rw [List.filter_eq_nil_iff]
    intro e _
    simp [stateAt_untouched σ S e na .now (h e), blank]
  simp only [namedEnt, hna, hf]

theorem fill_blank (src : State) : fill blank src = src := by
  funext x
  simp [fill, blank]

/-- A link fills a gap: when nothing in force is said about e's a, e's linked answer is exactly the answer of
    the entity its link names. -/
theorem linked_fills_gap (σ : Schema) (S : List Lesson) (e : Entity) (a la : Attr) (t : Entity)
    (hl : σ.linkOf a = some la) (ht : entOf σ (sumAt σ S e la .now) = some t)
    (hown : ∀ l ∈ S, touches σ e a l = false) : linkedState σ S e a = stateAt σ S t a .now := by
  simp only [linkedState, hl, ht, stateAt_untouched σ S e a .now hown, fill_blank]

/-- Deep delete: deleting what was said outright about e's a keeps the answer known when the lessons kept still
    imply it through a link. It becomes exactly the linked entity's answer. -/
theorem deep_delete_implied (σ : Schema) (log : List Entry) (i : Nat) (e : Entity) (a la : Attr) (t : Entity)
    (hl : σ.linkOf a = some la) (honly : ∀ l ∈ effective log, touches σ e a l = true → l.id = i)
    (ht : entOf σ (sumAt σ (effective (log ++ [.delete i])) e la .now) = some t) :
    linkedState σ (effective (log ++ [.delete i])) e a = stateAt σ (effective (log ++ [.delete i])) t a .now :=
  linked_fills_gap σ _ e a la t hl ht (untouched_after_delete σ log i e a honly)

/-- What is said outright wins over a link: when every value of e's a is known, the link changes nothing. -/
theorem own_wins (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr)
    (h : ∀ x, stateAt σ S e a .now x ≠ .unk) : linkedState σ S e a = stateAt σ S e a .now := by
  unfold linkedState
  split
  · rfl
  · split
    · rfl
    · funext x
      simp [fill, h x]

theorem functional_fill (own src : State) (ho : Functional own) (hs : Functional src) :
    Functional (fill own src) := by
  intro v w hv hw
  simp only [fill] at hv ⊢
  by_cases hov : own v = .unk
  · rw [ite_eq_left hov] at hv
    by_cases how : own w = .unk
    · rw [ite_eq_left how]
      exact hs v w hv hw
    · rw [ite_eq_right how]
      cases hw' : own w with
      | yes =>
        have := ho w v hw' (fun h => hw h.symm)
        rw [hov] at this
        cases this
      | no => rfl
      | unk => exact absurd hw' how
  · rw [ite_eq_right hov] at hv
    have hno : own w = .no := ho v w hv hw
    simp [hno]

/-- A linked answer on a single-valued attribute never holds two values, links and contradictions included. -/
theorem functional_linked (σ : Schema) (S : List Lesson) (e : Entity) (a : Attr) (hm : σ.multi a = false) :
    Functional (linkedState σ S e a) := by
  unfold linkedState
  split
  · exact functional_stateAt σ S e a .now hm
  · split
    · exact functional_stateAt σ S e a .now hm
    · exact functional_fill _ _ (functional_stateAt σ S e a .now hm) (functional_stateAt σ S _ a .now hm)

/-- The natural step gives a country only from a known city. -/
theorem natSum_value_iff (σ : Schema) (sm : Summary) (c : Val) :
    natSum σ sm = .value c ↔ ∃ w, sm = .value w ∧ σ.natMap w = some c := by
  cases sm with
  | value w => cases h : σ.natMap w <;> simp [natSum, h]
  | none => simp [natSum]
  | unknown ex => simp [natSum]
  | set ys ns => simp [natSum]

/-- Controls, evaluated by Lean's kernel. Entities 0 = me, 1 = my sister, 2 = the boat; attribute 0 = eye colour
    (values 0, 1, 2), 1 = owner and 2 = "eyes like" (values 10 = me, 11 = my sister). The boat belongs to me, my
    eyes are 1, my sister's are 2: the owner's eyes are 1. After "the boat belongs to my sister now" they are 2;
    without the owner lesson, "don't know". The cat-like link: entity 1's eyes are like mine, and nothing else
    is said about them: 1. -/
def ctl6 : Schema := {
  multi := fun _ => false,
  dom := fun a => match a with
    | 0 => [0, 1, 2]
    | _ => [10, 11],
  valEnt := fun v => match v with
    | 10 => some 0
    | 11 => some 1
    | _ => none,
  linkOf := fun a => match a with
    | 0 => some 2
    | _ => none }

example : chainSum ctl6 [⟨1, 100, .assert 2 1 10 none⟩, ⟨2, 200, .assert 0 0 1 none⟩,
    ⟨3, 300, .assert 1 0 2 none⟩] [0, 1, 2] (.ent 2) [1] 0 false = .value 1 := by decide

example : chainSum ctl6 [⟨1, 100, .assert 2 1 10 none⟩, ⟨2, 200, .assert 0 0 1 none⟩,
    ⟨3, 300, .assert 1 0 2 none⟩, ⟨4, 400, .change 2 1 11⟩] [0, 1, 2] (.ent 2) [1] 0 false = .value 2 := by decide

example : chainSum ctl6 [⟨2, 200, .assert 0 0 1 none⟩, ⟨3, 300, .assert 1 0 2 none⟩] [0, 1, 2] (.ent 2) [1] 0 false
    = .unknown [] := by decide

example : chainSum ctl6 [⟨1, 100, .assert 1 2 10 none⟩, ⟨2, 200, .assert 0 0 1 none⟩] [0, 1, 2] (.ent 1) [] 0 false
    = .value 1 := by decide

end LwLogic
