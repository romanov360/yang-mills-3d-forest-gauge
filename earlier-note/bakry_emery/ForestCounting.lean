/-
Kernel-checked combinatorial core of the gauge-fixing-forest bound (Mathlib-free).

A periodic choice function mu : (Z/P)^d -> {0..d-1} fixes the edge (x - e_mu(x), mu(x)) at every site x.
The edge (y, a) is fixed iff mu(y + e_a) = a. For the plaquette (y; a < b) with edges
(y,a), (y+e_a, b), (y+e_b, a), (y, b), the number of fixed edges is
  f = [mu(y+e_a) = a] + [mu(y+e_a+e_b) = b] + [mu(y+e_a+e_b) = a] + [mu(y+e_b) = b].
A free edge (y, a) lies in the plaquettes (y; a, b) and (y - e_b; a, b) for every b /= a; its coefficient is
  c(y, a) = sum over those 2(d-1) plaquettes of (4 - f).
The fixed set is a forest for every mu (the coordinate sum decreases strictly along parent pointers); that part is a
one-line argument in the paper and is not formalised here. The theorems below check, over one period cell (which
suffices because everything is P-periodic):
  d = 4, P = 2: every plaquette has exactly one fixed edge, and every free edge has c = 18;
  d = 3, P = 3: every free edge has c <= 11.
-/

/-- site of (Z/P)^d encoded as a list of d coordinates; shift coordinate `a` by `s` (mod P). -/
def shiftC (P : Nat) (x : List Nat) (a : Nat) (s : Int) : List Nat :=
  x.zipIdx.map fun (xi, i) => if i == a then ((((xi : Int) + s) % P + P) % P).toNat else xi

/-- index of a site in lexicographic order (first coordinate most significant). -/
def idx (P : Nat) (x : List Nat) : Nat := x.foldl (fun acc xi => acc * P + xi) 0

def muAt (P : Nat) (pat : List Nat) (x : List Nat) : Nat := pat.getD (idx P x) 0

def fixedE (P : Nat) (pat : List Nat) (y : List Nat) (a : Nat) : Bool :=
  muAt P pat (shiftC P y a 1) == a

/-- number of fixed edges in the plaquette (y; a, b), a /= b (order-independent formula). -/
def fixedCount (P : Nat) (pat : List Nat) (y : List Nat) (a b : Nat) : Nat :=
  let ya := shiftC P y a 1
  let yb := shiftC P y b 1
  let yab := shiftC P ya b 1
  (if muAt P pat ya == a then 1 else 0) + (if muAt P pat yb == b then 1 else 0) +
  (if muAt P pat yab == a then 1 else 0) + (if muAt P pat yab == b then 1 else 0)

def allSites (P d : Nat) : List (List Nat) :=
  (List.range d).foldl (fun acc _ => acc.flatMap fun x => (List.range P).map fun c => x ++ [c]) [[]]

/-- coefficient of the free edge (y, a): plaquettes (y; a,b) and (y - e_b; a,b) for b /= a. -/
def coeff (P d : Nat) (pat : List Nat) (y : List Nat) (a : Nat) : Nat :=
  ((List.range d).filter (· != a)).foldl (fun acc b =>
    acc + (4 - fixedCount P pat y a b) + (4 - fixedCount P pat (shiftC P y b (-1)) a b)) 0

def pat4 : List Nat := [0,2,1,2,3,3,1,0,0,1,3,3,2,1,2,0]
def pat3 : List Nat := [2,1,0,1,0,2,1,2,0,1,0,2,1,2,0,2,1,0,1,2,0,2,1,0,1,0,2]

/-- d = 4: every plaquette of the period cell contains exactly one fixed edge. -/
def perfect4 : Bool :=
  (allSites 2 4).all fun y => (List.range 4).all fun a => (List.range 4).all fun b =>
    a == b || fixedCount 2 pat4 y a b == 1

/-- d = 4: every free edge has coefficient 18. -/
def coeff4 : Bool :=
  (allSites 2 4).all fun y => (List.range 4).all fun a => fixedE 2 pat4 y a || coeff 2 4 pat4 y a == 18

/-- d = 3: every free edge has coefficient at most 11. -/
def coeff3 : Bool :=
  (allSites 3 3).all fun y => (List.range 3).all fun a => fixedE 3 pat3 y a || coeff 3 3 pat3 y a ≤ 11

/-- sanity: one fixed edge per site, i.e. the cell has P^d fixed edges, and the cells have the expected size. -/
def sizes : Bool := (allSites 2 4).length == 16 && (allSites 3 3).length == 27 && pat4.length == 16 && pat3.length == 27

/-- control: the temporal gauge (mu = 0) in d = 4 has free-edge coefficient 20, not 18. -/
def temporal4 : Bool :=
  (allSites 2 4).all fun y => (List.range 4).all fun a =>
    fixedE 2 (List.replicate 16 0) y a || coeff 2 4 (List.replicate 16 0) y a == 20

theorem sizes_ok : sizes = true := by decide +kernel
theorem perfect4_ok : perfect4 = true := by decide +kernel
theorem coeff4_ok : coeff4 = true := by decide +kernel
theorem coeff3_ok : coeff3 = true := by decide +kernel
theorem temporal4_ok : temporal4 = true := by decide +kernel

#print axioms perfect4_ok
#print axioms coeff4_ok
#print axioms coeff3_ok
#print axioms temporal4_ok
