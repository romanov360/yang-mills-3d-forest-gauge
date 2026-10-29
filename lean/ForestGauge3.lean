/-
Kernel-checked combinatorics of the three-dimensional forest gauge (Mathlib-free).

The pattern mu : (Z/3)^3 -> {0,1,2} fixes the edge (x - e_mu(x), mu(x)) at every site x, so the edge (y, a) is fixed
iff mu(y + e_a) = a.  The fixed set is a forest on Z^3: along a cycle, the site of largest coordinate sum would carry
two fixed edges coming in from below, but each site has exactly one.  Everything below is 3-periodic, so checking
one period cell suffices.  For a free edge e, the coefficient c(e) counts the free edges of its four plaquettes with
multiplicity, e itself once per plaquette; two edges share at most one plaquette, so e has exactly c(e) - 4 distinct
free partners.  Checked:
  * every plaquette has one or two fixed edges (54 with one, 27 with two, per cell of 81 plaquettes);
  * every free edge has c = 11, i.e. exactly 7 free partners (54 free edges per cell);
  * control: in the temporal gauge (mu = 0) every free edge has c = 12, i.e. 8 partners.
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

def pat3 : List Nat := [2,1,0,1,0,2,1,2,0,1,0,2,1,2,0,2,1,0,1,2,0,2,1,0,1,0,2]

def plaqs3 : List (List Nat × Nat × Nat) :=
  (allSites 3 3).flatMap fun y => [(y, 0, 1), (y, 0, 2), (y, 1, 2)]

/-- every plaquette of the cell has one or two fixed edges. -/
def plaqOK : Bool := plaqs3.all fun (y, a, b) => fixedCount 3 pat3 y a b == 1 || fixedCount 3 pat3 y a b == 2

def nOne : Nat := (plaqs3.filter fun (y, a, b) => fixedCount 3 pat3 y a b == 1).length
def nTwo : Nat := (plaqs3.filter fun (y, a, b) => fixedCount 3 pat3 y a b == 2).length

def freeEdges : List (List Nat × Nat) :=
  (allSites 3 3).flatMap fun y => ((List.range 3).filter fun a => !fixedE 3 pat3 y a).map fun a => (y, a)

/-- every free edge has coefficient exactly 11, i.e. seven free partners. -/
def coeffEq : Bool := freeEdges.all fun (y, a) => coeff 3 3 pat3 y a == 11

/-- control: in the temporal gauge every free edge has 8 partners (coefficient 12). -/
def temporal3 : Bool :=
  (allSites 3 3).all fun y => (List.range 3).all fun a =>
    fixedE 3 (List.replicate 27 0) y a || coeff 3 3 (List.replicate 27 0) y a == 12

theorem plaq_ok : plaqOK = true := by decide +kernel
theorem one_count : nOne = 54 := by decide +kernel
theorem two_count : nTwo = 27 := by decide +kernel
theorem free_count : freeEdges.length = 54 := by decide +kernel
theorem coeff_eq : coeffEq = true := by decide +kernel
theorem temporal3_ok : temporal3 = true := by decide +kernel

#print axioms plaq_ok
#print axioms one_count
#print axioms two_count
#print axioms free_count
#print axioms coeff_eq
#print axioms temporal3_ok
