From mathcomp Require Import all_ssreflect all_algebra.
From mathcomp Require Import reals.

Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

Open Scope ring_scope.

Parameter R : realType.
Definition putnam_1963_b1_solution : int := 2.
Theorem putnam_1963_b1
    (a : int)
    (Ply : int -> {poly R} := fun n => ('X^2 - 'X + (n%:~R)%:P))
    (D : int -> bool := fun n => (Ply n %| ('X^13 + 'X + (90%:R)%:P)))
    : D a = true <-> a = putnam_1963_b1_solution.
Proof. Admitted.