From mathcomp Require Import all_ssreflect all_algebra.
From mathcomp Require Import reals.
From mathcomp Require Import classical_sets.

Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

Open Scope ring_scope.
Open Scope classical_set_scope.

Local Parameter R : realType.
Theorem putnam_1962_b2
    : exists f : R -> set nat, forall a b : R, a < b -> f a `<` f b.
Proof. Admitted.