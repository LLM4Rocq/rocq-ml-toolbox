From mathcomp Require Import all_ssreflect all_algebra.
From mathcomp Require Import reals.
From mathcomp Require Import complex.

Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

Local Open Scope ring_scope.

Parameter R : realType.
Theorem putnam_2014_a5
    (P : nat -> {poly R[i]} := fun n => \sum_(1 <= i < n.+1) i%:R *: 'X^(i.-1))
    : forall j k : nat, (gt j 0 /\ gt k 0) -> j <> k -> gcdp (P j) (P k) = 1.
Proof. Admitted.
