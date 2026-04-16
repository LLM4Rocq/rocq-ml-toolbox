From mathcomp Require Import all_ssreflect all_algebra.
From mathcomp Require Import reals.

Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

Local Open Scope ring_scope.

Parameter R : realType.
Theorem putnam_1973_a3
    (b : int -> R)
    (hbminle : forall n : int, (forall k : int, k > 0 -> b n <= k%:~R + (n%:~R/k%:~R)) /\ (exists k : int, k > 0 /\ b n =  k%:~R + (n%:~R/k%:~R)))
    : forall n : int, n > 0 -> Num.floor (b n) = Num.floor (@Num.sqrt R (4 * n%:~R + 1)).
Proof. Admitted.