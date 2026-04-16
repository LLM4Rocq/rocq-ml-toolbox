From mathcomp Require Import all_ssreflect all_algebra.
From mathcomp Require Import reals sequences.

Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

Local Open Scope ring_scope.

Parameter R : realType.
Theorem putnam_1966_a4
    (a : nat -> int)
    (ha1 : a 1%nat = 2)
    (hai : forall n : nat, ge n 1 -> ((exists m : int, a n + 1 = m ^+ 2)  /\ a (n.+1) = a n + 2) \/ ((~ exists m : int, a n + 1 = m^+ 2) /\ a (n.+1) = a n + 1))
    (f : nat -> R := fun n => n%:R + (Num.floor (@Num.sqrt R n%:R + 1/2))%:~R)
    : forall n : nat, ge n 1 -> (a n)%:~R = f n.
Proof. Admitted.