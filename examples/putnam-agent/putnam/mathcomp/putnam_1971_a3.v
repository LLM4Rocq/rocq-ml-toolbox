From mathcomp Require Import all_ssreflect all_algebra.
From mathcomp Require Import reals normedtype.

Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

Local Open Scope ring_scope.

Parameter R : realType.
Theorem putnam_1971_a3
    (a b c : R * R)
    (r : R)
    (habclattice : fst a = (Num.floor (fst a))%:~R /\ snd a = (Num.floor (snd a))%:~R /\ fst b = (Num.floor (fst b))%:~R /\ snd b = (Num.floor (snd b))%:~R /\ fst c = (Num.floor (fst c))%:~R /\ snd c = (Num.floor (snd c))%:~R)
    (habcneq : a <> b /\ b <> c /\ c <> a)
    (hr : r > 0)
    (oncircle : (R * R) -> R -> (R * R) -> Prop := fun C rad p => (fst C - fst p)^+2 + (snd C - snd p)^+2 = rad^+2)
    (hcircle : exists C : R * R, oncircle C r a /\ oncircle C r b /\ oncircle C r c)
    : ((fst a - fst b)^+2 + (snd a - snd b)^+2) * ((fst b - fst c)^+2 + (snd b - snd c)^+2) * ((fst c - fst a)^+2 + (snd c - snd a)^+2) >= 4 * r ^+ 2.
Proof. Admitted.
