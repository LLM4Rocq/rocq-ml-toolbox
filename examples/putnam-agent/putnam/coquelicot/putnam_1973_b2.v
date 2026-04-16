Require Import Reals QArith Coquelicot.Complex.
Theorem putnam_1973_b2
    (z : C)
    (hzrat : exists q1 q2 : Q, Re z = Q2R q1 /\ Im z = Q2R q2)
    (hznorm : Cmod z = 1%R)
    : forall n : nat, exists q1 q2 : Q, Cmod (z ^ (2 * n) - RtoC 1) = Q2R q1 /\ Cmod (RtoC 1 / (z ^ (2 * n) - RtoC 1)) = Q2R q2.
Proof. Admitted.
