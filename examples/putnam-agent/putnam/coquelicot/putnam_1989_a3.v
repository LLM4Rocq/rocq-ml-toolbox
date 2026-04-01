Require Import Reals Coquelicot.Coquelicot.
Open Scope C.
Theorem putnam_1989_a3
    (f : C -> C := fun z => 11 * z ^ 10 + 10 * Ci * z ^ 9 + 10 * Ci * z - 11)
    : forall (x: C), f x = 0 -> Cmod x = R1.
Proof. Admitted.
