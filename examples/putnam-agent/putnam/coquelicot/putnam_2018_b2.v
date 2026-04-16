Require Import Reals Coquelicot.Coquelicot.
Open Scope C_scope.
Theorem putnam_2018_b2
    (n : nat)
    (hn : gt n 0)
    (f : nat -> C -> C)
    (hf : forall z : C, f n z = sum_n_m (fun i => (((RtoC (INR n)) - (RtoC (INR i))) * z ^ i)) 0 (n-1))
    : forall (z : C), Cmod z <= 1%R -> f n z <> 0.
Proof. Admitted.
