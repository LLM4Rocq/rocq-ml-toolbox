From Coq Require Import Arith.

Axiom fake_oracle : 0 = 0.

Theorem oracle_demo : 0 = 0.
Proof.
  exact fake_oracle.
Qed.
