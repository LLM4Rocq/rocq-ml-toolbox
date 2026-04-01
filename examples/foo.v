Fixpoint burn (n : nat) : bool :=
  match n with
  | O => true
  | S n' => burn n'
  end.

Goal burn 5000000000 = true.

