from pytanque import Pytanque

with Pytanque('127.0.0.1', 8765) as client:
    filepath = 'stress_test_light/source/algebra/fraction.v'

    state = client.start(filepath, "addN_l")

    for step in ['elim/quotW=> x; apply/eqP; rewrite piE /equivf.', 'rewrite /addf /oppf !numden_Ratio ?(oner_eq0, mulf_neq0, domP) //.', 'by rewrite mulr1 mulr0 mulNr addNr.', 'Qed.']:
        state = client.run(state, step)
    
    print(client.goals(state))
    notations = client.list_notations_in_statement(state, "Lemma addN_l : left_inverse 0%:F opp add.")
    print(notations)