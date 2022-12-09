from supports import SupportTester

class SupportTester_PartyCausalSymmetry(SupportTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = lambda idx, val, par: -self.vpool.id(
            'v_{2}==0'.format(idx, val, par)) if idx in self.binary_variables and val == 1 else self.vpool.id(
            'v_{2}=={1}'.format(idx, val, par))

class SupportTester_FullCausalSymmetry(SupportTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = lambda idx, val, par: -self.vpool.id(
            'v_{2}==0'.format(idx, val, sorted(par))) if idx in self.binary_variables and val == 1 else self.vpool.id(
            'v_{2}=={1}'.format(idx, val, sorted(par)))

if __name__ == '__main__':
    import numpy as np
    import itertools

# Special problem for Victor Gitton
    parents_of = np.array(([3, 4], [4, 5], [5, 3]))
    observed_cardinalities = (3, 3, 3)
    occurring_events_card3 = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)]
                                     + list(itertools.permutations(range(3))))


    st = SupportTester(parents_of=parents_of,
                       observed_cardinalities=observed_cardinalities,
                       nof_events=(3 + 6))
    print("3 outcome with no symmetry: ",
          st.feasibleQ_from_matrix(occurring_events_card3, name='mgh'))

    st = SupportTester_PartyCausalSymmetry(parents_of=parents_of,
                                           observed_cardinalities=observed_cardinalities,
                                           nof_events=(3 + 6))
    print("3 outcome with only party symmetry: ",
          st.feasibleQ_from_matrix(occurring_events_card3, name='mgh'))

    st = SupportTester_FullCausalSymmetry(parents_of=parents_of,
                                          observed_cardinalities=observed_cardinalities,
                                          nof_events=(3 + 6))
    print("3 outcome w/ transposition symmetry: ",
          st.feasibleQ_from_matrix_CONSERVATIVE(occurring_events_card3,
                                                name='mgh'))
