import andrey
from andrey.data import load_dataset

if __name__ == "__main__":  # worker processes re-import this script
    # 7,466 cells in nine conditions; env gives each cell's condition, 0 for the observational one.
    sachs = load_dataset("sachs", conditions=True)
    data, env = sachs.data, sachs.env
    consensus = {frozenset(edge[:2]) for edge in sachs.graph.oriented_edges()}

    def summary(out):
        edges = out.structure.oriented_edges()
        known = sum(frozenset(edge[:2]) in consensus for edge in edges)
        directed = sum(kind == "directed" for *_, kind in edges)
        return f"{len(edges)} edges, {known} in the consensus network, {directed} directed"

    # PC on the observational cells alone, and on all nine conditions pooled without env.
    print(f"pc, observational:  {summary(andrey.pc(data[env == 0]))}")
    print(f"pc, pooled:         {summary(andrey.pc(data))}")

    # The conditions are categories, not numbers: these tests and scores read env as one.
    cdnod = andrey.cdnod(data, env, indep_test="cond_gauss")
    print(f"cdnod:              {summary(cdnod)}")
    scores = {"score_func": "cond_gauss", "categorical": []}
    ges = andrey.ges(data, env, **scores)
    print(f"ges:                {summary(ges)}")
    boss = andrey.boss(data, env, **scores)
    print(f"boss:               {summary(boss)}")

    # The experimenters set each cell's condition, so no hidden variable causes it.
    fci_jci = andrey.fci_jci(data, env, indep_test="cond_gauss", confounded_index=False)
    print(f"fci_jci:            {summary(fci_jci)}")
    boss_fci = andrey.boss_fci(data, env, indep_test="cond_gauss", confounded_index=False, **scores)
    print(f"boss_fci:           {summary(boss_fci)}")

    # What the conditions change, and the index's edges into those variables.
    print(f"changing variables: {cdnod.metadata['changing_variables']}")
    index = fci_jci.metadata["index_edges"]
    print(f"fci_jci's index:    {len(index)} edges, {', '.join({kind for _, kind in index})}")
