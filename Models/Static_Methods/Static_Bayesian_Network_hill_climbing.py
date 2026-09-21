from pgmpy.models import BayesianNetwork

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
import networkx as nx
import matplotlib.pyplot as plt
    

def get_best_PGM_hill_climb(data):
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data),)

    return best_model


def synthetic_population_from_static_BN_hill(training_data, 
                      folder_sampling, n_generation):
    
    print("Structure research...")
    # Computing the best sructure for Bayesian Network
    PGM = get_best_PGM_hill_climb(training_data.astype(str))
    
    nx.draw_circular(
        PGM, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight="bold"
    )
    plt.savefig(f"{folder_sampling}/best_graph.png")
    plt.close()
    print("Structure Found")
    
    
    
    print("Fitting the CPTs")    
    model = BayesianNetwork(PGM)
    model.fit(training_data.astype(str))
    print("Model fitted")
    
    print("Population generation...")
    synthetic_population = model.simulate(n_generation)    
    
    return synthetic_population
        

