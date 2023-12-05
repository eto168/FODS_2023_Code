""" 
Script for hierarchical bayesian modelling

@author: Ethan Tse 
"""

import pyro
import pyro.distributions as dist
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyro import poutine
import tqdm
from pyro.distributions.constraints import positive
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.optim import Adam
from pyro.infer.autoguide import AutoNormal

# LKJcholesky prior for covariance matrix
# y ~ MVN
# mu ~ N(0, 1)


def probability_function(tumour_content, variant_allele_count, copy_number_state, cancer_cell_fraction):
    """ A function to unify parameters into a probability for observing X number of reads. 

    NOTE:
        - copy_number_state is a vector

    Must return a probability (i.e. value between {0, 1})
    """

    # cancer_cell_fraction = cancer_cell_fraction / \
    #     torch.max(cancer_cell_fraction)

    # normal, reference, variant all CN = 2. Our default assumption
    # copy_number_vector = [2, 2, copy_number_state]
    copy_number_vector = copy_number_state

    # this is equivalent to saying, only variant alleles are observed in the variant population
    # variant_allele_vector = [0, 0, variant_allele_count]
    variant_allele_vector = variant_allele_count

    # # Pyclone's function
    # c1 = (1-tumour_content)*copy_number_vector[0]
    # c2 = tumour_content*(1-cancer_cell_fraction)*copy_number_vector[1]
    # c3 = tumour_content*cancer_cell_fraction*copy_number_vector[2]

    # z_constant = (1-tumour_content)*copy_number_vector[0] + \
    #     tumour_content*(1-cancer_cell_fraction)*copy_number_vector[1] + \
    #     tumour_content*cancer_cell_fraction*copy_number_vector[2]

    # normal_pop = (((1-tumour_content)*copy_number_vector[0])/z_constant) * \
    #     (variant_allele_vector[0]/copy_number_vector[0])
    # reference_pop = ((tumour_content*(1-cancer_cell_fraction)*copy_number_vector[1])/z_constant) * \
    #     (variant_allele_vector[1]/copy_number_vector[1])
    # variant_pop = ((tumour_content*cancer_cell_fraction*copy_number_vector[2])/z_constant) * \
    #     (variant_allele_vector[2]/copy_number_vector[2])

    # Pyclone's function
    c1 = (1-tumour_content)*copy_number_vector
    c2 = tumour_content*(1-cancer_cell_fraction)*copy_number_vector
    c3 = tumour_content*cancer_cell_fraction*copy_number_vector

    z_constant = (1-tumour_content)*copy_number_vector + \
        tumour_content*(1-cancer_cell_fraction)*copy_number_vector + \
        tumour_content*cancer_cell_fraction*copy_number_vector

    normal_pop = (((1-tumour_content)*copy_number_vector)/z_constant) * \
        (variant_allele_vector/copy_number_vector)
    reference_pop = ((tumour_content*(1-cancer_cell_fraction)*copy_number_vector)/z_constant) * \
        (variant_allele_vector/copy_number_vector)
    variant_pop = ((tumour_content*cancer_cell_fraction*copy_number_vector)/z_constant) * \
        (variant_allele_vector/copy_number_vector)

    probability = normal_pop + reference_pop + variant_pop

    return probability


@config_enumerate
def model(data):
    """ 
    Our Bayesian Hierarchical Gaussian Mixture Model

    We infer one-dimensional gaussian distributions. 
    """
    # interpret dictionary
    K = data["num_clust"]
    num_svs = data["num_svs"]
    num_supporting_reads = data["num_supporting_reads"]
    num_reference_reads = data["num_reference_reads"]
    num_total_reads = data["num_total_reads"]
    vaf = data["vaf"]
    variant_allele_count = data["variant_allele_count"]
    copy_number_state = data["copy_number_state"]
    tumour_content = data["tumour_content"]

    # Global variables

    # # we assume every mutation in one sample has the same tumour content
    # # tumour content is between 0 and 1, but it has no prior. it is fully observed.
    # tumour_content = pyro.sample(
    #     "t_n", fn=dist.Normal(loc=0, scale=1), obs=tumour_content)

    # -------------------------------------------------------------------------------------------

    ##### Hyperparameters #####
    dirichlet_alpha = torch.ones(K)

    # torch.ones(K) is an uniformative dirichlet prior (we put no emphasis on any possibility.)
    mixture_coefficients = pyro.sample(
        "pi", dist.Dirichlet(dirichlet_alpha))

    # we do not assume clusters have different SDs
    # LKJ params: d, concentration, where d is the number of clusters
    cluster_cov = pyro.sample(
        "cov", dist.InverseGamma(1, 1))

    # -------------------------------------------------------------------------------------------

    # Cluster-specific means
    with pyro.plate("k", K):
        # prior on cluster means
        cluster_means = pyro.sample(
            "mu", dist.Normal(torch.tensor([0.0]), torch.tensor([1.0])))

    # -------------------------------------------------------------------------------------------

    """ 
    Information per SV: 
    
    Latent Parameters (These are inferred):
    - Each SV will have its own cluster assignment. (i.e. SV is assigned to cluster k)
    - Each SV will have its own CCF. (i.e. 60% of cancer cells have this SV)
    
    Observed Values: 
    - Each SV will have its own copy number state
    - Each SV will have its own number of variant alleles
    - Each SV will have its own number of total reads
    - Each SV will have its own number of supporting reads
    """
    with pyro.plate("SV", num_svs):
        ##### latent params #####

        # these are what we infer: what cluster does a mutation belong to? what is its CCF?
        cluster_assignment = pyro.sample(
            "h", dist.Categorical(mixture_coefficients))

        # our CCF is modelled with a gaussian mixture model:
        cancer_cell_fraction = pyro.sample(
            "CCF", dist.Normal(loc=cluster_means[cluster_assignment], scale=cluster_cov))

        # using expit, convert normal output to value between [0, 1] since CCF is necessarily a proportion
        cancer_cell_fraction = torch.special.expit(cancer_cell_fraction)

        ############# Observed values with NO PRIORS ##############
        # Genotype Values ---------------------
        # copy_number_state = pyro.sample(
        #     "c_n", fn=dist.Normal(loc=cancer_cell_fraction, scale=1), obs=copy_number_state)

        # variant_allele_count = pyro.sample(
        #     "vac_n", fn=dist.Normal(loc=cancer_cell_fraction, scale=1), obs=variant_allele_count)

        # num_total_reads = pyro.sample(
        #     "d_n", fn=dist.Normal(loc=cancer_cell_fraction, scale=1), obs=num_total_reads)

        # # we assume every mutation in one sample has the same tumour content
        # # tumour content is between 0 and 1, but it has no prior. it is fully observed.
        # tumour_content = pyro.sample(
        #     "t_n", fn=dist.Normal(loc=cancer_cell_fraction, scale=1), obs=tumour_content)

        ############## Observed values WITH PRIORS ##############
        # Read information ----------------------
        # num supporting reads follows a binomial distribution, with number of tries = number of total reads covered
        num_supporting_reads = pyro.sample(
            "r_n", fn=dist.Binomial(num_total_reads, probability_function(tumour_content, variant_allele_count, copy_number_state, cancer_cell_fraction)), obs=num_supporting_reads)

    return cluster_assignment, cancer_cell_fraction


def train_gmm(data, num_steps=10000):
    optimizer = Adam(optim_args={"lr": 0.001})  # decrease learning rate
    guide = AutoNormal(poutine.block(model, hide=["h"]))

    trace = poutine.trace(model).get_trace(data)
    trace.compute_log_prob()
    print("---------- Tensor Shapes ------------")
    print(trace.format_shapes())

    print(trace.log_prob_sum())

    svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO())

    # save losses
    losses = []
    for step in tqdm.trange(num_steps):  # Consider running for more steps.
        loss = svi.step(data)
        losses.append(loss)
        if step % 1000 == 0 or step == num_steps:
            print(f"Step {step}/{num_steps}, Loss: {loss}")
    return losses


def classifier(data, trained_model, temperature=0):
    inferred_model = infer_discrete(
        trained_model, temperature=temperature, first_available_dim=-2
    )  # avoid conflict with data plate

    trace = pyro.poutine.trace(inferred_model).get_trace(data)

    ccf = trace.nodes["CCF"]["value"]
    cluster = trace.nodes["h"]["value"]

    result = {"ccf": ccf,
              "cluster": cluster
              }
    return result


def plot_training_error(loss, num_steps):
    # Plot the training loss
    plt.plot(loss)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('SVI Training Loss')
    plt.show()

    plt.savefig(f"./code_outputs/model/loss_{num_steps}.png")


def process_outputs(ccf, clusters):
    np.savetxt("./code_outputs/model/tsv/ccf.tsv", ccf, delimiter='\t')
    np.savetxt("./code_outputs/model/tsv/clusters.tsv",
               clusters, delimiter='\t')


def main():
    K = 10  # K clusters

    # -------------------------
    df = pd.read_csv("./somatic_svs.tsv",
                     delimiter="\t", index_col=None)

    # Get params:
    num_svs = df.shape[0]  # get number of rows == number of svs

    # Counts information
    num_supporting_reads = df["variant_counts"].to_numpy()
    num_reference_reads = df["ref_counts"].to_numpy()
    num_total_reads = df["total_counts"].to_numpy()
    vaf = df["VAF"].to_numpy()

    # Genotype Information
    # genotype = df["genotype"].to_numpy()
    variant_allele_count = df["variant_allele_count"].to_numpy()
    # we assume all are diploid to start
    copy_number_state = df["copy_number"].to_numpy()

    # Tumour content
    # assume tumour content is 1
    tumour_content = df["tumour_content"].to_numpy()

    data = {
        "num_clust": K,
        "num_svs": num_svs,
        "num_supporting_reads": torch.tensor(num_supporting_reads),
        "num_reference_reads": torch.tensor(num_reference_reads),
        "num_total_reads": torch.tensor(num_total_reads),
        "vaf": torch.tensor(vaf),
        "variant_allele_count": torch.tensor(variant_allele_count),
        "copy_number_state": torch.tensor(copy_number_state),
        "tumour_content": torch.tensor(tumour_content)
    }

    # ------------------------- Render model
    # pyro.render_model(model,
    #                   model_args=data,
    #                   filename="./code_outputs/model/model.png",
    #                   render_distributions=True)

    # ------------------------- Run Model
    num_steps = 100

    losses = train_gmm(data, num_steps=num_steps)

    plot_training_error(loss=losses, num_steps=num_steps)

    pyro.render_model(model, model_args=(data,),
                      filename="./code_outputs/model/model.png")

    print("Finish")

    # Get information:
    autoguide = AutoNormal(pyro.poutine.block(
        model, hide=['h']))
    guide_trace = pyro.poutine.trace(autoguide).get_trace(
        data)  # record the globals
    trained_model = pyro.poutine.replay(
        model, trace=guide_trace)  # replay the globals

    test_values = classifier(data, trained_model=trained_model)

    cluster_assignments = test_values["cluster"].to_numpy()
    ccfs = test_values["ccf"].to_numpy()

    process_outputs(ccf=ccfs, clusters=cluster_assignments)


if __name__ == "__main__":
    main()
