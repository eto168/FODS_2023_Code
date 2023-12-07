""" 
Script for hierarchical bayesian modelling

@author: Ethan Tse 
"""

from statistics import mode
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
from pyro.infer.autoguide.initialization import init_to_mean, init_to_sample


def probability_function(tumour_content, variant_allele_count, copy_number_state, cancer_cell_fraction):
    """ A function to unify parameters into a probability for observing X number of reads. 

    NOTE:
        - copy_number_state is a vector

    Must return a probability (i.e. value between {0, 1})
    """

    # normal, reference, variant all CN = 2. Our default assumption
    copy_number_vector = copy_number_state

    # this is equivalent to saying, only variant alleles are observed in the variant population
    variant_allele_vector = variant_allele_count

    # Pyclone's function

    z_constant = (1-tumour_content)*copy_number_vector + \
        tumour_content*(1-cancer_cell_fraction)*copy_number_vector + \
        tumour_content*cancer_cell_fraction*copy_number_vector

    normal_pop = (((1-tumour_content)*copy_number_vector)/z_constant) * \
        (variant_allele_vector[0]/copy_number_vector)
    reference_pop = ((tumour_content*(1-cancer_cell_fraction)*copy_number_vector)/z_constant) * \
        (variant_allele_vector[1]/copy_number_vector)
    variant_pop = ((tumour_content*cancer_cell_fraction*copy_number_vector)/z_constant) * \
        (variant_allele_vector[2]/copy_number_vector)

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
        "mixture_coef", dist.Dirichlet(dirichlet_alpha))

    # we do not assume clusters have different SDs
    # LKJ params: d, concentration, where d is the number of clusters
    cluster_cov = pyro.sample(
        "sd", dist.InverseGamma(1, 1))

    # -------------------------------------------------------------------------------------------

    # Cluster-specific means
    with pyro.plate("k", K):
        # prior on cluster means
        cluster_means = pyro.sample(
            "mu", dist.Normal(0, 1))
        
        cluster_means = torch.special.expit(cluster_means)
        
        # # for a beta distribution, a gamma prior works. with 0.01, 0.01 makes it uninformative 
        # # the support for gamma is strictly > 0
        # beta_alpha = pyro.sample("alpha", dist.Normal(0, 1))
        # beta_beta = pyro.sample("beta", dist.Normal(0, 1))
        
        # beta_alpha = torch.special.expit(beta_alpha)
        # beta_beta = torch.special.expit(beta_beta)

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
            "cluster_assignment", dist.Categorical(mixture_coefficients))

        # our CCF is modelled with a gaussian mixture model:
        cancer_cell_fraction = pyro.sample(
            "CCF", dist.Normal(loc=cluster_means[cluster_assignment], scale=cluster_cov))

        # using expit, convert normal output to value between [0, 1] since CCF is necessarily a proportion
        cancer_cell_fraction = torch.special.expit(cancer_cell_fraction)
        
        
        # # Model CCF with a beta distribution (its support is [0, 1])
        # cancer_cell_fraction = pyro.sample("CCF", dist.Beta(beta_alpha[cluster_assignment], beta_beta[cluster_assignment]))
        

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
            "supporting_reads", fn=dist.Binomial(num_total_reads, probability_function(tumour_content, variant_allele_count, copy_number_state, cancer_cell_fraction)), obs=num_supporting_reads)

        # num_supporting_reads = pyro.sample(
        #     "supporting_reads", fn=dist.Binomial(num_total_reads, cancer_cell_fraction), obs=num_supporting_reads)

    return cluster_assignment, cancer_cell_fraction


def train_gmm(data, num_steps=10000):
    optimizer = Adam(optim_args={"lr": 0.05})  # decrease learning rate
    autoguide = AutoNormal(poutine.block(model, hide=["cluster_assignment"]), init_loc_fn=init_to_sample)

    trace = poutine.trace(model).get_trace(data)
    trace.compute_log_prob()
    print("---------- Tensor Shapes ------------")
    print(trace.format_shapes())

    print(trace.log_prob_sum())

    svi = SVI(model, autoguide, optimizer, loss=TraceEnum_ELBO())

    # save losses
    losses = []
    for step in tqdm.trange(num_steps):  # Consider running for more steps.
        loss = svi.step(data)
        losses.append(loss)
        if step % 1000 == 0 or step == num_steps:
            print(f"Step {step}/{num_steps}, Loss: {loss}")
            
    # grab the learned variational parameters
    # Get the keys of parameters in the parameter store
    param_keys = pyro.get_param_store().keys()

    # Print the keys
    print("Parameter Keys:", param_keys)
    
    mu_loc = pyro.param("AutoNormal.locs.mu")
    print(torch.special.expit((mu_loc)))
    
    ccf_loc = pyro.param("AutoNormal.locs.CCF")
    print(torch.special.expit((ccf_loc)))
    
    ccf_scales = pyro.param("AutoNormal.scales.CCF")
    print(ccf_scales)
    
    print(f"CCF values are: {torch.special.expit((ccf_loc)).detach().numpy()}")
    
    return losses, torch.special.expit((ccf_loc)).detach().numpy(), torch.special.expit((mu_loc)).detach().numpy()


def classifier(data, trained_model, temperature=0):
    inferred_model = infer_discrete(
        trained_model, temperature=temperature, first_available_dim=-2
    )  # avoid conflict with data plate

    trace = pyro.poutine.trace(inferred_model).get_trace(data)

    ccf = trace.nodes["CCF"]["value"]
    cluster = trace.nodes["cluster_assignment"]["value"]

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

    plt.savefig(f"./code_outputs/model/loss_{num_steps}.png")


def process_outputs(ccf, clusters):
    np.savetxt("./code_outputs/model/tsv/ccf.tsv", ccf, delimiter='\t')
    np.savetxt("./code_outputs/model/tsv/clusters.tsv",
               clusters, delimiter='\t')


def main():
    pyro.util.set_rng_seed(12345)
    # K clusters
    K = 10
    # -------------------------
    # df = pd.read_csv("./COLO829_somatic_forced_synthetic.tsv",
    #                  delimiter="\t", index_col=None)
    
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
    variant_allele_count = df["variant_allele_count"].to_numpy()
    variant_allele_count = [variant_allele_count*0, variant_allele_count*0, variant_allele_count]
    
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

    # ------------------------------ Run Model
    num_steps = 10000

    losses, ccfs, cluster_means = train_gmm(data, num_steps=num_steps)
    
    # Save inferred values
    df["ccf"] = ccfs

    plot_training_error(loss=losses, num_steps=num_steps)

    # ------------------------------ Render Model
    pyro.render_model(model, model_args=(data,),
                      filename="./code_outputs/model/model.png", 
                      render_params=True)

    print("Finish")

    # ------------------------------ After SVI, we want to get the estimates (e.g. mean of the posterior)
    # Get information:
    autoguide = AutoNormal(pyro.poutine.block(
        model, hide=['cluster_assignment']))
    guide_trace = pyro.poutine.trace(autoguide).get_trace(
        data)  # record the globals
    trained_model = pyro.poutine.replay(
        model, trace=guide_trace)  # replay the globals
    
    num_samples = 1000
    ccf_samples = []
    cluster_assignment_samples = []
    for sample in range(num_samples):
        guide_trace = pyro.poutine.trace(autoguide).get_trace(data)
        posterior_predictive = pyro.poutine.trace(pyro.poutine.replay(model, guide_trace)).get_trace(data)
        ccf_samples.append(torch.special.expit(posterior_predictive.nodes['CCF']['value']).detach().numpy())   
        cluster_assignment_samples.append(posterior_predictive.nodes['cluster_assignment']['value'].detach().numpy())    

    ccf_sampling_df = pd.DataFrame(ccf_samples)
    cluster_assignment_sampling_df = pd.DataFrame(cluster_assignment_samples)
    
    # Calculate the mode for each column
    cluster_id_list = cluster_assignment_sampling_df.mode().iloc[0].tolist()
    cluster_id_list = list(map(int, cluster_id_list))
    
    print(cluster_means)
    # Save inferred values
    df["cluster_id"] = cluster_id_list
    df['cluster_mean'] = df['cluster_id'].map(lambda x: cluster_means[x])
    
    # Writing to a CSV file
    df.to_csv('./code_outputs/model/tsv/df_with_ccf.tsv', index=False, sep="\t")

    print("hello")

if __name__ == "__main__":
    main()
