# MasterThesisDemoCrossNetwork
MasterThesisDemoCrossNetwork
A demo of the code used during my masterthesis project

Required packages:

    numpy
    tensforflow
    scipy
    networkx
    pandas
    matplotlib
    seaborn

In order to run this code. Make sure you have imported the required data from:

https://drive.google.com/file/d/1BdVfoH77G1X0gyBPVBfScMR0iGVBOr7W/view?usp=sharing and,

https://drive.google.com/file/d/1b8BWx_EReYlWgbMleP_pExbJGHGzBVJd/view?usp=sharing

These directories were to big to be imported to github, but include the Data and temporary pickle directory which include the communication data set.

We will be making use of the Enron data set as well as an open data set provided by stanford concerning the email communication of a EU commission. https://snap.stanford.edu/data/email-Eu-core.html. The datasets are provided in the data set linked to above. 
After these directories are imported to the root of the project and the required packages are installed, the networks can be created by running:

the Create_data_sets.py, Create_network.py, Create_input_features.py, and Proprocess_labels.py files in the corresponding order, which can be fond in the Preprocessing_Enron directory.

Additionally, the EU data sets needs to be pre-processed which can be done by executing the following files:

Create_network_data_eu.py, Create_network.py, Create_input_features.py, and Preprocess_labels_py.

These files will create the required network files using Networkx. These files might take considerable time to run, all generated files are therefore also included in the directory posted above which can be retrieved from google drive. 

The ACDNE directory needs to be used in order to make the embeddings using the ACDNE model, in order to do so, execute the "ACDNE_embeddings_enron_eu.py" file. 

Once the networks are made the different embeddings techniques can be evaluated by using the "evaluate_embeddings.py" file. The parameters are to be provide in the beginning of the file, where an example is given.

In order to validate GraRep, GraphWave, DeepWalk, Role2Vec, Node2Vec, or ASNE make sure the "embeddings_known" has been set to false False as the embeddings will be generated.

An example coniguration to verify GraRep is shown below the model does have to be speficied as the embeddings are generated in the provided code.

model = GraRep()
has_attributes = False
embeddings_known = False
ACDNE_used = False
if ACDNE_used:
    location_embeddings_acdne = "../Data/enron_euemb.mat"
if has_attributes:
    with open('../pickle_temporary_data/attr_matrix_enron.pickle',
            'rb') as f:
        attr_matrix = pickle.load(f)
location_results = "../Results/GraRep.txt"
location_embedding_network1 = "../Data/embedding_grarep_network1.npy"
location_embedding_network2 = "../Data/embedding_grarep_network2.npy"

The same structure can be applied for GraphWave, DeepWalk, Role2Vec, and Node2Vec. ASNE is almost similar yet "has_attributes" should be True

The resulting embeddings and its location should be provided to the "location_embeddings" variable in the "evaluated_embeddings.py" script.

When verifying the application of DRNE, make sure the use the edgelist file generated in the "Create_node_features.py" file to generate embeddings using the original implementation of DRNE as provided on https://github.com/tadpole/DRNE example usage is provided in their readme. The edgelist files are also provide in the data set provided via google drive as mentioned above. And make sure the correct location for the embeddings of both networks provided.

An example coniguration to verify DRNE is shown below the model does not have to be speficied as the embeddings are generated using the original implementation.

model = None
has_attributes = False
embeddings_known = True
ACDNE_used = False
if ACDNE_used:
    location_embeddings_acdne = "../Data/enron_euemb.mat"
if has_attributes:
    with open('../pickle_temporary_data/attr_matrix_enron.pickle',
            'rb') as f:
        attr_matrix = pickle.load(f)
location_results = "../Results/Drne.txt"
location_embedding_network1 = "../Data/embedding_Drne_network1.npy"
location_embedding_network2 = "../Data/embedding_Drne_network2.npy"

The ACDNE model uses a similar configuration. Yet slightly differently as the ACDNE model generates .mat files and combines the embeddings of both networks into one. An example configuration to verify ACDNE is provided below. 

model = None
has_attributes = False
embeddings_known = True
ACDNE_used = True
if ACDNE_used:
    location_embeddings_acdne = "../Data/enron_euemb.mat"
if has_attributes:
    with open('../pickle_temporary_data/attr_matrix_enron.pickle',
            'rb') as f:
        attr_matrix = pickle.load(f)
location_results = "../Results/ACDNE.txt"
location_embedding_network1 = None
location_embedding_network2 = None


The "location_results" is the location where the results will be posted, make sure the file extension is set to ".txt".

The Has_attributes variable should only be set to TRUE when the embeddings technique being used is ASNE()

The key_actor_list should be used to include all actors for which you would like to validate similar nodes. Currently this has been set to all high management in the Enron dataset.

The evaluation of the embeddings takes as input the metrics which need to be verified. For demo purposes this has been set to ["eigenvector", "pagerank"] as betweenness centrality takes a lot more time to process. If you wish to verify the results using betweenness centrality as well simply change the input of the evaluate_embeddings to ["eigenvector", "pagerank", "betweenness"]

The evaluate_embeddings_example_key_actors function also takes into account the number of embeddings and the top n percent. During my research I used top 10 % overlap, which means the top_n_embeddings have been set to 10 percent of the number of nodes and the top_n_percent has been set to 10. However, this function does allow for a variety of combination to be applicable.
