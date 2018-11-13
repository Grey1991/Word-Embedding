import os
import sys
import numpy as np
from os import walk


def read_dic(dic_path):
    dev_adjectives = []
    Synonyms = {}
    for (dirpath, dirnames, filenames) in walk(dic_path):
        dev_adjectives.extend(filenames)
    for filename in dev_adjectives:
        with open(os.path.join(dic_path, filename), 'r') as infile:
            syn = [line.strip() for line in infile]
        Synonyms[filename] = syn
    return Synonyms, dev_adjectives


def Compute_Hits(adjective, output_list, Synonyms):
    synonym_list = Synonyms.get(adjective)
    result = 0.0
    for word in output_list:
        if (word in synonym_list):
            result = result + 1
    return result

if __name__ == '__main__':
    dic_path = "./dev_set"
    if len(sys.argv) > 1:
        num_steps = int(sys.argv[1]) # Reads num_steps from command-line parameters in Dockerfile.
    else:
        print('Please Enter value for (num_steps) in the Dockerfile')
        exit(0)

    import submission as submission

    #######################################################
    ## Part 1:
    data_file = None
    part1 = None
    print('Processing Raw Data')
    data_file = submission.process_data('./BBC_Data.zip')
    if(os.path.isfile(data_file)):
        print('Writing Processed Data file(Success)\n')
        part1 = 1
    else:
        print('Writing Processed Data file(Failed)\n')

    #######################################################
    ## Part 2:
    part2 = None
    emb_file = 'adjective_embeddings.txt'
    embedding_dim = 200
    print('Training model with %d iterations '%(num_steps))
    submission.adjective_embeddings(data_file, emb_file, num_steps, embedding_dim)
    if(os.path.isfile('./adjective_embeddings.txt')):
        print('Writing Embedding file(Success)\n')
        part2 = 1
    else:
        print('Writing Embedding file(Failed)\n')

    #######################################################
    ## Part 3:
    part3 = None
    top_k = 100
    model_file = './adjective_embeddings.txt'
    Synonyms, dev_adjectives = read_dic('./dev_set')
    print('Reading Trained Model')
    if (os.path.isfile('./adjective_embeddings.txt')):
        total_hits = []
        for adjective in dev_adjectives:
            output_list = submission.Compute_topk(model_file, adjective, top_k)
            hits = Compute_Hits(adjective, output_list, Synonyms)
            total_hits.append(hits)
            result = np.average([x for x in total_hits])
            part3 = 1
        print('Reading Trained Model(Success)')
        print('Average Hits on Dev Set is = %f ' %(result))
    else:
        print('Reading Trained Model(Failure)\n')
    #######################################################
    result = [part1, part2, part3]
    for item in result:
        if item == None:
            print('Error Please check your code')


