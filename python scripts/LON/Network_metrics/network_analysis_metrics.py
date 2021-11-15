# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:49:04 2021

@author: mk633
"""
# Network Analysis with metrics
import networkx as nx 
#from networkx.algorithm import community
from operator import itemgetter
import matplotlib.pyplot as plt
from statistics import mean,stdev

#%%  ILK RUN

##########################   number of LO
num_nodes = []
num_nodes.append(len(fvals_list))


##########################   network density
density= []
density.append(nx.density(G)) # the ratio of actual edges in the network to all possible edges in the network

################################################## average in degree/ outdegree
nnodes = G.number_of_nodes()
degrees_in = [d for n, d in G.in_degree()]
degrees_out = [d for n, d in G.out_degree()]
in_deg = []
avrg_degree_in = sum(degrees_in) / float(nnodes)
in_deg.append(avrg_degree_in)


out_deg = []
avrg_degree_out = sum(degrees_out) / float(nnodes)
out_deg.append(avrg_degree_out)


#in_values = sorted(set(degrees_in))
#in_hist = [degrees_in.count(x) for x in in_values]
#out_values = sorted(set(degrees_out))
#out_hist = [degrees_out.count(x) for x in out_values]
 
#plt.figure()
#plt.plot(in_values,in_hist,'ro-') # in-degree
#plt.plot(out_values,out_hist,'bo-') # out-degree
#plt.legend(['In-degree','Out-degree'])
#plt.xlabel('Degree')
#plt.ylabel('Number of nodes')
#plt.title(' network')
#plt.close()

########################################################## degree centrality
# The degree centrality is measure for finding the importance of a node in a network.
centrality = []
c = sorted(nx.degree_centrality(G).items(), key=lambda x : x[1], reverse=True)[:5]
centrality.append(c[0][1])  

#######################################################  shortest path length
#for i in range(len(fvals_list)):
    #if fvals_list[i]==min(fvals_list):
        #ind = i
#nx.shortest_path_length(G, target=ind)
#print(list(nx.shortest_paths.all_pairs_dijkstra_path_length(G))[1])

######################################################### degree assortativity
# assortativity coefficient is the Pearson correlation coefficient of degree between pairs of linked nodes
#connected nodes share similar properties. tend to have similar fitness values by attractin the edges
# quantifies the tendency of nodes being connected to similar nodes in a complex network.
assortativity =[]
assortativity.append(nx.degree_assortativity_coefficient(G)) 



#%%  SONRAKI RUNLAR

num_nodes.append(len(fvals_list))

density.append(nx.density(G)) # the ratio of actual edges in the network to all possible edges in the network

degrees_in = [d for n, d in G.in_degree()]
avrg_degree_in = sum(degrees_in) / float(nnodes)
in_deg.append(avrg_degree_in)

degrees_out = [d for n, d in G.out_degree()]
avrg_degree_out = sum(degrees_out) / float(nnodes)
out_deg.append(avrg_degree_out)

c = sorted(nx.degree_centrality(G).items(), key=lambda x : x[1], reverse=True)[:5]
centrality.append(c[0][1]) 

assortativity.append(nx.degree_assortativity_coefficient(G)) 


#%%  ORTALAMA ZAMANI
mean_num_nodes =mean(num_nodes)
sd_num_nodes =stdev(num_nodes)

mean_density =mean(density)
sd_density=stdev(density)

mean_in_deg =mean(in_deg)
sd_in_deg=stdev(in_deg)

mean_out_deg=mean(out_deg)
sd_out_deg=stdev(out_deg)

mean_centrality=mean(centrality)
sd_centrality=stdev(centrality)

average_shortest_path_go = []

mean_assortativity=mean(assortativity)
sd_assortativity=stdev(assortativity)

#%%  IGNORE TAMAMEN
nx.clustering(G)
    

nx.is_strongly_connected(G)
nx.is_weakly_connected(G)  #This implies that our network is made up of more than one components, i.e., connected subgraphs of our network





#####################  breath search bak
nx.draw_circular(nx.breadth_first_search.bfs_tree(G, 4), with_labels=True, node_color="lime", font_color="red")





mean(bh)
stdev(bh)






















 

############################  
diameter = diameter(G, e=None) # give you a sense of the network’s overall size, the distance from one end of the network to another.
diameter = nx.diameter(G)
# If your Graph has more than one component, this will return False:
print(nx.is_connected(G))
nx.radius(G)
nx.is_connected(G)

path = []
for i in range(len(fvals_list)):
    print(i)
    path.append(nx.has_path(G,i,int(np.where(min(fvals_list)))))
# Next, use nx.connected_components to get the list of components,
# then use the max() command to find the largest one:
components = nx.connected_components(G)
largest_component = max(components, key=len)


# Create a "subgraph" of just the largest component
# Then calculate the diameter of the subgraph, just like you did with density.
#

subgraph = G.subgraph(largest_component)
diameter = nx.diameter(subgraph)
print("Network diameter of largest component:", diameter)






averageclustering = nx.average_clustering(G, nodes=None, weight=True, count_zeros=True)

clustering = nx.clustering(G, nodes=None, weight=True)
triadic_closure = nx.transitivity(G)
print("Triadic closure:", triadic_closure) #Transitivity allows you a way of thinking about all the relationships in your graph that may exist but currently do not.
# Transitivity is the ratio of all triangles over all possible triangles.




#The nodes with the highest degree in a social network are the people who know the most people. These nodes are often referred to as hubs, and calculating degree is the quickest way of identifying hubs.
degree_dict = dict(G.degree(G.nodes()))
nx.set_node_attributes(G, degree_dict, 'degree')
sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)





# Modularity is a measure of relative density in your network: a community (called a module or modularity class) has high density relative to other nodes within its module but low density with those outside. Modularity gives you an overall score of how fractious your network is, and that score can be used to partition the network and return the individual communities
communities = community.greedy_modularity_communities(G)
modularity_dict = {} # Create a blank dictionary
for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
    for name in c: # Loop through each person in a community
        modularity_dict[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.

# Now you can add modularity information like we did the other metrics
nx.set_node_attributes(G, modularity_dict, 'modularity')








